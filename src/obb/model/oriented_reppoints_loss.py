import torch
import torch.nn as nn
from typing import Dict
from einops import rearrange
from kornia.losses import focal_loss

from obb.model.custom_model import DetectionModel
from obb.utils.polygon import convex_hull, polygon_intersection, polygon_area, polygon_iou
from obb.utils.box_ops import out_of_box_distance

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OBBPointAssigner:
    """
    Assign a corresponding oriented gt box or background to each point.

    Source:
        https://github.com/LiWentomng/OrientedRepPoints/blob/main/mmdet/core/bbox/assigners/oriented_point_assigner.py

    Each proposal will be assigned with `0`, or a positive integer indicating the ground truth index.
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    """

    def __init__(self, scale=4, pos_num=1):
        self.scale = scale
        self.pos_num = pos_num

    @staticmethod
    def _obb_to_ob(obb_xs, obb_ys):
        gt_xmin, _ = obb_xs.min(1)
        gt_ymin, _ = obb_ys.min(1)
        gt_xmax, _ = obb_xs.max(1)
        gt_ymax, _ = obb_ys.max(1)
        return torch.cat([gt_xmin[:, None], gt_ymin[:, None], gt_xmax[:, None], gt_ymax[:, None]], dim=1)

    def assign(self, points, gt_obboxes, gt_labels=None):
        """
        Assign oriented gt boxes to points.

        This method assign a gt obox to every points set, each points set will be assigned with 0, or a positive number.
        0 means negative sample, positive number is the index (1-based) of assigned gt.

        The assignment is done in following steps (the order matters):
            1. assign every points to 0
            2. A point is assigned to some gt bbox if:
                (i) the point is within the k closest points to the gt bbox
                (ii) the distance between this point and the gt is smaller than other gt bboxes
        Args:
            points (Tensor): points to be assigned, shape(n, 3) while last dimension stands for (x, y, stride).
            gt_obboxes (Tensor): groundtruth oriented boxes, shape (k, 8).
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).
        Returns:
            :obj:`AssignResult`: The assign results.
        """

        scale = self.scale
        num_points = points.shape[0]

        if gt_obboxes.ndim == 1:
            # If no truth assign everything to the background
            assigned_gt_inds = points.new_full((num_points,), 0, dtype=torch.long)
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = points.new_zeros((num_points,), dtype=torch.long)
            return assigned_gt_inds, assigned_labels

        points_xy = points[:, :2]
        points_stride = points[:, 2]
        points_lvl = torch.log2(points_stride).int()  # [3...,4...,5...,6...,7...]
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()

        # assign gt rbox
        gt_bboxes = self._obb_to_ob(obb_xs=gt_obboxes[:, 0::2], obb_ys=gt_obboxes[:, 1::2])
        gt_bboxes_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
        gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min=1e-6)
        gt_bboxes_lvl = ((torch.log2(gt_bboxes_wh[:, 0] / scale) + torch.log2(gt_bboxes_wh[:, 1] / scale)) / 2).int()
        gt_bboxes_lvl = torch.clamp(gt_bboxes_lvl, min=lvl_min, max=lvl_max)

        # stores the assigned gt index of each point
        assigned_gt_inds = points.new_zeros((num_points,), dtype=torch.long).to(device)

        # stores the assigned gt dist (to this point) of each point
        assigned_gt_dist = points.new_full((num_points,), float('inf'))
        points_range = torch.arange(points.shape[0])

        for idx in range(gt_obboxes.shape[0]):
            gt_lvl = gt_bboxes_lvl[idx]

            # get the index of points in this level
            lvl_idx = gt_lvl == points_lvl
            points_index = points_range[lvl_idx]

            lvl_points = points_xy[lvl_idx, :]
            gt_center_point = gt_bboxes_xy[[idx], :].to(device)
            gt_wh = gt_bboxes_wh[[idx], :].to(device)

            # compute the distance between gt center and all points in this level
            points_gt_dist = ((lvl_points - gt_center_point) / gt_wh).norm(dim=1)

            # find the nearest k points to gt center in this level
            min_dist, min_dist_index = torch.topk(points_gt_dist, self.pos_num, largest=False)

            # the index of nearest k points to gt center in this level
            min_dist_points_index = points_index[min_dist_index]

            # The less_than_recorded_index stores the index of min_dist that is less than the assigned_gt_dist,
            # assigned_gt_dist stores the dist from previous assigned gt (if exist) to each point.
            less_than_recorded_index = min_dist < assigned_gt_dist[min_dist_points_index]

            # The min_dist_points_index stores the index of points satisfies:
            #   (1) it is k nearest to current gt center in this level.
            #   (2) it is closer to current gt center than other gt center.
            min_dist_points_index = min_dist_points_index[less_than_recorded_index]

            # assign the result
            assigned_gt_inds[min_dist_points_index] = idx + 1
            assigned_gt_dist[min_dist_points_index] = min_dist[less_than_recorded_index]

        if gt_labels is not None:
            gt_labels = gt_labels.to(device)
            assigned_labels = assigned_gt_inds.new_zeros((num_points,)).to(device)
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze().to(device)
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return assigned_gt_inds, assigned_labels


class OBBMaxIoUAssigner:
    """
    Assign a corresponding gt oriented bounding box or background to each oriented reppoints.

    Each oriented points will be assigned with `0`, or a positive integer - index (1-based) of assigned gt.
    Args:
        pos_iou_thr (float): IoU threshold for positive obboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative obboxes.
        min_pos_iou (float): Minimum iou for a obbox to be considered as a positive bbox. Positive samples can
                             have smaller IoU than pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all obboxes with the same highest overlap with some gt to that gt.

    """

    def __init__(self,
                 pos_iou_thr=0.1,
                 neg_iou_thr=0.1,
                 min_pos_iou=.0,
                 gt_max_assign_all=True):

        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all

    @staticmethod
    def convex_overlaps(gt_obb, rep_points):
        # rep_points_hull = [convex_hull(points.reshape(-1, 2)) for points in rep_points]
        rep_points_hull = convex_hull(rep_points)

        iou_matrix_rows = []
        for gt_points in gt_obb:
            iou_row = torch.stack(
                [polygon_iou(gt_points.reshape(-1, 2), hull_points) for hull_points in rep_points_hull]
            )
            iou_matrix_rows.append(iou_row)

        iou_matrix = torch.stack(iou_matrix_rows, dim=0)  # [num_gts_obb, num_rep_points_obb]
        return iou_matrix

    def assign(self, points, gt_obb, gt_labels=None):
        overlaps = self.convex_overlaps(gt_obb, points[:, :-1])  # Without stride
        num_gts, num_rep_points_obb = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_rep_points_obb,), -1, dtype=torch.long)

        if num_gts == 0 or num_rep_points_obb == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_rep_points_obb,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_zeros((num_rep_points_obb,), dtype=torch.long)
            return assigned_gt_inds, assigned_labels

        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0]) & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. assign fg: for each gt, proposals with highest oriented IoU
        for i in range(num_gts):
            if gt_max_overlaps[i] >= self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                    assigned_gt_inds[max_iou_inds] = i + 1
                else:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_rep_points_obb,))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return assigned_gt_inds, assigned_labels


def giou_loss(gt_points, pred_points):
    gt_points = gt_points.to(device)

    intersection_points = polygon_intersection(gt_points, pred_points)
    poly1_area = polygon_area(pred_points)
    poly2_area = polygon_area(gt_points)
    intersection_area = polygon_area(intersection_points)
    union_area = poly1_area + poly2_area - intersection_area
    iou = intersection_area / (union_area + 1e-16)

    all_points = torch.cat([gt_points, pred_points])
    all_points_area = polygon_area(convex_hull(all_points))
    giou = iou - (all_points_area - union_area) / (all_points_area + 1e-16)
    return 1 - giou


def out_of_box_loss(gt_points, pred_points):
    dists = out_of_box_distance(pred_points, gt_points)
    return torch.mean(dists)


class OrientedRepPointsLoss(nn.Module):
    def __init__(self, strides):
        super().__init__()

        self.strides = strides
        self.feature_maps_names = list(self.strides.keys())

        # balance parameters between components of the loss. values from oriented rep points config file.
        self.classification_weight = 1.
        self.init_localization_weight = 0.3
        self.refine_localization_weight = 1.
        self.init_spatial_constraint_weight = 0.05
        self.refine_spatial_constraint_weight = 0.1

        self.init_assigner = OBBPointAssigner()
        # TODO switch to OBBMaxIoUAssigner ASAP
        self.refine_assigner = OBBPointAssigner()

    def _flatten_head_output(self, raw_rep_points_init, raw_rep_points_refine, raw_classification):
        """
        Convert the output of Oriented Rep Points head from dicts to tensors for loss calculation.

        :param raw_rep_points_init: rep points from initialization stage
                {'P2': shape[b xy w h], ...}
        :param raw_rep_points_refine: rep points from refinement step
                {'P2': shape[b xy w h], ...}
        :param raw_classification: rep points classification
                {'P2': shape[b cls w h], ...}
        :return:
                rep_points_init (torch.tensor): shape [..., xy]
                rep_points_refine (torch.tensor): shape [..., xy]
                classification (torch.tensor): shape [..., cls]
        """
        multi_level_rep_points_init = []
        multi_level_rep_points_refine = []
        multi_level_classification = []
        for feature_map in self.feature_maps_names:
            # get rep points of the current feature map
            curr_level_rep_points_init = raw_rep_points_init[feature_map]
            curr_level_rep_points_refine = raw_rep_points_refine[feature_map]
            curr_level_classification = raw_classification[feature_map]

            # flatten the rep points and classification tensor
            multi_level_rep_points_init.append(rearrange(curr_level_rep_points_init, 'b xy w h -> (b w h) xy'))
            multi_level_rep_points_refine.append(rearrange(curr_level_rep_points_refine, 'b xy w h -> (b w h) xy'))
            multi_level_classification.append(rearrange(curr_level_classification, 'b cls w h -> (b w h) cls'))

        # concat multi-level features
        rep_points_init = torch.cat(multi_level_rep_points_init, dim=0)
        rep_points_refine = torch.cat(multi_level_rep_points_refine, dim=0)
        classification = torch.cat(multi_level_classification, dim=0)

        return rep_points_init, rep_points_refine, classification

    def _get_centers(self, raw_rep_points_init, raw_rep_points_refine):
        """
        Find ground truth rep points centers for the point assigners, in format [..., x y stride].

        :param raw_rep_points_init: rep points from initialization stage
                {'P2': shape[b xy w h], ...}
        :param raw_rep_points_refine: rep points from refinement step
                {'P2': shape[b xy w h], ...}
        :return:
                centers_init (torch.tensor): shape (..., 3)
                centers_refine (torch.tensor): shape (..., 3)
        """
        multi_level_centers_init = []
        multi_level_centers_refine = []
        for feature_map in self.feature_maps_names:
            # get the (xy) center of the rep points for the current feature map
            centers1 = raw_rep_points_init[feature_map][:, 8:10, :, :]
            centers2 = raw_rep_points_refine[feature_map][:, 8:10, :, :]

            centers1_flattened = rearrange(centers1, 'b xy w h -> (b w h) xy')
            centers2_flattened = rearrange(centers2, 'b xy w h -> (b w h) xy')

            stride = self.strides[feature_map]
            stride_vec = stride * torch.ones(centers1_flattened.shape[0], 1).to(device)

            multi_level_centers_init.append(torch.cat([centers1_flattened, stride_vec], dim=1))  # [..., 3])
            multi_level_centers_refine.append(torch.cat([centers2_flattened, stride_vec], dim=1))  # [..., 3])

        # concat multi-level features
        centers_init = torch.cat(multi_level_centers_init, dim=0)
        centers_refine = torch.cat(multi_level_centers_refine, dim=0)

        return centers_init, centers_refine

    def _box_regression_loss(self, rep_points_init, gt_obb, assigned_gt_idxs_init, assigned_labels_init):
        if gt_obb.ndim == 1:
            return torch.Tensor([0, 0])

        # initialization stage assigner
        positive_samples_idx_init = torch.where(assigned_labels_init > 0)
        positive_assigned_gt_idxs_init = assigned_gt_idxs_init[positive_samples_idx_init]
        positive_rep_points_init = rep_points_init[positive_samples_idx_init]

        # initialize losses with 0
        localization_loss = torch.zeros(1, dtype=torch.float, device=device)
        spatial_constraint_loss = torch.zeros(1, dtype=torch.float, device=device)

        # iterate over positive samples and add its loss to the total loss
        for i, pred_points in enumerate(positive_rep_points_init):
            pred_points = pred_points.reshape(-1, 2)

            # ground truth obb for the current rep points
            gt_box_idx = int(positive_assigned_gt_idxs_init[i]) - 1
            gt_points = gt_obb[gt_box_idx].reshape(-1, 2).float().to(device)
            gt_points.requires_grad = False
            pred_points_convex_hull = convex_hull(pred_points)
            localization_loss += giou_loss(gt_points, pred_points_convex_hull)
            spatial_constraint_loss += out_of_box_loss(gt_points, pred_points)

        # divide by number of positive samples
        if len(positive_rep_points_init) > 0:
            localization_loss /= len(positive_rep_points_init)
            spatial_constraint_loss /= len(positive_rep_points_init)

        return localization_loss, spatial_constraint_loss

    def get_loss(
            self,
            raw_rep_points_init: Dict[str, torch.Tensor],
            raw_rep_points_refine: Dict[str, torch.Tensor],
            raw_classification: Dict[str, torch.Tensor],
            gt_obb: torch.Tensor,
            gt_labels: torch.Tensor
    ):
        """
        Loss function for Oriented Rep Points.

        :param raw_rep_points_init: rep points from initialization stage
                {'P2': shape[b xy w h], ...}
        :param raw_rep_points_refine: rep points from refinement step
                {'P2': shape[b xy w h], ...}
        :param raw_classification: rep points classification
                {'P2': shape[b cls w h], ...}
        :param gt_obb: ground truth oriented bounding box
                [[x1, y1, x2, y2, x3, y3, x4, y4], [...]]
        :param gt_labels: ground truth label for each bounding box
                [bbox1_cls, bbox2_cls, ...]
        :return:
        """
        # convert the output of Rep Points head to flattened representation
        rep_points_init, rep_points_refine, classification = self._flatten_head_output(
            raw_rep_points_init, raw_rep_points_refine, raw_classification
        )
        centers_init, centers_refine = self._get_centers(raw_rep_points_init, raw_rep_points_refine)

        # initialization step
        assigned_gt_idxs_init, assigned_labels_init = self.init_assigner.assign(centers_init, gt_obb, gt_labels)
        classification_loss = focal_loss(classification, assigned_labels_init, alpha=0.25, gamma=2, reduction='mean')
        classification_loss *= self.classification_weight
        localization_init_loss, spatial_constraint_init_loss = self._box_regression_loss(
            rep_points_init, gt_obb, assigned_gt_idxs_init, assigned_labels_init
        )
        box_regression_init_loss = (self.init_localization_weight * localization_init_loss +
                                    self.init_spatial_constraint_weight * spatial_constraint_init_loss)

        # refinement_step
        assigned_gt_idxs_refine, assigned_labels_refine = self.refine_assigner.assign(centers_refine, gt_obb, gt_labels)
        localization_refine_loss, spatial_constraint_refine_loss = self._box_regression_loss(
            rep_points_refine, gt_obb, assigned_gt_idxs_refine, assigned_labels_refine
        )
        box_regression_refine_loss = (self.refine_localization_weight * localization_refine_loss +
                                      self.refine_spatial_constraint_weight * spatial_constraint_refine_loss)

        # ReLU to solve divergent loss + ignore NaNs
        if type(classification_loss) != int:
            classification_loss = torch.relu(classification_loss)
            if torch.isnan(classification_loss):
                classification_loss = torch.zeros(1).to(device)
        if type(box_regression_init_loss) != int:
            box_regression_init_loss = torch.relu(box_regression_init_loss)
            if torch.isnan(box_regression_init_loss):
                box_regression_init_loss = torch.zeros(1).to(device)
        if type(box_regression_refine_loss) != int:
            box_regression_refine_loss = torch.relu(box_regression_refine_loss)
            if torch.isnan(box_regression_refine_loss):
                box_regression_refine_loss = torch.zeros(1).to(device)

        # Classification precision (TODO rename as recall)
        pos_idxs = torch.where(assigned_labels_init > 0)
        neg_idxs = torch.where(assigned_labels_init == 0)
        assigned_labels_init_pos = assigned_labels_init[pos_idxs]
        classification_hard = torch.argmax(classification, dim=1)
        classification_pos_hard = classification_hard[pos_idxs]
        classification_neg_hard = classification_hard[neg_idxs]
        p = len(assigned_labels_init_pos)
        n = len(assigned_labels_init) - p
        tp = torch.sum(classification_pos_hard == assigned_labels_init_pos)
        tn = torch.sum(classification_neg_hard == 0)
        precision_pos = tp / p if p != 0 else torch.Tensor([0]).to(device)
        precision_neg = tn / n if n != 0 else torch.Tensor([0]).to(device)

        return (classification_loss + box_regression_init_loss + box_regression_refine_loss,
                classification_loss, box_regression_init_loss, box_regression_refine_loss, precision_pos, precision_neg)


if __name__ == '__main__':
    model = DetectionModel().to(device)
    img_in = torch.rand(1, 3, 256, 256).to(device)

    # fake ground truth data
    gt_labels_ = torch.tensor([3, 1]).to(device)
    gt_obboxes_ = torch.tensor([[1, 1, 1, 10, 10, 10, 10, 1],
                                [10, 10, 10, 50, 50, 50, 50, 10]]).to(device)

    # fake empty ground truth data
    # gt_labels_ = torch.tensor([])
    # gt_obboxes_ = torch.tensor([])

    rep_points_init_, rep_points_refine_, classification_ = model(img_in)

    rep_points_loss = OrientedRepPointsLoss(strides=model.feature_map_strides)
    loss, cls_loss, reg_init_loss, reg_refine_loss, precision_pos, precision_neg = rep_points_loss.get_loss(
        rep_points_init_, rep_points_refine_, classification_, gt_obboxes_, gt_labels_)

    # print(torch.max(classification_['P5'], dim=1))
    print(loss, cls_loss, reg_init_loss, reg_refine_loss, precision_pos, precision_neg, sep='\n')

    torch.autograd.set_detect_anomaly(True)

    learning_rate = 1e-3
    weight_decay = 5e-5

    # select optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        amsgrad=True
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
