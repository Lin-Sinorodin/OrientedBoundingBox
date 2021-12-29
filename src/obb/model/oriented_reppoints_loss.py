import torch
import torch.nn as nn
from typing import Dict
from einops import rearrange
from kornia.losses import focal_loss

from obb.model.custom_model import DetectionModel
from obb.utils.polygon import convex_hull, polygon_intersection, polygon_area

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OBBPointAssigner:
    """
    Assign a corresponding oriented gt box or background to each point.

    Source:
        https://github.com/LiWentomng/OrientedRepPoints/blob/main/mmdet/core/bbox/assigners/oriented_point_assigner.py

    Each proposals will be assigned with `0`, or a positive integer indicating the ground truth index.
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
        assert gt_obboxes.size(1) == 8, 'gt_obboxes should be (N * 8)'

        scale = self.scale
        num_points = points.shape[0]
        num_gts = gt_obboxes.shape[0]

        if num_gts == 0 or num_points == 0:
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
        assigned_gt_inds = points.new_zeros((num_points,), dtype=torch.long)

        # stores the assigned gt dist (to this point) of each point
        assigned_gt_dist = points.new_full((num_points,), float('inf'))
        points_range = torch.arange(points.shape[0])

        for idx in range(num_gts):
            gt_lvl = gt_bboxes_lvl[idx]

            # get the index of points in this level
            lvl_idx = gt_lvl == points_lvl
            points_index = points_range[lvl_idx]

            lvl_points = points_xy[lvl_idx, :]
            gt_center_point = gt_bboxes_xy[[idx], :]
            gt_wh = gt_bboxes_wh[[idx], :]

            # compute the distance between gt center and all points in this level
            points_gt_dist = ((lvl_points - gt_center_point) / gt_wh).norm(dim=1)

            # find the nearest k points to gt center in this level
            min_dist, min_dist_index = torch.topk(points_gt_dist, self.pos_num, largest=False)

            # the index of nearest k points to gt center in this level
            min_dist_points_index = points_index[min_dist_index]

            # The less_than_recorded_index stores the index of min_dist that is less then the assigned_gt_dist,
            # assigned_gt_dist stores the dist from previous assigned gt (if exist) to each point.
            less_than_recorded_index = min_dist < assigned_gt_dist[min_dist_points_index]

            # The min_dist_points_index stores the index of points satisfy:
            #   (1) it is k nearest to current gt center in this level.
            #   (2) it is closer to current gt center than other gt center.
            min_dist_points_index = min_dist_points_index[less_than_recorded_index]

            # assign the result
            assigned_gt_inds[min_dist_points_index] = idx + 1
            assigned_gt_dist[min_dist_points_index] = min_dist[less_than_recorded_index]

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_points,))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return assigned_gt_inds, assigned_labels


def giou_loss(gt_points, pred_points):
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


# TODO
def out_of_box_loss(gt_points, pred_points):
    return 0


class OrientedRepPointsLoss(nn.Module):
    def __init__(self, strides):
        super().__init__()

        self.strides = strides
        self.feature_maps_names = list(self.strides.keys())

        # balance parameters between components of the loss. values from oriented rep points config file.
        self.init_localization_weight = 0.3
        self.refine_localization_weight = 1.
        self.init_spatial_constraint_weight = 0.05
        self.refine_spatial_constraint_weight = 0.1

        self.init_assigner = OBBPointAssigner()
        self.refine_assigner = ...  # TODO

    def _flatten_head_output(self, raw_rep_points_init, raw_rep_points_refine, raw_classification):
        """Convert the output of Oriented Rep Points head from dicts to tensors for loss calculation"""
        multi_level_rep_points_init = {}
        multi_level_rep_points_refine = {}
        multi_level_classification = {}
        for feature_map in self.feature_maps_names:
            # get rep points of the current feature map
            curr_rep_points1 = raw_rep_points_init[feature_map]
            curr_rep_points2 = raw_rep_points_refine[feature_map]
            curr_classification = raw_classification[feature_map]

            # flatten the rep points and classification tensor
            multi_level_rep_points_init[feature_map] = rearrange(curr_rep_points1, 'b xy w h -> (b w h) xy')
            multi_level_rep_points_refine[feature_map] = rearrange(curr_rep_points2, 'b xy w h -> (b w h) xy')
            multi_level_classification[feature_map] = rearrange(curr_classification, 'b cls w h -> (b w h) cls')

        # concat multi-level features
        rep_points_init = torch.cat(list(multi_level_rep_points_init.values()), dim=0)
        rep_points_refine = torch.cat(list(multi_level_rep_points_refine.values()), dim=0)
        classification = torch.cat(list(multi_level_classification.values()), dim=0)

        return rep_points_init, rep_points_refine, classification

    def _get_centers(self, raw_rep_points_init, raw_rep_points_refine):
        """Find ground truth rep points centers for the point assigners, in format [..., x y stride]."""
        multi_level_centers_init = {}
        multi_level_centers_refine = {}
        for feature_map in self.feature_maps_names:
            # get the center of the rep points for the current feature map
            centers1 = raw_rep_points_init[feature_map][:, 8:10, :, :]
            centers2 = raw_rep_points_refine[feature_map][:, 8:10, :, :]

            centers1_flattened = rearrange(centers1, 'b xy w h -> (b w h) xy')
            centers2_flattened = rearrange(centers2, 'b xy w h -> (b w h) xy')
            stride_vec = self.strides.get(feature_map) * torch.ones(centers1_flattened.shape[0], 1).to(device)
            multi_level_centers_init[feature_map] = torch.cat([centers1_flattened, stride_vec], dim=1)  # [..., 3]
            multi_level_centers_refine[feature_map] = torch.cat([centers2_flattened, stride_vec], dim=1)  # [..., 3]

        # concat multi-level features
        centers_init = torch.cat(list(multi_level_centers_init.values()), dim=0)
        centers_refine = torch.cat(list(multi_level_centers_refine.values()), dim=0)

        return centers_init, centers_refine

    def _initialization_step_loss(self, rep_points_init, gt_obb, assigned_gt_idxs_init, assigned_labels_init):
        # initialization stage assigner
        positive_samples_idx_init = torch.where(assigned_labels_init > 0)
        positive_assigned_gt_idxs_init = assigned_gt_idxs_init[positive_samples_idx_init]
        positive_rep_points_init = rep_points_init[positive_samples_idx_init]

        # initialize losses with 0
        localization_loss = torch.zeros(1, dtype=torch.float, device=device)
        spatial_constraint_loss = torch.zeros(1, dtype=torch.float, device=device)  # TODO

        # iterate over positive samples and add it's loss to the total loss
        for i, pred_points in enumerate(positive_rep_points_init):
            pred_points = pred_points.reshape(-1, 2)
            # ground truth obb for the current rep points
            gt_box_idx = int(positive_assigned_gt_idxs_init[i]) - 1
            gt_points = gt_obb[gt_box_idx].reshape(-1, 2).float()
            gt_points.requires_grad = False

            pred_points_convex_hull = convex_hull(pred_points)
            localization_loss += giou_loss(gt_points, pred_points_convex_hull)
            spatial_constraint_loss += out_of_box_loss(gt_points, pred_points)  # TODO

        return (self.init_localization_weight * localization_loss +
                self.init_spatial_constraint_weight * spatial_constraint_loss)

    def _refinement_step_loss(self):
        # initialize losses with 0
        localization_loss = torch.zeros(1, dtype=torch.float, device=device)  # TODO
        spatial_constraint_loss = torch.zeros(1, dtype=torch.float, device=device)  # TODO

        return (self.refine_localization_weight * localization_loss +
                self.refine_spatial_constraint_weight * spatial_constraint_loss)

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

        assigned_gt_idxs_init, assigned_labels_init = self.init_assigner.assign(centers_init, gt_obb, gt_labels)
        classification_loss = focal_loss(classification, assigned_labels_init, alpha=0.25, gamma=2, reduction='mean')
        initialization_loss = self._initialization_step_loss(
            rep_points_init, gt_obb, assigned_gt_idxs_init, assigned_labels_init
        )
        refinement_loss = self._refinement_step_loss()

        return classification_loss + initialization_loss + refinement_loss


if __name__ == '__main__':
    model = DetectionModel()
    img_in = torch.rand(1, 3, 256, 256)

    # fake ground truth data
    gt_labels_ = torch.tensor([3, 1])
    gt_obboxes_ = torch.tensor([[1, 1, 1, 10, 10, 10, 10, 1],
                                [10, 10, 10, 50, 50, 50, 50, 10]])

    rep_points_init_, rep_points_refine_, classification_ = model(img_in)

    rep_points_loss = OrientedRepPointsLoss(strides=model.feature_map_strides)
    loss = rep_points_loss.get_loss(rep_points_init_, rep_points_refine_, classification_, gt_obboxes_, gt_labels_)
    print(loss)

    torch.autograd.set_detect_anomaly(True)

    learning_rate = 1e-3
    whight_decay = 5e-5

    # select optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=whight_decay,
        amsgrad=True
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
