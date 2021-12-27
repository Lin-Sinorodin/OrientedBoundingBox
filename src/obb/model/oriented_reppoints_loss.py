import torch
import torch.nn as nn
from einops import rearrange

from obb.model.custom_model import DetectionModel
from obb.utils.loss import FocalLoss
from obb.utils.box_ops import convex_hull
from obb.utils.poly_intersection import PolygonClipper, polygon_area
from typing import Dict

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

        self.classification_loss_metric = FocalLoss(gamma=2, alpha=0.25)

        self.init_assigner = OBBPointAssigner()
        self.refine_assigner = ...  # TODO

    def _process_data(self, raw_rep_points_init, raw_rep_points_refine, raw_classification):
        multi_level_rep_points_init = {}
        multi_level_rep_points_refine = {}
        multi_level_classification = {}
        multi_level_centers_init = {}
        multi_level_centers_refine = {}
        for feature_map in self.feature_maps_names:
            # get rep points of the current feature map
            curr_rep_points1 = raw_rep_points_init[feature_map]
            curr_rep_points2 = raw_rep_points_refine[feature_map]
            curr_classification = raw_classification[feature_map]

            # flatten the rep points and classification tensor
            multi_level_rep_points_init[feature_map] = rearrange(curr_rep_points1, 'b xy w h -> (b w h) xy')
            multi_level_rep_points_refine[feature_map] = rearrange(curr_rep_points2, 'b xy w h -> (b w h) xy')
            multi_level_classification[feature_map] = rearrange(curr_classification, 'b cls w h -> (b w h) cls')

            # find rep points centers for the ground truth assigner
            centers1 = rearrange(curr_rep_points1[:, 8:10, :, :], 'b xy w h -> (b w h) xy')
            centers2 = rearrange(curr_rep_points2[:, 8:10, :, :], 'b xy w h -> (b w h) xy')
            stride_vec = self.strides.get(feature_map) * torch.ones(centers1.shape[0], 1).to(device)
            multi_level_centers_init[feature_map] = torch.cat([centers1, stride_vec], dim=1)  # [..., 3]
            multi_level_centers_refine[feature_map] = torch.cat([centers2, stride_vec], dim=1)  # [..., 3]

        # concat multi-level features
        rep_points_init = torch.cat(list(multi_level_rep_points_init.values()), dim=0)
        rep_points_refine = torch.cat(list(multi_level_rep_points_refine.values()), dim=0)
        classification = torch.cat(list(multi_level_classification.values()), dim=0)
        centers_init = torch.cat(list(multi_level_centers_init.values()), dim=0)
        multi_level_centers_refine = torch.cat(list(multi_level_centers_refine.values()), dim=0)

        return rep_points_init, rep_points_refine, classification, centers_init, multi_level_centers_refine

    def _initialization_step_loss(self, rep_points_init, centers_init, gt_obboxes, gt_labels):
        # initialization stage assigner
        assigned_gt_idxs_init, assigned_labels_init = self.init_assigner.assign(centers_init, gt_obboxes, gt_labels)
        positive_samples_idx_init = torch.where(assigned_labels_init > 0)
        positive_rep_points_init = rep_points_init[positive_samples_idx_init]

        # initialize losses with 0
        localization_loss = torch.zeros(1).float().to(device)
        spatial_constraint_loss = torch.zeros(1).float().to(device)  # TODO

        # iterate over positive samples and add it's loss to the total loss
        for i, points in enumerate(positive_rep_points_init):
            curr_rep_points = points.reshape(-1, 2).unsqueeze(dim=0)

            # convex hull of the current rep points
            hull, hull_size = convex_hull(curr_rep_points)
            hull_points = hull[:, :int(hull_size[0]), :].squeeze(dim=0)
            hull_points = torch.flip(hull_points, [0])  # counterclockwise -> clockwise

            # ground truth obb for the current rep points
            gt_points = gt_obboxes[i].reshape(-1, 2).float()
            gt_points.requires_grad = True

            # calculate IOU for localization loss
            intersection_points = PolygonClipper()(hull_points, gt_points)
            poly1_area = polygon_area(hull_points)
            poly2_area = polygon_area(gt_points)
            intersection_area = polygon_area(intersection_points)
            union_area = poly1_area + poly2_area - intersection_area
            iou = intersection_area / union_area  # TODO convert it to GIoU
            localization_loss += 1 - iou

            # calculate spatial_constraint_loss
            # TODO

        return (self.init_localization_weight * localization_loss +
                self.init_spatial_constraint_weight * spatial_constraint_loss)

    def _refinement_step_loss(self):
        # initialize losses with 0
        localization_loss = torch.zeros(1).float().to(device)        # TODO
        spatial_constraint_loss = torch.zeros(1).float().to(device)  # TODO

        return (self.refine_localization_weight * localization_loss +
                self.refine_spatial_constraint_weight * spatial_constraint_loss)

    def get_loss(
            self,
            raw_rep_points_init: Dict[str, torch.Tensor],  # {'P2': shape[b xy w h], ...}
            raw_rep_points_refine: Dict[str, torch.Tensor],  # {'P2': shape[b xy w h], ...}
            raw_classification: Dict[str, torch.Tensor],  # {'P2': shape[b cls w h], ...}
            gt_obboxes: torch.Tensor,  # [[x1, y1, x2, y2, x3, y3, x4, y4], [...]]
            gt_labels: torch.Tensor  # [bbox1_cls, bbox2_cls, ...]
    ):
        """Loss function for Oriented Rep Points"""
        # convert the output of Rep Points head to flattened representation
        rep_points_init, rep_points_refine, classification, centers_init, centers_refine = self._process_data(
            raw_rep_points_init, raw_rep_points_refine, raw_classification
        )

        classification_loss = self.classification_loss_metric(classification, gt_labels)
        initialization_loss = self._initialization_step_loss(rep_points_init, centers_init, gt_obboxes, gt_labels)
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
