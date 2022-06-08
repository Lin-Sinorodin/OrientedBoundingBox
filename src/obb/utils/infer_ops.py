import torch
from einops import rearrange

from .box_ops import *

NUM_CLASSES = 15
FEATURE_MAP_LVLS = ['P3', 'P4', 'P5']


def flatten_head_output(classification, rep_points):
    multi_level_classification = []
    multi_level_rep_points = []

    for lvl in FEATURE_MAP_LVLS:
        # Get predictions from current feature map
        curr_level_classification = classification[lvl]
        curr_level_rep_points = rep_points[lvl]

        # Flatten prediction tensors
        multi_level_classification.append(rearrange(curr_level_classification, 'b xy w h -> (b w h) xy'))
        multi_level_rep_points.append(rearrange(curr_level_rep_points, 'b xy w h -> (b w h) xy'))

    # Concatenate multi-level features
    classification_flattened = torch.cat(multi_level_classification, dim=0)
    rep_points_flattened = torch.cat(multi_level_rep_points, dim=0)

    return classification_flattened, rep_points_flattened


def nms_kl(classification, rep_points, cls_thr=0.9, nms_thr=1):
    if len(classification) == 0:
        return torch.Tensor([]), torch.Tensor([])

    # Convert classification logits into probabilities
    classification = classification.softmax(dim=1)
    classification_hard = classification.argmax(dim=1) + 1

    # Initialize list of final predictions
    classification_final, rep_points_final = [], []

    # Convert RepPoints to Gaussian distribution
    mu, S = rep_points_to_gaussian(rep_points.reshape(-1, 9, 2))

    # Iterate over every class
    for class_num in range(1, NUM_CLASSES + 1):
        # Get predictions of current class
        curr_class_idx = torch.where(classification_hard == class_num)
        classification_curr = classification[curr_class_idx]
        rep_points_curr = rep_points[curr_class_idx]
        mu_curr = mu[curr_class_idx]
        S_curr = S[curr_class_idx]

        # Discard predictions below classification threshold
        above_cls_thr_idx = torch.where(classification_curr[:, class_num - 1] > cls_thr)
        classification_curr = classification_curr[above_cls_thr_idx]
        rep_points_curr = rep_points_curr[above_cls_thr_idx]
        mu_curr = mu_curr[above_cls_thr_idx]
        S_curr = S_curr[above_cls_thr_idx]

        while classification_curr.shape[0] > 0:
            print(classification_curr.shape[0])
            classification_max_idx = torch.argmax(classification_curr[:, class_num - 1])
            classification_final.append(classification_curr[classification_max_idx])
            rep_points_final.append(rep_points_curr[classification_max_idx])

            # Compute KL divergence of every distribution with the one that has maximal score
            mu_max = mu_curr[classification_max_idx]
            S_max = S_curr[classification_max_idx]
            kl_with_cls_max = torch.Tensor(classification_curr.shape[0])
            # TODO make KL divergence calculation batched
            for i in range(classification_curr.shape[0]):
                kl_with_cls_max[i] = kl_divergence_gaussian(mu_max, S_max, mu_curr[i], S_curr[i])

            # Discard predictions with KL divergence below threshold
            keep = torch.where(kl_with_cls_max > nms_thr)
            classification_curr = classification_curr[keep]
            rep_points_curr = rep_points_curr[keep]
            mu_curr = mu_curr[keep]
            S_curr = S_curr[keep]

    if len(classification_final) == 0:
        return torch.Tensor([]), torch.Tensor([])

    return torch.stack(classification_final, dim=0), torch.stack(rep_points_final, dim=0)


if __name__ == '__main__':
    b, cls, xy, h, w = 1, 15, 18, 64, 64
    lvls = ['P3', 'P4', 'P5']
    classification_dict = {lvl: torch.rand(b, cls, h, w) for lvl in lvls}
    rep_point_dict = {lvl: torch.rand(b, xy, h, w) for lvl in lvls}

    classification_flattened, rep_points_flattened = flatten_head_output(classification_dict, rep_point_dict)
    print(classification_flattened.shape)
    print(rep_points_flattened.shape)

    classification_nms, rep_points_nms = nms_kl(classification_flattened, rep_points_flattened, cls_thr=0.12, nms_thr=1)
    print(classification_nms.shape)
    print(rep_points_nms.shape)
