import numpy as np
import torch
import torch.nn as nn

from obb.model.oriented_reppoints import OrientedRepPointsHead
from obb.model.feature_map import FeatureMap, BACKBONE, NECK, REMEMBER_LAYERS, NUM_FEATURE_MAPS


class DetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_map = FeatureMap(BACKBONE, NECK, REMEMBER_LAYERS, NUM_FEATURE_MAPS)
        self.feature_map_strides = {'P2': 4, 'P3': 8, 'P4': 16, 'P5': 32}
        self.oriented_rep_points_head = OrientedRepPointsHead(num_offsets=9, num_classes=15)

    def forward(self, x):
        # get feature maps from backbone + neck
        P2, P3, P4, P5 = self.feature_map(x)
        feature_maps = {'P2': P2, 'P3': P3, 'P4': P4, 'P5': P5}

        # get rep points and classification from detection head
        rep_points_init, rep_points_refine, classifications = {}, {}, {}
        for feature_map_name, feature_map in feature_maps.items():
            curr_points_init, curr_points_refine, classification = self.oriented_rep_points_head(feature_map)
            rep_points_init[feature_map_name] = curr_points_init
            rep_points_refine[feature_map_name] = curr_points_refine
            classifications[feature_map_name] = classification

        return rep_points_init, rep_points_refine, classifications


if __name__ == '__main__':
    model = DetectionModel()
    img_in = torch.rand(1, 3, 160, 160)
    result = model(img_in)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
