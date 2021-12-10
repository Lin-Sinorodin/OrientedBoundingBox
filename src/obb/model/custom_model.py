import numpy as np
import torch
import torch.nn as nn

from obb.model.feature_map import FeatureMap, BACKBONE, NECK, REMEMBER_LAYERS, NUM_FEATURE_MAPS


class DetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_map = FeatureMap(BACKBONE, NECK, REMEMBER_LAYERS, NUM_FEATURE_MAPS)

    def forward(self, x):
        P2, P3, P4, P5 = self.feature_map(x)
        # put here the detection head...
        return P2, P3, P4, P5


if __name__ == '__main__':
    model = DetectionModel()
    img_in = torch.rand(1, 3, 160, 160)
    result = model(img_in)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
