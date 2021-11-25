import numpy as np
import torch
import torch.nn as nn

from src.model import YOLOv5Features


class DetectionModel(nn.Module):
    def __init__(self, yolo_requires_grad=False):
        super().__init__()
        self.yolo_features = YOLOv5Features(requires_grad=yolo_requires_grad)

    def forward(self, x):
        P3, P4, P5 = self.yolo_features(x)
        # put here the detection head...
        return P3, P4, P5


model = DetectionModel(True)
x = torch.rand(1, 3, 640, 640)
result = model(x)
# print(result)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)
