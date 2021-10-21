import torch
import torch.nn as nn

from code.model import YOLOv5


class DetectionModel(nn.Module):
    def __init__(self):
        super(DetectionModel, self).__init__()
        self.yolov5 = YOLOv5(cpu=True)

    def forward(self, x):
        P3, P4, P5 = self.yolov5.get_features(x)
        # put here the detection head...
        return P3, P4, P5


model = DetectionModel()
x = torch.rand(1, 3, 640, 640)
print(model(x))
