import numpy as np
import torch
import torch.nn as nn

from obb.model.yolov5 import YOLOv5Features
from obb.model.oriented_reppoints import OrientedRepPointsHead
from obb.model.oriented_reppoints import rep_point_to_img_space

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DetectionModel(nn.Module):
    def __init__(self, train_backbone: bool = False):
        super().__init__()
        self.feature_map = YOLOv5Features(requires_grad=train_backbone).to(device)
        self.feature_map_strides = {'P3': 8, 'P4': 16, 'P5': 32}

        conv_kwargs = {'kernel_size': 3, 'padding': 1, 'bias': False, 'device': device}
        self.P3_resample = nn.Conv2d(in_channels=192, out_channels=256, **conv_kwargs)
        self.P4_resample = nn.Conv2d(in_channels=384, out_channels=256, **conv_kwargs)
        self.P5_resample = nn.Conv2d(in_channels=768, out_channels=256, **conv_kwargs)

        self.oriented_rep_points_head = OrientedRepPointsHead(num_offsets=9, num_classes=15)

    def forward(self, x):
        # get feature maps from backbone + neck
        P3, P4, P5 = self.feature_map(x)

        # resample feature map to fit RepPoints head
        P3 = self.P3_resample(P3)
        P4 = self.P4_resample(P4)
        P5 = self.P5_resample(P5)

        feature_maps = {'P3': P3, 'P4': P4, 'P5': P5}

        # get rep points and classification from detection head
        rep_points_init, rep_points_refine, classifications = {}, {}, {}
        for feature_map_name, feature_map in feature_maps.items():
            stride = self.feature_map_strides[feature_map_name]
            curr_points_init, curr_points_refine, classification = self.oriented_rep_points_head(feature_map)
            rep_points_init[feature_map_name] = rep_point_to_img_space(curr_points_init, stride)
            rep_points_refine[feature_map_name] = rep_point_to_img_space(curr_points_refine, stride)
            classifications[feature_map_name] = classification

        return rep_points_init, rep_points_refine, classifications


class DetectionModelFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_map_strides = {'P3': 8, 'P4': 16, 'P5': 32}

        conv_kwargs = {'kernel_size': 3, 'padding': 1, 'bias': False, 'device': device}
        self.P3_resample = nn.Conv2d(in_channels=192, out_channels=256, **conv_kwargs)
        self.P4_resample = nn.Conv2d(in_channels=384, out_channels=256, **conv_kwargs)
        self.P5_resample = nn.Conv2d(in_channels=768, out_channels=256, **conv_kwargs)

        self.oriented_rep_points_head = OrientedRepPointsHead(num_offsets=9, num_classes=15)

    def forward(self, P3, P4, P5):
        # resample feature map to fit RepPoints head
        P3 = self.P3_resample(P3)
        P4 = self.P4_resample(P4)
        P5 = self.P5_resample(P5)

        feature_maps = {'P3': P3, 'P4': P4, 'P5': P5}

        # get rep points and classification from detection head
        rep_points_init, rep_points_refine, classifications = {}, {}, {}
        for feature_map_name, feature_map in feature_maps.items():
            stride = self.feature_map_strides[feature_map_name]
            curr_points_init, curr_points_refine, classification = self.oriented_rep_points_head(feature_map)
            rep_points_init[feature_map_name] = rep_point_to_img_space(curr_points_init, stride)
            rep_points_refine[feature_map_name] = rep_point_to_img_space(curr_points_refine, stride)
            classifications[feature_map_name] = classification

        return rep_points_init, rep_points_refine, classifications


if __name__ == '__main__':
    model = DetectionModel().to(device)
    img_in = torch.rand(1, 3, 512, 512).to(device)
    rep_points_init, rep_points_refine, classifications = model(img_in)

    trainable_params = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])
    frozen_params = sum([np.prod(p.size()) for p in filter(lambda p: not p.requires_grad, model.parameters())])
    print(f'Trainable Parameters: {trainable_params:3,}')
    print(f'Frozen Parameters: {frozen_params:3,}')
