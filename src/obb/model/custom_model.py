import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from obb.model.yolov5 import YOLOv5Features
from obb.model.oriented_reppoints import OrientedRepPointsHead
from obb.model.oriented_reppoints import rep_point_to_img_space

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_pretrained_state_dict(path):
    state_dict_mapper = {
        'bbox_head.cls_convs.0.conv.weight': 'classification_conv.0.weight',
        'bbox_head.cls_convs.0.gn.weight': 'classification_conv.1.weight',
        'bbox_head.cls_convs.0.gn.bias': 'classification_conv.1.bias',
        'bbox_head.cls_convs.1.conv.weight': 'classification_conv.2.weight',
        'bbox_head.cls_convs.1.gn.weight': 'classification_conv.3.weight',
        'bbox_head.cls_convs.1.gn.bias': 'classification_conv.3.bias',
        'bbox_head.cls_convs.2.conv.weight': 'classification_conv.4.weight',
        'bbox_head.cls_convs.2.gn.weight': 'classification_conv.5.weight',
        'bbox_head.cls_convs.2.gn.bias': 'classification_conv.5.bias',
        'bbox_head.reg_convs.0.conv.weight': 'localization_conv.0.weight',
        'bbox_head.reg_convs.0.gn.weight': 'localization_conv.1.weight',
        'bbox_head.reg_convs.0.gn.bias': 'localization_conv.1.bias',
        'bbox_head.reg_convs.1.conv.weight': 'localization_conv.2.weight',
        'bbox_head.reg_convs.1.gn.weight': 'localization_conv.3.weight',
        'bbox_head.reg_convs.1.gn.bias': 'localization_conv.3.bias',
        'bbox_head.reg_convs.2.conv.weight': 'localization_conv.4.weight',
        'bbox_head.reg_convs.2.gn.weight': 'localization_conv.5.weight',
        'bbox_head.reg_convs.2.gn.bias': 'localization_conv.5.bias',
        'bbox_head.reppoints_cls_conv.weight': 'classification_deform_conv.weight',
        'bbox_head.reppoints_cls_out.weight': 'classification_conv_out.weight',
        'bbox_head.reppoints_cls_out.bias': 'classification_conv_out.bias',
        'bbox_head.reppoints_pts_init_conv.weight': 'points_init_conv.weight',
        'bbox_head.reppoints_pts_init_conv.bias': 'points_init_conv.bias',
        'bbox_head.reppoints_pts_init_out.weight': 'points_init_offset_conv.weight',
        'bbox_head.reppoints_pts_init_out.bias': 'points_init_offset_conv.bias',
        'bbox_head.reppoints_pts_refine_conv.weight': 'points_refine_deform_conv.weight',
        'bbox_head.reppoints_pts_refine_out.weight': 'points_refine_offset_conv.weight',
        'bbox_head.reppoints_pts_refine_out.bias': 'points_refine_offset_conv.bias',
    }

    pretrained_weights = torch.load(path, map_location=device)['state_dict']

    rep_points_head = OrientedRepPointsHead()

    sd = rep_points_head.state_dict()
    for pretrained_key, model_key in state_dict_mapper.items():
        sd[model_key] = pretrained_weights[pretrained_key]

    return sd


class DetectionModel(nn.Module):
    def __init__(self, train_backbone: bool = False, state_dict_path: Optional[str] = None):
        super().__init__()
        self.feature_map = YOLOv5Features(requires_grad=train_backbone).to(device)
        self.feature_map_strides = {'P3': 8, 'P4': 16, 'P5': 32}

        conv_kwargs = {'kernel_size': 3, 'padding': 1, 'bias': False, 'device': device}
        self.P3_resample = nn.Conv2d(in_channels=192, out_channels=256, **conv_kwargs)
        self.P4_resample = nn.Conv2d(in_channels=384, out_channels=256, **conv_kwargs)
        self.P5_resample = nn.Conv2d(in_channels=768, out_channels=256, **conv_kwargs)

        self.oriented_rep_points_head = OrientedRepPointsHead(num_offsets=9, num_classes=15)

        if state_dict_path:
            pretrained_state_dict = get_pretrained_state_dict(state_dict_path)
            self.oriented_rep_points_head.load_state_dict(pretrained_state_dict)

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
    model = DetectionModel(state_dict_path='weights/epoch_40.pth').to(device)
    img_in = torch.rand(1, 3, 512, 512).to(device)
    rep_points_init, rep_points_refine, classifications = model(img_in)

    trainable_params = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])
    frozen_params = sum([np.prod(p.size()) for p in filter(lambda p: not p.requires_grad, model.parameters())])
    print(f'Trainable Parameters: {trainable_params:3,}')
    print(f'Frozen Parameters: {frozen_params:3,}')
