import torch
from obb.model.oriented_reppoints import OrientedRepPointsHead


def test_OrientedRepPointsHead():
    img_h, img_w = (512, 512)
    strides = {'P2': 4, 'P3': 8, 'P4': 16, 'P5': 32}
    feature_maps = {name: torch.rand(1, 256, img_h // stride, img_w // stride) for name, stride in strides.items()}

    num_offsets = 9
    num_classes = 15
    rotated_RepPoints_head = OrientedRepPointsHead(num_offsets=num_offsets, num_classes=num_classes)

    for name, feature_map in feature_maps.items():
        stride = strides[name]
        feature_space_h = img_h // stride
        feature_space_w = img_w // stride

        rep_points1_, rep_points2_, classification_ = rotated_RepPoints_head(feature_map)
        assert tuple(rep_points1_.shape) == (1, num_offsets * 2, feature_space_h, feature_space_w)
        assert tuple(rep_points2_.shape) == (1, num_offsets * 2, feature_space_h, feature_space_w)
        assert tuple(classification_.shape) == (1, num_classes, feature_space_h, feature_space_w)
