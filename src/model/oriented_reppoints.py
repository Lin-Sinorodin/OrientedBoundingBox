import torch
import torch.nn as nn
from einops import rearrange, repeat
from torchvision.ops import DeformConv2d


def initialize_rep_points_centers(feature_map_size: tuple[int, int]) -> torch.tensor:
    """
    Initialize (y, x) center points for a given feature map (feature map coordinates).

    :param feature_map_size: (height, width) tuple of the feature map.
    :return: tensor of center points with shape [batch, 2, height, width], each center
             is (y, x) point in feature map coordinates.
    """
    h, w = feature_map_size
    grid_y, grid_x = torch.meshgrid(torch.arange(0., h), torch.arange(0., w))
    return rearrange(torch.stack([grid_y, grid_x], dim=-1), 'h w yx -> 1 yx h w')


def initialize_rep_points(feature_map_size: tuple[int, int]) -> torch.tensor:
    """
    Initialize [num_points * (x, y)] rep points for a given feature map. (feature map coordinates)

    :param feature_map_size: (height, width) tuple of the feature map.
    :return: tensor of center points with shape [batch, 2 * num_offsets, height, width].
    """
    points_center = initialize_rep_points_centers(feature_map_size)
    points_offset = torch.stack(torch.meshgrid([torch.tensor([-1, 0, 1])]*2), dim=-1).reshape(1, -1, 1, 1)
    rep_points_init = repeat(points_center, '1 xy h w -> 1 (repeat xy) h w', repeat=9)
    return rep_points_init + points_offset


class OrientedRepPointsHead(nn.Module):
    def __init__(self, num_offsets: int = 9, num_classes: int = 15):
        super().__init__()
        self.conv_params = {'kernel_size': (3, 3), 'stride': (1, 1), 'padding': 1, 'bias': False}

        # classification subnet
        self.classification_conv = self._get_features_subnet()
        self.classification_deform_conv = DeformConv2d(in_channels=256, out_channels=256, **self.conv_params)
        self.classification_conv_out = nn.Conv2d(in_channels=256, out_channels=num_classes, **self.conv_params)

        # localization subnet
        self.localization_conv = self._get_features_subnet()
        self.points_init_conv = nn.Conv2d(in_channels=256, out_channels=256, **self.conv_params)
        self.points_init_offset_conv = nn.Conv2d(in_channels=256, out_channels=num_offsets * 2, **self.conv_params)
        self.points_refine_deform_conv = DeformConv2d(in_channels=256, out_channels=256, **self.conv_params)
        self.points_refine_offset_conv = nn.Conv2d(in_channels=256, out_channels=num_offsets * 2, **self.conv_params)

    def _get_features_subnet(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, **self.conv_params),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.Conv2d(in_channels=256, out_channels=256, **self.conv_params),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.Conv2d(in_channels=256, out_channels=256, **self.conv_params),
            nn.GroupNorm(num_groups=32, num_channels=256),
        )

    def forward(self, feature: torch.tensor):
        # get offsets
        localization_features = self.localization_conv(feature)
        offset1 = self.points_init_offset_conv(self.points_init_conv(localization_features))
        offset2 = self.points_refine_offset_conv(
            self.points_refine_deform_conv(input=localization_features, offset=offset1)
        )

        # get points classification
        classification_features = self.classification_conv(feature)
        classification = self.classification_conv_out(
            self.classification_deform_conv(input=classification_features, offset=offset1)
        )

        # get rep points
        rep_points_init = initialize_rep_points(feature_map_size=tuple(feature.shape[2:4]))
        rep_points1 = rep_points_init + offset1
        rep_points2 = rep_points1 + offset2

        return rep_points1, rep_points2, classification


if __name__ == "__main__":
    img_h, img_w = (512, 512)
    strides = {'P2': 4, 'P3': 8, 'P4': 16, 'P5': 32}
    feature_maps = {name: torch.rand(1, 256, img_h // stride, img_w // stride) for name, stride in strides.items()}

    rotated_RepPoints_head = OrientedRepPointsHead()

    for name, feature_map in feature_maps.items():
        rep_points1_, rep_points2_, classification_ = rotated_RepPoints_head(feature_map)
        print('\n\t'.join([
            f'\nfeature map {name} with stride = {strides[name]}:',
            f'{feature_map.shape = }',
            f'{rep_points1_.shape = }',
            f'{rep_points2_.shape = }',
            f'{classification_.shape = }'
        ]))
