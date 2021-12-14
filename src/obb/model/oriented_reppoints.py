import torch
import torch.nn as nn
from einops import rearrange, repeat
from torchvision.ops import DeformConv2d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    points_offset = torch.stack(torch.meshgrid([torch.tensor([-1, 0, 1])] * 2), dim=-1).reshape(1, -1, 1, 1)
    rep_points_init = repeat(points_center, '1 xy h w -> 1 (repeat xy) h w', repeat=9)
    return rep_points_init + points_offset


def rep_point_to_img_space(feature_space_tensor: torch.tensor, stride: int) -> torch.tensor:
    """
    Converts rep point tensor from feature space (of a feature map with a given stride) to the image space.

    :param feature_space_tensor: the tensor in feature space to be converted to image space
    :param stride: the stride of the current feature map is the scale between feature space and image space
    :return: the input tensor in the image space
    """
    return torch.tensor([stride]).reshape(1, -1, 1, 1) * feature_space_tensor


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

    rotated_RepPoints_head = OrientedRepPointsHead().to(device)

    for name, feature_map in feature_maps.items():
        stride = strides[name]
        rep_points1_, rep_points2_, classification_ = rotated_RepPoints_head(feature_map)

        # convert rep points to image space in order to calculate the losses
        rep_points1_ = rep_point_to_img_space(rep_points1_, stride)
        rep_points2_ = rep_point_to_img_space(rep_points2_, stride)

        print('\n\t'.join([
            f'\nfeature map {name} with {stride = }:',
            f'{feature_map.shape = }',
            f'{rep_points1_.shape = }',
            f'{rep_points2_.shape = }',
            f'{classification_.shape = }'
        ]))

    """
    feature map P2 with stride = 4:
        feature_map.shape = torch.Size([1, 256, 128, 128])
        rep_points1_.shape = torch.Size([1, 18, 128, 128])
        rep_points2_.shape = torch.Size([1, 18, 128, 128])
        classification_.shape = torch.Size([1, 15, 128, 128])

    feature map P3 with stride = 8:
        feature_map.shape = torch.Size([1, 256, 64, 64])
        rep_points1_.shape = torch.Size([1, 18, 64, 64])
        rep_points2_.shape = torch.Size([1, 18, 64, 64])
        classification_.shape = torch.Size([1, 15, 64, 64])

    feature map P4 with stride = 16:
        feature_map.shape = torch.Size([1, 256, 32, 32])
        rep_points1_.shape = torch.Size([1, 18, 32, 32])
        rep_points2_.shape = torch.Size([1, 18, 32, 32])
        classification_.shape = torch.Size([1, 15, 32, 32])

    feature map P5 with stride = 32:
        feature_map.shape = torch.Size([1, 256, 16, 16])
        rep_points1_.shape = torch.Size([1, 18, 16, 16])
        rep_points2_.shape = torch.Size([1, 18, 16, 16])
        classification_.shape = torch.Size([1, 15, 16, 16])
    """
