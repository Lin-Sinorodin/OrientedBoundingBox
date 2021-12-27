import torch
from typing import Tuple, List

device = 'cpu'


def column_vector_meshgrid(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    """Returns a column vector with (x, y) values from the meshgrid of the arrays"""
    grid_x, grid_y = torch.meshgrid(x, y)
    return torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)


def get_feature_map_center_points(feature_map_size: Tuple[int, int]) -> torch.tensor:
    """Initialize list of (x, y) center points for a given feature map. (feature map coordinates)"""
    h, w = feature_map_size
    x = torch.arange(0., w, device=device)
    y = torch.arange(0., h, device=device)
    return column_vector_meshgrid(x, y)


def get_feature_map_center_point_offset() -> List[torch.tensor]:
    """Initialize list of (x, y) offsets for a given center point. (feature map coordinates)"""
    x = torch.arange(-1, 1+1, device=device)
    y = torch.arange(-1, 1+1, device=device)
    return column_vector_meshgrid(x, y)


def initialize_center_points(feature_maps) -> List[torch.tensor]:
    """Initialize a list of center points for each feature map."""
    feature_maps_center_points_xy = []
    for feature_map in feature_maps:
        _, _, h, w = feature_map.shape
        center_points_xy = get_feature_map_center_points((h, w))   # get center points in feature map coordinates
        center_points_xy = center_points_xy * feature_map.stride   # feature map coordinates -> img coordinates
        feature_maps_center_points_xy.append(center_points_xy)
    return feature_maps_center_points_xy


def initialize_center_points_offsets(feature_maps_center_points: List[torch.tensor]) -> List[torch.tensor]:
    """Initialize a list of offsets from center point for each feature map."""
    feature_maps_center_points_offsets = []
    for feature_map_center_points in feature_maps_center_points:
        center_points_offsets = []
        for center_point in feature_map_center_points:
            offsets = get_feature_map_center_point_offset()   # initialize offsets from center
            point_offsets = center_point + offsets            # offsets -> points in feature map coordinates
            center_points_offsets.append(point_offsets)
        feature_maps_center_points_offsets.append(torch.stack(center_points_offsets))
    return feature_maps_center_points_offsets


if __name__ == '__main__':
    img_h, img_w = 512, 512
    img_in = torch.rand(1, 3, img_h, img_w)

    feature_maps_strides = [8, 16, 32, 64]
    feature_maps_depth = [128, 256, 512, 1024]

    feature_maps_ = []
    for stride, depth in zip(feature_maps_strides, feature_maps_depth):
        feature_map_ = torch.rand(1, depth, img_h // stride, img_w // stride)
        feature_map_.stride = stride
        feature_maps_.append(feature_map_)

    # initialize each point in the feature map coordinates as a center point
    feature_maps_center_points_ = initialize_center_points(feature_maps_)

    # initialize for each center point a grid of offsets
    feature_maps_center_points_offsets_ = initialize_center_points_offsets(feature_maps_center_points_)

    for idx in range(len(feature_maps_)):
        print(f'feature map shape: {feature_maps_[idx].shape}')
        print(f'center points shape: {feature_maps_center_points_[idx].shape}')
        print(f'offsets shape: {feature_maps_center_points_offsets_[idx].shape}')
        print('')

    """
    feature map shape: torch.Size([1, 128, 64, 64])
    center points shape: torch.Size([4096, 2])
    offsets shape: torch.Size([4096, 9, 2])

    feature map shape: torch.Size([1, 256, 32, 32])
    center points shape: torch.Size([1024, 2])
    offsets shape: torch.Size([1024, 9, 2])

    feature map shape: torch.Size([1, 512, 16, 16])
    center points shape: torch.Size([256, 2])
    offsets shape: torch.Size([256, 9, 2])

    feature map shape: torch.Size([1, 1024, 8, 8])
    center points shape: torch.Size([64, 2])
    offsets shape: torch.Size([64, 9, 2])
    """
