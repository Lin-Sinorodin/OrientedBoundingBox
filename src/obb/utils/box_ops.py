"""
Utilities for oriented bounding box manipulation and GIoU.
Credit: https://github.com/jw9730/ori-giou
"""
import torch


def convex_hull(points):
    """
    PyTorch-Compliant, vectorized convex hull implementation using the monotone chain algorithm.
    Details about the algorithm: https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain#Python
    TODO Add CUDA support

    :param points: (Tensor[..., N, 2]) Arbitrarily-dimensional tensor, with he last two dimensions
    containing sets of N points.
    :return: (Tensor[..., N, 2]) Tensor of the points belonging to the convex hull of each set, sorted counterclockwise.
    Cells after the last index in convex hull are assigned inf.
    """

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])

    D = points.shape[0]  # Dimensions excluding cluster size and coordinate number (2)
    # TODO Add support for Arbitrarily-dimensional tensors (current version works only for points.shape[:-2] = 1D).
    N = points.shape[-2]  # Number of points in each set

    # Sort points lexicographically by x-value
    # Trick: add negligible noise to enforce unique x-values
    eps = 1e-5
    points = points + eps * torch.randn(points.shape)
    indices = points[..., 0].argsort(dim=-1, descending=False)  # [..., N]
    indices = torch.stack([indices, indices], dim=-1)  # [..., N, 2]
    points = points.gather(dim=-2, index=indices)

    # Initialize lower + upper hulls
    lower = torch.zeros(points.shape).fill_(torch.inf)  # [..., N, 2]
    upper = torch.zeros(points.shape).fill_(torch.inf)  # [..., N, 2]

    lower_sizes = torch.zeros(D).long()  # [...]
    upper_sizes = torch.zeros(D).long()  # [...]

    for k in range(N):
        # Build lower hull
        while True:
            mask = (lower_sizes >= 2) & \
                   (cross(lower[torch.arange(D), lower_sizes - 2, :],
                          lower[torch.arange(D), lower_sizes - 1, :],
                          points[..., k, :]) <= 0)
            lower_sizes = lower_sizes - mask.long()
            if not mask.any():
                break

        lower[torch.arange(D), lower_sizes, :] = points[..., k, :]
        lower_sizes += 1

        # Build upper hull
        while True:
            mask = (upper_sizes >= 2) & \
                   (cross(upper[torch.arange(D), upper_sizes - 2, :],
                          upper[torch.arange(D), upper_sizes - 1, :],
                          points[:, N - 1 - k, :]) <= 0)
            upper_sizes = upper_sizes - mask.long()
            if not mask.any():
                break

        upper[torch.arange(D), upper_sizes, :] = points[..., N - 1 - k, :]
        upper_sizes += 1

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list.

    hull = torch.zeros(points.shape).fill_(torch.inf)  # [..., N, 2]
    sizes = torch.zeros(D).long()  # [...]

    for k in range(N):
        mask = lower_sizes > k + 1
        hull[mask, k, :] = lower[mask, k, :]
        sizes += mask.long()

    for k in range(N):
        mask = upper_sizes > k + 1

        push = upper[:, k, :].clone()
        keep = hull[torch.arange(D), sizes - 1, :]

        push[~mask, :] = 0
        keep[mask, :] = 0

        sizes += mask.long()
        hull[torch.arange(D), sizes - 1, :] = push + keep

    return hull
