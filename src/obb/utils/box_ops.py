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
    :return: 1. (Tensor[..., N, 2]) Tensor of the points belonging to the convex hull of each set, sorted
    counterclockwise. Cells after the last index in convex hull are assigned arbitrary values.
    2. (Tensor[...]) Tensor of the number of points belonging to the convex hull of each set.
    """

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])

    D = points.shape[0]  # Dimensions excluding cluster size and coordinate number (2)
    # TODO Add support for Arbitrarily-dimensional tensors (current version works only when points.shape[:-2] is 1D).
    N = points.shape[-2]  # Number of points in each set

    # Sort points lexicographically by x-value
    # Trick: add negligible noise to enforce unique x-values
    eps = 1e-5
    points = points + eps * torch.randn(points.shape)
    indices = points[..., 0].argsort(dim=-1, descending=False)  # [..., N]
    indices = indices.unsqueeze(-1).repeat(1, 1, 2)  # [..., N, 2]
    points = points.gather(dim=-2, index=indices)

    # Initialize lower + upper hulls
    lower = torch.zeros(points.shape)  # [..., N, 2]
    upper = torch.zeros(points.shape)  # [..., N, 2]

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

    hull = torch.zeros(points.shape).fill_(torch.nan)  # [..., N, 2]
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

    return hull, sizes


def min_area_rect(points):
    """
    PyTorch-Compliant, vectorized convex hull implementation of the cv2 MinAreaRect function, which finds for a given
    set of points the bounding rectangle with the minimal area. The algorithm works by computing the convex hull of the
    point set, finding the minimal rectangles with sides parallel to those of the hull, and choosing the rectangle with
    the minimal area from them.
    Implementation inspired by: https://gis.stackexchange.com/questions/22895/finding-minimum-area-rectangle-for-given-points
    TODO Add CUDA support

    :param points: (Tensor[..., N, 2]) Arbitrarily-dimensional tensor, with he last two dimensions
    containing sets of N points.
    :return: (Tensor[..., 4, 2]) Tensor containing the minimal rectangle for each point set in
    (x0, y0),...,(x3, y3) format, sorted counterclockwise.
    """

    def apply_rot(points, c, s, inv=False):
        """Applying passive rotation by angle theta on points, given c = cos(theta), s = sin(theta)."""
        c = c.unsqueeze(-1)  # [..., 1]
        s = s.unsqueeze(-1)  # [..., 1]

        if inv:
            return torch.stack([c * points[..., 0] - s * points[..., 1],
                                s * points[..., 0] + c * points[..., 1]], dim=-1)
        else:
            return torch.stack([c * points[..., 0] + s * points[..., 1],
                                -s * points[..., 0] + c * points[..., 1]], dim=-1)

    D = points.shape[0]  # Dimensions excluding cluster size and coordinate number (2)
    # TODO Add support for Arbitrarily-dimensional tensors (current version works only when points.shape[:-2] is 1D).
    N = points.shape[-2]  # Number of points in each set
    # Find convex hull of point set

    hull, sizes = convex_hull(points)

    # Retrieve hull edges and their respective angles relative to the positive x-axis
    edges = torch.zeros(points.shape)  # [..., N, 2]
    edges[..., 1:, :] = hull[..., 1:, :] - hull[..., :-1, :]
    edges[..., 0, :] = hull[..., 0, :] - hull[torch.arange(D), sizes - 1, :]
    angles = torch.atan2(edges[..., 1], edges[..., 0])  # [..., N]

    rect_min = torch.zeros(D, 4, 2)  # [..., 4, 2]
    area_min = torch.zeros(D).fill_(torch.inf)  # [...]

    for k in range(N):
        # Apply rotation by corresponding edge angles
        theta = angles[..., k]
        c = torch.cos(theta)  # [...]
        s = torch.sin(theta)  # [...]

        hull_rot = apply_rot(hull, c, s, inv=False)  # [..., N, 2]

        # Find minimal and maximal x, y coordinates in the rotated frame
        x_min, _ = torch.min(hull_rot[..., 0].nan_to_num(torch.inf), dim=-1)  # [...]
        x_max, _ = torch.max(hull_rot[..., 0].nan_to_num(-torch.inf), dim=-1)  # [...]
        y_min, _ = torch.min(hull_rot[..., 1].nan_to_num(torch.inf), dim=-1)  # [...]
        y_max, _ = torch.max(hull_rot[..., 1].nan_to_num(-torch.inf), dim=-1)  # [...]

        # Calculate area of resultant rectangle
        area = (x_max - x_min) * (y_max - y_min)

        # Updating rectangle vertices and minimal area for each point set achieving smaller area than current minimum
        mask = (area < area_min) & (sizes > k)
        area_min[mask] = area[mask]

        rect_x = torch.stack([x_min, x_max, x_max, x_min], dim=-1)  # [..., 4]
        rect_y = torch.stack([y_min, y_min, y_max, y_max], dim=-1)  # [..., 4]
        rect = torch.stack([rect_x, rect_y], dim=-1)  # [..., 4, 2]

        rect_min[mask] = apply_rot(rect[mask], c[mask], s[mask], inv=True)

    return rect_min
