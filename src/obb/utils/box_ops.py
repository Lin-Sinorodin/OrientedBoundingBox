"""
Utilities for oriented bounding box manipulation and GIoU.
Credit: https://github.com/jw9730/ori-giou
"""
import torch


def cross(o: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.tensor:
    """
    2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.

    :param o: point O for OA and OB vector.
    :param a: point A for OA vector.
    :param b: point B for OB vector.
    :return: positive if OAB makes a counter-clockwise turn, negative for clockwise, zero if the points are collinear.
    """
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])


def convex_hull(points: torch.Tensor):
    """
    PyTorch-Compliant, vectorized convex hull implementation using the monotone chain algorithm.
    Details about the algorithm:
        https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain#Python

    :param points: (Tensor[..., N, 2]) Arbitrarily-dimensional tensor, with the last two dimensions
    containing sets of N points.
    :return: 1. (Tensor[..., N, 2]) Tensor of the points belonging to the convex hull of each set, sorted
    counterclockwise. Cells after the last index in convex hull are assigned arbitrary values.
    2. (Tensor[...]) Tensor of the number of points belonging to the convex hull of each set.
    """
    device = points.device

    D = points.shape[0]  # Dimensions excluding cluster size and coordinate number (2)
    # TODO Add support for Arbitrarily-dimensional tensors (current version works only when points.shape[:-2] is 1D).
    N = points.shape[-2]  # Number of points in each set

    # Sort points lexicographically by x-value
    # Trick: add negligible noise to enforce unique x-values
    eps = 1e-5
    points = points + eps * torch.randn(points.shape, device=device)
    indices = points[..., 0].argsort(dim=-1, descending=False)  # [..., N]
    indices = indices.unsqueeze(-1).repeat(1, 1, 2)  # [..., N, 2]
    points = points.gather(dim=-2, index=indices)

    # Initialize lower + upper hulls
    lower = torch.zeros(points.shape, device=device)  # [..., N, 2]
    upper = torch.zeros(points.shape, device=device)  # [..., N, 2]

    lower_sizes = torch.zeros(D, device=device).long()  # [...]
    upper_sizes = torch.zeros(D, device=device).long()  # [...]

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

    hull = torch.zeros(points.shape, device=device).fill_(float('nan'))  # [..., N, 2]
    sizes = torch.zeros(D, device=device).long()  # [...]

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


def diff_convex_hull(points: torch.Tensor):
    """
    PyTorch-Compliant, vectorized convex hull implementation using the monotone chain algorithm.
    Details about the algorithm:
        https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain#Python

    :param points: (Tensor[..., N, 2]) Arbitrarily-dimensional tensor, with he last two dimensions
    containing sets of N points.
    :return: 1. (Tensor[..., N, 2]) Tensor of the points belonging to the convex hull of each set, sorted
    counterclockwise. Cells after the last index in convex hull are assigned arbitrary values.
    2. (Tensor[...]) Tensor of the number of points belonging to the convex hull of each set.
    """
    device = points.device

    if points.shape[0] != 1:
        raise NotImplementedError

    # TODO Add support for Arbitrarily-dimensional tensors (current version works only when points.shape[:-2] is 1D).
    N = points.shape[-2]  # Number of points in each set

    # Sort points lexicographically by x-value
    # Trick: add negligible noise to enforce unique x-values
    eps = 1e-5
    points = points + eps * torch.randn(points.shape, device=device)
    indices = points[..., 0].argsort(dim=-1, descending=False)  # [..., N]
    indices = indices.unsqueeze(-1).repeat(1, 1, 2)  # [..., N, 2]
    points = points.gather(dim=-2, index=indices)

    # Initialize lower + upper hulls
    lower = torch.zeros_like(points, device=device)  # [..., N, 2]
    upper = torch.zeros_like(points, device=device)  # [..., N, 2]

    lower_sizes = 0
    upper_sizes = 0

    for k in range(N):
        # Build lower hull
        while True:
            mask = (lower_sizes >= 2) * (cross(lower[0, lower_sizes - 2, :],
                                               lower[0, lower_sizes - 1, :],
                                               points[..., k, :]) <= 0)
            lower_sizes = lower_sizes - mask.long()
            if not mask.any():
                break

        lower[0, lower_sizes, :] = points[..., k, :]
        lower_sizes += 1

        # Build upper hull
        while True:
            mask = (upper_sizes >= 2) * (cross(upper[0, upper_sizes - 2, :],
                                               upper[0, upper_sizes - 1, :],
                                               points[:, N - 1 - k, :]) <= 0)
            upper_sizes = upper_sizes - mask.long()
            if not mask.any():
                break

        upper[0, upper_sizes, :] = points[..., N - 1 - k, :]
        upper_sizes += 1

    positive_upper = int(torch.where((upper[0, :, 0] > 0) * (upper[0, :, 1] > 0))[0][-1])
    positive_lower = int(torch.where((lower[0, :, 0] > 0) * (lower[0, :, 1] > 0))[0][-1])
    return torch.cat([upper[0, :positive_upper, :], lower[0, :positive_lower, :]], dim=0)


def apply_rot(points: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inv: bool = False):
    """Applying passive rotation by angle theta on points, given cos(theta), sin(theta)."""
    cos = cos.unsqueeze(-1)  # [..., 1]
    sin = sin.unsqueeze(-1)  # [..., 1]

    if inv:
        return torch.stack([cos * points[..., 0] - sin * points[..., 1],
                            sin * points[..., 0] + cos * points[..., 1]], dim=-1)
    else:
        return torch.stack([cos * points[..., 0] + sin * points[..., 1],
                            -sin * points[..., 0] + cos * points[..., 1]], dim=-1)


def min_area_rect(points: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-Compliant, vectorized convex hull implementation of the cv2 MinAreaRect function, which finds for a given
    set of points the bounding rectangle with the minimal area.
    The algorithm works by computing the convex hull of the point set, finding the minimal rectangles with sides
    parallel to those of the hull, and choosing the rectangle with the minimal area from them.
    Implementation inspired by:
        https://gis.stackexchange.com/questions/22895/finding-minimum-area-rectangle-for-given-points

    :param points: (Tensor[..., N, 2]) Arbitrarily-dimensional tensor, with the last two dimensions
    containing sets of N points.
    :return: (Tensor[..., 4, 2]) Tensor containing the minimal rectangle for each point set in
    (x0, y0),...,(x3, y3) format, sorted counterclockwise.
    """
    device = points.device

    D = points.shape[0]  # Dimensions excluding cluster size and coordinate number (2)
    # TODO Add support for Arbitrarily-dimensional tensors (current version works only when points.shape[:-2] is 1D).
    N = points.shape[-2]  # Number of points in each set
    # Find convex hull of point set

    hull, sizes = convex_hull(points)

    # Retrieve hull edges and their respective angles relative to the positive x-axis
    edges = torch.zeros(points.shape, device=device)  # [..., N, 2]
    edges[..., 1:, :] = hull[..., 1:, :] - hull[..., :-1, :]
    edges[..., 0, :] = hull[..., 0, :] - hull[torch.arange(D), sizes - 1, :]
    angles = torch.atan2(edges[..., 1], edges[..., 0])  # [..., N]

    rect_min = torch.zeros(D, 4, 2, device=device)  # [..., 4, 2]
    area_min = torch.zeros(D, device=device).fill_(float('inf'))  # [...]

    for k in range(N):
        # Apply rotation by corresponding edge angles
        theta = angles[..., k]
        cos = torch.cos(theta)  # [...]
        sin = torch.sin(theta)  # [...]

        hull_rot = apply_rot(hull, cos, sin, inv=False)  # [..., N, 2]

        # Find minimal and maximal x, y coordinates in the rotated frame
        x_min, _ = torch.min(hull_rot[..., 0].nan_to_num(float('inf')), dim=-1)  # [...]
        x_max, _ = torch.max(hull_rot[..., 0].nan_to_num(-float('inf')), dim=-1)  # [...]
        y_min, _ = torch.min(hull_rot[..., 1].nan_to_num(float('inf')), dim=-1)  # [...]
        y_max, _ = torch.max(hull_rot[..., 1].nan_to_num(-float('inf')), dim=-1)  # [...]

        # Calculate area of resultant rectangle
        area = (x_max - x_min) * (y_max - y_min)

        # Updating rectangle vertices and minimal area for each point set achieving smaller area than current minimum
        mask = (area < area_min) & (sizes > k)
        area_min[mask] = area[mask]

        rect_x = torch.stack([x_min, x_max, x_max, x_min], dim=-1)  # [..., 4]
        rect_y = torch.stack([y_min, y_min, y_max, y_max], dim=-1)  # [..., 4]
        rect = torch.stack([rect_x, rect_y], dim=-1)  # [..., 4, 2]

        rect_min[mask] = apply_rot(rect[mask], cos[mask], sin[mask], inv=True)

    return rect_min


def is_inside_polygon(poly: torch.Tensor, sizes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Checking if point is inside a convex polygon.
    The algorithm works by calculating cross products between segments connecting the point and consecutive vertices
    of the polygon, and the point is inside iff all results have identical signs (equivalent to a winding number of zero).
    Implementation is PyTorch-compliant.

    :param poly: (Tensor[..., N, 2]) Arbitrarily-dimensional tensor, with the last two dimensions
    containing the convex polygons' vertices in counterclockwise order.
    :param sizes: (Tensor[...]) Tensor containing the number of vertices in each polygon.
    :param points: (Tensor[...]) Tensor containing the points to be tested relative to the polygons.
    :return: (Tensor[...]) Boolean tensor indicating whether each point is inside its corresponding polygon
    (1 - Inside, 0 - Outside).
    """
    device = poly.device

    D = poly.shape[0]  # Dimensions excluding number of vertices and coordinate number (2)
    # TODO Add support for Arbitrarily-dimensional tensors (current version works only when points.shape[:-2] is 1D).
    N = poly.shape[-2]  # Maximal possible number of vertices in a polygon

    pos = torch.zeros(D, device=device)
    neg = torch.zeros(D, device=device)

    # Calculate cross product signs
    for k in range(N - 1):
        mask = (k < sizes)
        k_succ = torch.remainder(k + 1, sizes)
        crosses = cross(points, poly[..., k, :], poly[torch.arange(D), k_succ, :])
        pos += (crosses > 0) * mask
        neg += (crosses < 0) * mask

    return torch.logical_or(pos == 0, neg == 0)
