import torch
from math import pi


def gen_random_clusters(num_clusters=10, points_per_cluster=9, im_dim=(256, 256), cluster_dim=(40, 40)):
    """ Generates a [B, N, 2] tensor with B clusters, N points each."""

    x_im, y_im = im_dim
    x_cluster, y_cluster = cluster_dim

    # Generate random centers
    x0_lst = torch.randint(0, x_im, (num_clusters,))
    y0_lst = torch.randint(0, y_im, (num_clusters,))

    # Generate random angles uniformly-distributed in the interval [-0.5*pi,0.5*pi)
    theta_lst = torch.rand((num_clusters,)) * pi - 0.5 * pi
    c_lst = torch.cos(theta_lst)
    s_lst = torch.sin(theta_lst)

    # Generate random cluster dimensions
    x_scale_lst = torch.randint(5, x_cluster, (num_clusters,))
    y_scale_lst = torch.randint(5, y_cluster, (num_clusters,))

    # Generate the random points for each cluster
    point_lst = []

    for x0, y0, c, s, x_scale, y_scale in zip(x0_lst, y0_lst, c_lst, s_lst, x_scale_lst, y_scale_lst):
        # Generate random offsets
        dx = torch.randint(0, x_scale, (points_per_cluster,))
        dy = torch.randint(0, y_scale, (points_per_cluster,))

        # Apply rotation
        dx_rot = c * dx - s * dy
        dy_rot = s * dx + c * dy

        points = torch.stack([x0 + dx_rot, y0 + dy_rot], dim=1)  # [N, 2]
        point_lst.append(points)

    return torch.stack(point_lst)  # [B, N, 2]


def cross(o: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.tensor:
    """
    2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.

    :param o: point O for OA and OB vector.
    :param a: point A for OA vector.
    :param b: point B for OB vector.
    :return: positive if OAB makes a counter-clockwise turn, negative for clockwise, zero if the points are collinear.
    """
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])


def _angle_to_point(point, centre):
    """calculate angle in 2-D between points and x axis"""
    delta = point - centre
    res = torch.atan(delta[1] / delta[0])
    if delta[0] < 0:
        res += pi
    return res


def _area_of_triangle(p1, p2, p3):
    """calculate area of any triangle given co-ordinates of the corners"""
    return torch.norm(cross(p1, p2, p3)) / 2.


def convex_hull(points):
    """
    Calculate subset of points that make a convex hull around points.

    This code used unchanged from:
        https://github.com/maudzung/Complex-YOLOv4-Pytorch/blob/564e8e35ad81f5a9f1a24ca4ceaf10908a100bfd/src/utils/convex_hull_torch.py

    Recursively eliminates points that lie inside two neighbouring points until only convex hull is remaining.
    :Parameters:
        points : (m, 2) array of points for which to find hull
    :Returns:
        hull_points : ndarray (n x 2) convex hull surrounding points
    """

    n_pts = points.size(0)
    assert (n_pts >= 4)
    centre = points.mean(0)
    angles = torch.stack([_angle_to_point(point, centre) for point in points], dim=0)
    pts_ord = points[angles.argsort(), :]
    pts = [x[0] for x in zip(pts_ord)]
    prev_pts = len(pts) + 1
    k = 0
    while prev_pts > n_pts:
        prev_pts = n_pts
        n_pts = len(pts)
        i = -2
        while i < (n_pts - 2):
            Aij = _area_of_triangle(centre, pts[i], pts[(i + 1) % n_pts])
            Ajk = _area_of_triangle(centre, pts[(i + 1) % n_pts], pts[(i + 2) % n_pts])
            Aik = _area_of_triangle(centre, pts[i], pts[(i + 2) % n_pts])
            if Aij + Ajk < Aik:
                del pts[i + 1]
            i += 1
            n_pts = len(pts)
        k += 1
    return torch.stack(pts)


def lines_intersection(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor):
    """
    Given points p1 and p2 on line L1, compute the equation of L1 in the format of y = m1 * x + b1.
    Given points p3 and p4 on line L2, compute the equation of L2 in the format of y = m2 * x + b2.

    To compute the point of intersection of the two lines, equate the two line equations together:
                                    m1 * x + b1 = m2 * x + b2
    and solve for x. Once x is obtained, substitute it into one of the equations to obtain the value of y.

    if one of the lines is vertical, then the x-coordinate of the point of intersection will be the
    x-coordinate of the vertical line. Note that there is no need to check if both lines are vertical
    (parallel), since this function is only called if we know that the lines intersect.
    """

    # if first line is vertical
    if p2[0] - p1[0] == 0:
        # slope and intercept of second line
        m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
        b2 = p3[1] - m2 * p3[0]

        x = p1[0]
        y = m2 * x + b2

    # if second line is vertical
    elif p4[0] - p3[0] == 0:
        # slope and intercept of first line
        m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b1 = p1[1] - m1 * p1[0]

        x = p3[0]
        y = m1 * x + b1

    # if neither line is vertical
    else:
        # slope and intercept of first line
        m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b1 = p1[1] - m1 * p1[0]

        # slope and intercept of second line
        m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
        b2 = p3[1] - m2 * p3[0]

        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1

    return torch.stack((x, y)).unsqueeze(0)


def polygon_intersection(gt_polygon: torch.Tensor, clipping_polygon: torch.Tensor):
    """
    Find the intersection polygon between gt_polygon and clipping_polygon.
    Based on: https://github.com/mdabdk/sutherland-hodgman

    Note:
        * it is assumed that requires_grad = True only for clipping_polygon.
        * polygons must be sorted clockwise for the function to work.

    :param gt_polygon: Nx2 tensor with vertices of the ground truth polygon
    :param clipping_polygon: Mx2 tensor with vertices of the other polygon
    :return: Kx2 tensor with vertices of the intersection polygon
    """
    device = clipping_polygon.device

    final_polygon = torch.clone(gt_polygon)
    for i in range(len(clipping_polygon)):
        # stores the vertices of the next iteration of the clipping procedure
        # final_polygon consists of list of 1 x 2 tensors
        next_polygon = torch.clone(final_polygon)

        # stores the vertices of the final clipped polygon. This will be
        # a K x 2 tensor, so need to initialize shape to match this
        final_polygon = torch.empty((0, 2), device=device)

        # these two vertices define a line segment (edge) in the clipping polygon.
        # It is assumed that indices wrap around, such that if i = 0, then i - 1 = M.
        c_edge_start = clipping_polygon[i - 1]
        c_edge_end = clipping_polygon[i]

        for j in range(len(next_polygon)):
            # these two vertices define a line segment (edge) in the subject polygon
            s_edge_start = next_polygon[j - 1]
            s_edge_end = next_polygon[j]

            if cross(c_edge_start, c_edge_end, s_edge_end) <= 0:
                if cross(c_edge_start, c_edge_end, s_edge_start) > 0:
                    intersection = lines_intersection(s_edge_start, s_edge_end, c_edge_start, c_edge_end)
                    final_polygon = torch.cat((final_polygon, intersection), dim=0)
                final_polygon = torch.cat((final_polygon, s_edge_end.unsqueeze(0)), dim=0)
            elif cross(c_edge_start, c_edge_end, s_edge_start) <= 0:
                intersection = lines_intersection(s_edge_start, s_edge_end, c_edge_start, c_edge_end)
                final_polygon = torch.cat((final_polygon, intersection), dim=0)

    return final_polygon


def polygon_area(pts: torch.Tensor) -> torch.Tensor:
    """
    Calculate the area of a polygon.

    :param pts: tensor with vertices of the polygon, shape: [num_vertices, 2]
    :return: the area of the polygon, in a tensor with shape [1]
    """
    roll_pts = torch.roll(pts, -1, dims=0)
    area = (pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]).sum().abs() * 0.5
    return area


def polygon_iou(gt_points, pred_points):
    intersection_points = polygon_intersection(gt_points, pred_points)
    poly1_area = polygon_area(pred_points)
    poly2_area = polygon_area(gt_points)
    intersection_area = polygon_area(intersection_points)
    union_area = poly1_area + poly2_area - intersection_area
    iou = intersection_area / (union_area + 1e-16)
    return iou


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # ~~~~~~ convex hull ~~~~~~
    clusters = gen_random_clusters()
    clusters.requires_grad = True

    plt.figure()
    for cluster in clusters:
        cluster_convex_hull = convex_hull(cluster)
        plt.scatter(cluster.detach().numpy()[:, 0], cluster.detach().numpy()[:, 1], s=0.5)
        plt.fill(cluster_convex_hull.detach().numpy()[:, 0], cluster_convex_hull.detach().numpy()[:, 1], alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ~~~~~~ polygon intersection ~~~~~~
    subject_polygon = [(-1, 1), (1, 1), (3, 0), (1, -1), (-1, -1)]
    clipping_polygon = [(0, 0), (0, 2), (2, 2), (2, 0)]

    subject_polygon = torch.tensor(subject_polygon)
    clipping_polygon = torch.tensor(clipping_polygon).float()
    clipping_polygon.requires_grad = True
    clipped_polygon = polygon_intersection(subject_polygon, clipping_polygon)

    plt.figure()
    plt.scatter(subject_polygon[:, 0], subject_polygon[:, 1], marker='o', alpha=0.7)
    plt.scatter(clipping_polygon.detach().numpy()[:, 0], clipping_polygon.detach().numpy()[:, 1], marker='o', alpha=0.7)
    plt.scatter(clipped_polygon.detach().numpy()[:, 0], clipped_polygon.detach().numpy()[:, 1], marker='x')
    plt.tight_layout()
    plt.show()

    poly1_area = polygon_area(subject_polygon)
    poly2_area = polygon_area(clipping_polygon)
    intersection_area = polygon_area(clipped_polygon)
    union_area = poly1_area + poly2_area - intersection_area
    iou = intersection_area / union_area
    print(f'poly1_area = {poly1_area}')
    print(f'poly2_area = {poly2_area}')
    print(f'intersection_area = {intersection_area}')
    print(f'union_area = {union_area}')
    print(f'iou = {iou}')
