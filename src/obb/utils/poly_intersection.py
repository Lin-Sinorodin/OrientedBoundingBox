import torch
import warnings


class PolygonClipper:
    """https://github.com/mdabdk/sutherland-hodgman"""
    # POINTS NEED TO BE PRESENTED CLOCKWISE OR ELSE THIS WONT WORK

    def __init__(self, warn_if_empty=True):
        self.warn_if_empty = warn_if_empty

    def is_inside(self, p1, p2, q):
        R = (p2[0] - p1[0]) * (q[1] - p1[1]) - (p2[1] - p1[1]) * (q[0] - p1[0])
        return True if R <= 0 else False

    def compute_intersection(self, p1, p2, p3, p4):

        """
        given points p1 and p2 on line L1, compute the equation of L1 in the
        format of y = m1 * x + b1. Also, given points p3 and p4 on line L2,
        compute the equation of L2 in the format of y = m2 * x + b2.

        To compute the point of intersection of the two lines, equate
        the two line equations together

        m1 * x + b1 = m2 * x + b2

        and solve for x. Once x is obtained, substitute it into one of the
        equations to obtain the value of y.

        if one of the lines is vertical, then the x-coordinate of the point of
        intersection will be the x-coordinate of the vertical line. Note that
        there is no need to check if both lines are vertical (parallel), since
        this function is only called if we know that the lines intersect.
        """

        # if first line is vertical
        if p2[0] - p1[0] == 0:
            x = p1[0]

            # slope and intercept of second line
            m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
            b2 = p3[1] - m2 * p3[0]

            # y-coordinate of intersection
            y = m2 * x + b2

        # if second line is vertical
        elif p4[0] - p3[0] == 0:
            x = p3[0]

            # slope and intercept of first line
            m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
            b1 = p1[1] - m1 * p1[0]

            # y-coordinate of intersection
            y = m1 * x + b1

        # if neither line is vertical
        else:
            m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
            b1 = p1[1] - m1 * p1[0]

            # slope and intercept of second line
            m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
            b2 = p3[1] - m2 * p3[0]

            # x-coordinate of intersection
            x = (b2 - b1) / (m1 - m2)

            # y-coordinate of intersection
            y = m1 * x + b1

        # need to unsqueeze so torch.cat doesn't complain outside func
        intersection = torch.stack((x, y)).unsqueeze(0)

        return intersection

    def clip(self, subject_polygon, clipping_polygon):
        # it is assumed that requires_grad = True only for clipping_polygon.
        # subject_polygon and clipping_polygon are N x 2 and M x 2 torch tensors respectively

        final_polygon = torch.clone(subject_polygon)
        for i in range(len(clipping_polygon)):
            # stores the vertices of the next iteration of the clipping procedure
            # final_polygon consists of list of 1 x 2 tensors
            next_polygon = torch.clone(final_polygon)

            # stores the vertices of the final clipped polygon. This will be
            # a K x 2 tensor, so need to initialize shape to match this
            final_polygon = torch.empty((0, 2))

            # these two vertices define a line segment (edge) in the clipping polygon.
            # It is assumed that indices wrap around, such that if i = 0, then i - 1 = M.
            c_edge_start = clipping_polygon[i - 1]
            c_edge_end = clipping_polygon[i]

            for j in range(len(next_polygon)):
                # these two vertices define a line segment (edge) in the subject polygon
                s_edge_start = next_polygon[j - 1]
                s_edge_end = next_polygon[j]

                if self.is_inside(c_edge_start, c_edge_end, s_edge_end):
                    if not self.is_inside(c_edge_start, c_edge_end, s_edge_start):
                        intersection = self.compute_intersection(s_edge_start, s_edge_end, c_edge_start, c_edge_end)
                        final_polygon = torch.cat((final_polygon, intersection), dim=0)
                    final_polygon = torch.cat((final_polygon, s_edge_end.unsqueeze(0)), dim=0)
                elif self.is_inside(c_edge_start, c_edge_end, s_edge_start):
                    intersection = self.compute_intersection(s_edge_start, s_edge_end, c_edge_start, c_edge_end)
                    final_polygon = torch.cat((final_polygon, intersection), dim=0)

        return final_polygon

    def __call__(self, A, B):
        clipped_polygon = self.clip(A, B)
        if len(clipped_polygon) == 0 and self.warn_if_empty:
            warnings.warn("No intersections found. Are you sure your \
                          polygon coordinates are in clockwise order?")

        return clipped_polygon


def polygon_area(pts: torch.Tensor) -> torch.Tensor:
    """
    Calculate the area of a polygon (differentiable).

    :param pts: tensor with vertices of the polygon, shape: [num_vertices, 2]
    :return: the area of the polygon, in a tensor with shape [1]
    """
    roll_pts = torch.roll(pts, -1, dims=0)
    area = (pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]).sum().abs() * 0.5
    return area


if __name__ == '__main__':
    clip = PolygonClipper()

    # squares
    subject_polygon = [(-1, 1), (1, 1), (3, 0), (1, -1), (-1, -1)]
    clipping_polygon = [(0, 0), (0, 2), (2, 2), (2, 0)]

    subject_polygon = torch.tensor(subject_polygon)
    clipping_polygon = torch.tensor(clipping_polygon).float()
    clipping_polygon.requires_grad = True
    clipped_polygon = clip(subject_polygon, clipping_polygon)

    import matplotlib.pyplot as plt
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
