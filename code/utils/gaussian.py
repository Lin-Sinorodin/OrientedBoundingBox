import numpy as np


class Gaussian2D:
    """
    An implementation of a 2D Gaussian on rotated bounding box.
    vectorized calculation based on:
        https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
    """
    def __init__(self, axis_w, axis_h, focus=1/2, min_value=0):
        self.focus = focus            # how focused on the center the gaussian is
        self.min_value = min_value    # make all values below min_value 0

        X, Y = np.meshgrid(np.linspace(0, axis_w, axis_w), np.linspace(0, axis_h, axis_h))
        self.pos = np.empty(X.shape + (2,))
        self.pos[:, :, 0] = X
        self.pos[:, :, 1] = Y

    def from_bbox(self, x_center, y_center, width, height, angle):
        """Return the multivariate Gaussian distribution on array pos."""
        eig1, eig2 = (self.focus * width/2)**2, (self.focus * height/2)**2  # eig = semi_axis^2
        sin, cos = np.sin(angle), np.cos(angle)
        mu = np.array([x_center, y_center])

        cov_inv = np.array([
            [sin**2/eig2 + cos**2/eig1, cos*sin*(eig2-eig1)/(eig2*eig1)],
            [cos*sin*(eig2-eig1)/(eig2*eig1), sin**2/eig1 + cos**2/eig2]
        ])

        gaussian = np.exp(-np.einsum('...k,kl,...l->...', self.pos-mu, cov_inv, self.pos-mu) / 2)
        gaussian = gaussian / np.max(gaussian)
        gaussian[gaussian < self.min_value] = 0
        return gaussian
