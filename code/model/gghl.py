"""
This is an implementation of the paper GGHL:
    'A General Gaussian Heatmap Labeling for Arbitrary-Oriented Object Detection'
"""
import numpy as np
from code.utils import Gaussian2D


class OLA:
    """
    This is an implementation of object-adaptation label assignment (OLA), from the paper:
        'A General Gaussian Heatmap Labeling for Arbitrary-Oriented Object Detection'
    """

    def __init__(self, img, bboxs, strides, T_IoU=0.3, gaussian_focus=1 / 2, gaussian_min_value=0.05):
        self.img = img
        self.bboxs = bboxs  # [[x_center, y_center, width, height, angle], ...]
        self.strides = strides
        self.T_IoU = T_IoU
        self.gaussian_properties = {'focus': gaussian_focus, 'min_value': gaussian_min_value}
        _, self.W, self.H = img.shape

        self.range_mins, self.range_maxs = self._get_assignment_ranges()
        self.bboxs_feature_maps = self._get_bboxs_feature_maps()
        self.gaussian_key_points = self._get_bboxs_gaussian_key_points

    def _get_assignment_ranges(self):
        """
        Get the assignment ranges that links bbox to certain feature map. See eq (8).
        """
        strides_sep = [self.strides[0], self.strides[-1]]
        ranges_sep = [(3 * 2 * stride) / (1 - self.T_IoU) for stride in strides_sep]
        range_mins = [1] + ranges_sep
        range_maxs = ranges_sep + [np.sqrt(2) * max(self.W, self.H)]
        return range_mins, range_maxs

    def _get_bboxs_feature_maps(self):
        """
        Get the selected feature map for each bbox, depends on it's assignment range.  See eq (8).
        """
        bboxs_feature_maps = []
        for bbox in self.bboxs:
            x, y, w, h, angle = bbox
            for feature_map, (range_min, range_max) in enumerate(zip(self.range_mins, self.range_maxs)):
                if range_min < max(w, h) < range_max:
                    bboxs_feature_maps.append(feature_map)
                    break

        return bboxs_feature_maps

    def _get_bboxs_gaussian_key_points(self):
        """
        For each bbox, find it's gaussian feature map according to the feature map size, and store
        it in a list (each element is with the shape of the corresponding feature map)
        """
        gaussian_key_points = []
        for bbox, feature_map in zip(self.bboxs, self.bboxs_feature_maps):
            stride = self.strides[feature_map]
            stride_bbox = np.round(bbox / stride)
            stride_bbox[..., 4] = bbox[..., 4]
            x, y, w, h, angle = stride_bbox

            gaussian_2d = Gaussian2D(self.W // stride, self.H // stride, **self.gaussian_properties)
            gaussian_key_points.append(gaussian_2d.from_bbox(x, y, w, h, angle))

        return gaussian_key_points


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = np.random.rand(3, 800, 800)
    strides = [16, 32, 64]
    bboxs = np.array([
        [100, 100, 30, 10, np.pi / 4],
        [400, 100, 50, 100, np.pi / 3],
        [400, 150, 50, 100, np.pi / 3],
        [100, 500, 200, 110, -np.pi / 4],
        [400, 400, 550, 100, np.pi / 4]
    ])

    ola = OLA(img, bboxs, strides)
    key_points = ola.gaussian_key_points()

    for key_point in key_points:
        plt.imshow(key_point)
        plt.show()
