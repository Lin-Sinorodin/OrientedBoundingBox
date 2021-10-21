import os
import gdown
import torch


class YOLOv5:
    def __init__(self, weights_dir='weights', cpu=False):
        self.weights_dir = weights_dir
        self.weights_path = self._download_weights()
        self.model = self._load_model()

        # the pretrained model doesn't run on cpu, so use yolov5s for debug on cpu
        if cpu:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        self.model_sequential = list(list(self.model.children())[0].children())[0]
        self.prev_layers = [
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, [-1, 6], -1, -1,
            -1, [-1, 4], -1, -1, [-1, 14], -1, -1, [-1, 10], -1, [17, 20, 23]
        ]

    def _download_weights(self):
        """Download pretrained weights from: https://github.com/hukaixuan19970627/YOLOv5_DOTA_OBB"""
        os.makedirs(self.weights_dir, exist_ok=True)
        weights_path = f'{self.weights_dir}/YOLOv5_DOTAv1.5_OBB.pt'
        if 'YOLOv5_DOTAv1.5_OBB.pt' not in os.listdir(self.weights_dir):
            gdown.download(id='171xlq49JEiKJ3L-UEV9tICXltPs92dLk', output=weights_path)
        return weights_path

    def _load_model(self):
        return torch.hub.load('ultralytics/yolov5', 'custom', self.weights_path)

    def get_features(self, img_cuda):
        """Get features from YOLOv5 Backbone+Neck on the given image"""
        x, y = img_cuda, []
        for layer_idx, layer in enumerate(self.model_sequential[:-1]):
            prev_layer = self.prev_layers[layer_idx]
            if prev_layer != -1:  # if current layer is concat or SPP
                if isinstance(prev_layer, int):
                    x = y[prev_layer]
                else:
                    x = [x if j == -1 else y[j] for j in prev_layer]

            x = layer(x)
            y.append(x if layer_idx in [4, 6, 10, 14, 17, 20, 23] else None)

        P3, P4, P5 = [i for i in y if i != None][-3:]
        return P3, P4, P5
