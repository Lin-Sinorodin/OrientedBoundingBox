import os
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class YOLOv5Features(nn.Module):
    def __init__(self, weights_dir='weights', requires_grad=False):
        super().__init__()
        self.weights_dir = weights_dir
        self.requires_grad = requires_grad
        self.weights_path = self._download_weights()
        self.features = self._load_model()

    def _download_weights(self):
        """Download pretrained weights from: https://github.com/hukaixuan19970627/YOLOv5_DOTA_OBB"""
        os.makedirs(self.weights_dir, exist_ok=True)
        weights_path = f'{self.weights_dir}/yolov5.pt'
        if 'yolov5.pt' not in os.listdir(self.weights_dir):
            url = 'https://drive.google.com/file/d/1tS0zvLBTuG2VWTNIlKb15qFKxGh43MTu/view?usp=sharing'
            raise ValueError(f'Please download the weights from {url} to the weights directory.')
        return weights_path

    def _load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'custom', self.weights_path).to(device)

        for k, v in model.named_parameters():
            v.requires_grad = self.requires_grad

        return list(list(list(model.children())[0].children())[0].children())[0][:-1]

    def forward(self, img):
        """Get features from YOLOv5 Backbone+Neck on the given image"""
        prev_layers = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, [-1, 6], -1, -1,
                       -1, [-1, 4], -1, -1, [-1, 14], -1, -1, [-1, 10], -1, [17, 20, 23]]

        x, y = img, []
        for layer_idx, layer in enumerate(self.features):
            prev_layer = prev_layers[layer_idx]
            if prev_layer != -1:  # if current layer is concat or SPP
                if isinstance(prev_layer, int):
                    x = y[prev_layer]
                else:
                    x = [x if j == -1 else y[j] for j in prev_layer]

            x = layer(x)
            y.append(x if layer_idx in [4, 6, 10, 14, 17, 20, 23] else None)

        P3, P4, P5 = [i for i in y if i is not None][-3:]

        P3.stride = 8
        P4.stride = 16
        P5.stride = 32

        return P3, P4, P5


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from obb.utils.dataset import Dataset

    train_dataset = Dataset(path='../../../assets/DOTA_sample_data/split')
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    yolov5 = YOLOv5Features().to(device)

    for img, obb, object_class in train_data_loader:
        img = img.to(device)
        print(img.shape)
        P3, P4, P5 = yolov5(img)
        print(f'{P3.shape}, {P3.stride}\n{P4.shape}, {P4.stride}\n{P5.shape}, {P5.stride}')
        break
