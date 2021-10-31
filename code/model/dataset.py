import os
import cv2
import numpy as np
import torch
import torch.utils.data

from code.utils import Label

DOTA_V1_5_NAMES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
    'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
    'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter',
    'container-crane'
]

DOTA_NAME_TO_INT = {name: idx for idx, name in enumerate(DOTA_V1_5_NAMES)}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, obb_format='xyxyxyxy'):
        assert obb_format in ['xyxyxyxy', 'xywha'], 'Unknown obb format'

        self.path = path
        self.obb_format = obb_format

        self.images_dir = '/'.join([self.path, 'images'])
        self.labels_dir = '/'.join([self.path, 'labelTxt'])
        self.images_names, self.labels_names = self._get_matching_image_label()
        self.images_paths, self.labels_paths = self._get_matches_paths()

    def _get_matching_image_label(self) -> ([], []):
        """Find the images that have a proper corresponding label file"""
        images_names = [name.replace('.png', '') for name in os.listdir(self.images_dir)]
        labels_names = [name.replace('.txt', '') for name in os.listdir(self.labels_dir)]

        matching_images_names, matching_labels_names = [], []
        for img_name in images_names:
            if img_name in labels_names:
                matching_images_names.append(f'{img_name}.png')
                matching_labels_names.append(f'{img_name}.txt')

        return matching_images_names, matching_labels_names

    def _get_matches_paths(self) -> ([], []):
        """from file names ['img1.png', ...] to full path ['path/to/img1.png']"""
        images_paths = [f'{self.images_dir}/{file}' for file in self.images_names]
        labels_paths = [f'{self.labels_dir}/{file}' for file in self.labels_names]
        return images_paths, labels_paths

    def _get_image(self, idx) -> torch.tensor:
        return torch.tensor(cv2.imread(self.images_paths[idx])[..., ::-1] / 255).permute(2, 0, 1).float()

    def _get_label(self, idx) -> (torch.tensor, torch.tensor):
        label = Label(self.labels_paths[idx])
        obb = label.xyxyxyxy if self.obb_format == 'xyxyxyxy' else label.xywha
        objects_class = label.objects_class
        return torch.tensor(obb), torch.tensor(objects_class)

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        img = self._get_image(idx)
        obb, objects_class = self._get_label(idx)
        return img, obb, objects_class
