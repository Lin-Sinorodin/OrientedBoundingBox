import os
import cv2
import gdown
import shutil
import numpy as np
import torch
import torch.utils.data
from typing import List, Tuple

DOTA_V1_5_NAMES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
    'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
    'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter',
    'container-crane'
]

DOTA_NAME_TO_INT = {name: idx for idx, name in enumerate(DOTA_V1_5_NAMES)}


class DatasetDownloader:
    """A class for downloading the DOTAv1.0 dataset from the official website."""

    def __init__(self, path):
        self.path = path
        self._make_directories()

    def _make_directories(self):
        """Create all necessary directories for the dataset before downloading it."""
        directories = ['train/labelTxt', 'train/images', 'val/labelTxt', 'val/images']
        for directory in directories:
            os.makedirs(f'{self.path}/{directory}', exist_ok=True)

    @property
    def _drive_data(self) -> {}:
        """DOTAv1.0 drive id and extraction paths for all files from: https://captain-whu.github.io/DOTA/dataset.html"""
        train_data = {'1BlaGYNNEKGmT6OjZjsJ8HoUYrTTmFcO2': f'{self.path}/train/part1.zip',
                      '1JBWCHdyZOd9ULX0ng5C9haAt3FMPXa3v': f'{self.path}/train/part2.zip',
                      '1pEmwJtugIWhiwgBqOtplNUtTG2T454zn': f'{self.path}/train/part3.zip',
                      '1I-faCP-DOxf6mxcjUTc8mYVPqUgSQxx6': f'{self.path}/train/labelTxt/labelTxt.zip'}

        val_data = {'1uCCCFhFQOJLfjBpcL5MC0DHJ9lgOaXWP': f'{self.path}/val/part1.zip',
                    '1uFwxA4B7H8zcI1oD11bj0U8z88qroMlG': f'{self.path}/val/labelTxt/labelTxt.zip'}

        return {**train_data, **val_data}

    def download_data_from_drive(self):
        """Download all the train and validation data from the drive ids to the given path"""
        for drive_id, zip_path in self._drive_data.items():
            gdown.download(id=drive_id, output=zip_path)
            shutil.unpack_archive(filename=zip_path, extract_dir='/'.join(zip_path.split('/')[:-1]))
            os.remove(zip_path)


class Label:
    def __init__(self, path):
        self.path = path
        self._dota_raw_label = self._load_from_dota()
        self.xyxy = self._xyxy_from_raw_label() if not self.empty else None
        self.xywha = self._xyxy_to_xywha() if not self.empty else None
        self.objects_class = self._objects_class_from_raw_label() if not self.empty else None
        self.objects_class_name = self._objects_class_name_from_raw_label() if not self.empty else None

    def _load_from_dota(self) -> np.array:
        """Load the raw data from DOTA dataset"""
        with open(self.path) as f:
            raw_label = f.read().split('\n')[:-1]
            raw_label = [line.split(' ') for line in raw_label if line[0].isdigit()]
            return np.array(raw_label)

    @property
    def empty(self):
        return self._dota_raw_label.ndim != 2

    def _xyxy_from_raw_label(self) -> np.array:
        """Convert raw DOTA label to ((x1, y1), (x2, y2), (x3, y3), (x4, y4))"""
        return np.float32(self._dota_raw_label[:, :8])

    def _objects_class_from_raw_label(self) -> List[int]:
        """Get objects class idx from raw DOTA label"""
        return [int(DOTA_NAME_TO_INT[name]) for name in self._dota_raw_label[:, 8]]

    def _objects_class_name_from_raw_label(self) -> List[str]:
        """Get objects class name from raw DOTA label"""
        return [name for name in self._dota_raw_label[:, 8]]

    def _xyxy_to_xywha(self) -> np.array:
        """Convert bounding box from ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) to (x, y, w, h, angle)"""
        xywha = []
        for bounding_box in self.xyxy:
            rect = cv2.minAreaRect(bounding_box.reshape((4, 2)))
            x_center, y_center = rect[0]
            width, height = rect[1]
            angle = rect[2]   # [-90, 0)

            xywha.append([x_center, y_center, width, height, angle])

        return np.float32(np.array(xywha))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path: str, obb_format: str = 'xyxy'):
        obb_formats = ['xyxy', 'xywha']
        assert obb_format in obb_formats, f'Unknown obb format: {obb_format}. Available formats: {obb_formats}'

        self.path = path
        self.obb_format = obb_format

        self.images_dir = '/'.join([self.path, 'images'])
        self.labels_dir = '/'.join([self.path, 'labelTxt'])
        self.images_names, self.labels_names = self._get_matching_image_label()
        self.images_paths, self.labels_paths = self._get_matches_paths()

    def _get_matching_image_label(self) -> Tuple[List[str], List[str]]:
        """Find the images that have a proper corresponding label file"""
        images_names = [name.replace('.png', '') for name in os.listdir(self.images_dir)]
        labels_names = [name.replace('.txt', '') for name in os.listdir(self.labels_dir)]

        matching_images_names, matching_labels_names = [], []
        for img_name in images_names:
            if img_name in labels_names:
                matching_images_names.append(f'{img_name}.png')
                matching_labels_names.append(f'{img_name}.txt')

        return matching_images_names, matching_labels_names

    def _get_matches_paths(self) -> Tuple[List[str], List[str]]:
        """from file names ['img1.png', ...] to full path ['path/to/img1.png']"""
        images_paths = [f'{self.images_dir}/{file}' for file in self.images_names]
        labels_paths = [f'{self.labels_dir}/{file}' for file in self.labels_names]
        return images_paths, labels_paths

    def _get_image(self, idx: int) -> torch.tensor:
        return torch.tensor(cv2.imread(self.images_paths[idx])[..., ::-1] / 255).permute(2, 0, 1).float()

    def _get_label(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        label = Label(self.labels_paths[idx])
        if label.empty:
            return torch.tensor([]), torch.tensor([])
        else:
            obb = label.xyxy if self.obb_format == 'xyxy' else label.xywha
            objects_class = label.objects_class
            return torch.tensor(obb), torch.tensor(objects_class)

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx: int):
        img = self._get_image(idx)
        obb, objects_class = self._get_label(idx)
        return img, obb, objects_class


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset_downloader = DatasetDownloader(path='../../../assets/DOTA')
    dataset_downloader.download_data_from_drive()

    train_dataset = Dataset(path='../../../assets/DOTA_sample_data/split')
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # get one sample
    img, obb, object_class = next(iter(train_data_loader))
    print(f'img.shape = {img.shape}')
    print(f'obb.shape = {obb.shape}')
    print(f'object_class.shape = {object_class.shape}')

    # iterate over all dataset
    for img, obb, object_class in train_data_loader:
        print(f'img.shape = {img.shape}')
        print(f'obb.shape = {obb.shape}')
        print(f'object_class.shape = {object_class.shape}')
        print('')
