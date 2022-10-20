import os
import cv2
import gdown
import shutil
import numpy as np
import torch
import torch.utils.data
import torchvision
from typing import List, Tuple
from random import choice

from torchvision.transforms.functional import resize, rotate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DOTA_V1_0_NAMES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
    'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
    'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter'
]

DOTA_NAME_TO_INT = {name: idx + 1 for idx, name in enumerate(DOTA_V1_0_NAMES)}


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
            angle = rect[2]  # [-90, 0)

            xywha.append([x_center, y_center, width, height, angle])

        return np.float32(np.array(xywha))


def resize_label(lbl: torch.Tensor, s: float) -> torch.Tensor:
    """Resize gt label"""
    return lbl / s


def resize_image(img: torch.Tensor, s: float) -> torch.Tensor:
    """Scale image by factor s"""
    h, w = img.shape[1:]
    return resize(img, [int(h / s), int(w / s)])


def rotate_label(lbl: torch.Tensor, angle: float, h: float, w: float) -> torch.Tensor:
    """Rotate gt label by given angle"""
    lbl = lbl.to(device)
    center = torch.Tensor([w / 2, h / 2]).to(device)
    lbl_centered = lbl.reshape(-1, 4, 2) - center
    c, s = torch.cos(angle * torch.pi / 180), torch.sin(angle * torch.pi / 180)
    R = torch.stack([c, s, -s, c], dim=-1).reshape(2, 2).to(device)  # Counter-clockwise, left-handed coordinates
    lbl_rotated = R @ lbl_centered.unsqueeze(dim=-1)
    lbl_recentered = center + lbl_rotated.squeeze(dim=-1)
    return lbl_recentered.reshape(-1, 8)


def rotate_image(img: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate image by given angle"""
    return rotate(img, float(angle))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path: str, obb_format: str = 'xyxy', return_features: bool = False, scale_jitter: bool = False, rotation_jitter: bool = False):
        obb_formats = ['xyxy', 'xywha']
        assert obb_format in obb_formats, f'Unknown obb format: {obb_format}. Available formats: {obb_formats}'

        self.path = path
        self.obb_format = obb_format
        self.return_features = return_features
        self.scale_jitter = scale_jitter
        self.rotation_jitter = rotation_jitter
        self.scale_lst = [1, 0.5, 2]
        self.scale_idx = 0

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
        return torchvision.io.read_image(self.images_paths[idx]) / 255.

    def _get_features(self, idx: int) -> torch.tensor:
        features_dir = '/'.join([self.path, 'images/features'])
        img_name = self.images_paths[idx].split('/')[-1].replace('.png', '')

        P3 = torch.load(f'{features_dir}/{img_name}_P3.pt')
        P4 = torch.load(f'{features_dir}/{img_name}_P4.pt')
        P5 = torch.load(f'{features_dir}/{img_name}_P5.pt')

        return P3, P4, P5

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
        obb, objects_class = self._get_label(idx)
        if self.return_features:
            P3, P4, P5 = self._get_features(idx)
            return P3, P4, P5, obb, objects_class
        else:
            img = self._get_image(idx)
            if self.scale_jitter:
                # Rescale sample by current scale
                s = self.scale_lst[self.scale_idx]
                self.scale_idx = (self.scale_idx + 1) % len(self.scale_lst)
                img = resize_image(img, s)
                obb = resize_label(obb, s)
            if self.rotation_jitter and len(obb) > 0:
                # Rotate image by random angle
                angle = -45 + 90 * torch.rand(1)  # Degrees
                h, w = img.shape[1:]
                img = rotate_image(img, angle)
                obb = rotate_label(obb, angle, h, w)
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

    train_dataset_features = Dataset(path='../../../assets/DOTA_sample_data/split', return_features=True)
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    for P3, P4, P5, obb, objects_class in train_dataset_features:
        print(P3.shape, P4.shape, P5.shape)
