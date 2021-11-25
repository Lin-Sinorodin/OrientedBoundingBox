import os
import cv2
import gdown
import shutil
import numpy as np

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
        self.xyxyxyxy, self.objects_class = self._load_from_dota()

    def _load_from_dota(self):
        """Load a bounding box in the DOTA format ((x1, y1), (x2, y2), (x3, y3), (x4, y4))"""
        with open(self.path) as f:
            raw_label = f.read().split('\n')[:-1]
            raw_label = [line.split(' ') for line in raw_label if line[0].isdigit()]
            raw_label = np.array(raw_label)

            bounding_boxes = np.float32(raw_label[:, :8])
            objects_class = [int(DOTA_NAME_TO_INT[name]) for name in raw_label[:, 8]]
            return bounding_boxes, objects_class

    def _xyxyxyxy_to_xywha(self):
        xywha = []
        for bounding_box in self.xyxyxyxy:
            rect = cv2.minAreaRect(bounding_box.reshape((4, 2)))
            x_center, y_center = rect[0]
            width, height = rect[1]
            angle = rect[2]   # [-90, 0)

            xywha.append([x_center, y_center, width, height, angle])

        return np.float32(np.array(xywha))

    @property
    def xywha(self):
        return self._xyxyxyxy_to_xywha()

