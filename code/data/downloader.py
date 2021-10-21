import os
import gdown
import shutil


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
