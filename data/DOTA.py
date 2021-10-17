import os
import gdown
import shutil


class DatasetDownloader:
    def __init__(self, path):
        self.path = path
        self._make_directories()

    def _make_directories(self):
        directories = ['train/labelTxt', 'train/images', 'val/labelTxt', 'val/images']
        for directory in directories:
            os.makedirs(f'{self.path}/{directory}', exist_ok=True)

    @property
    def _train_drive_data(self):
        train_drive_data = {'1BlaGYNNEKGmT6OjZjsJ8HoUYrTTmFcO2': f'{self.path}/train/part1.zip',
                            '1JBWCHdyZOd9ULX0ng5C9haAt3FMPXa3v': f'{self.path}/train/part2.zip',
                            '1pEmwJtugIWhiwgBqOtplNUtTG2T454zn': f'{self.path}/train/part3.zip',
                            '1I-faCP-DOxf6mxcjUTc8mYVPqUgSQxx6': f'{self.path}/train/labelTxt/labelTxt.zip'}
        return train_drive_data

    @property
    def _val_drive_data(self):
        val_drive_data = {'1uCCCFhFQOJLfjBpcL5MC0DHJ9lgOaXWP': f'{self.path}/val/part1.zip',
                          '1uFwxA4B7H8zcI1oD11bj0U8z88qroMlG': f'{self.path}/val/labelTxt/labelTxt.zip'}
        return val_drive_data

    def download_data_from_drive(self):
        for drive_data in [self._train_drive_data, self._val_drive_data]:
            for drive_id, zip_path in drive_data.items():
                gdown.download(id=drive_id, output=zip_path)
                shutil.unpack_archive(filename=zip_path, extract_dir='/'.join(zip_path.split('/')[:-1]))
                os.remove(zip_path)
