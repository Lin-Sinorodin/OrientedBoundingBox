import os
import torch
import torchvision

from obb.model.yolov5 import YOLOv5Features

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def export_yolov5_features(path):
    yolov5 = YOLOv5Features('../model/weights').to(device)
    os.makedirs(f'{path}/features', exist_ok=True)

    for img_file in os.listdir(path):
        if not (img_file.endswith('png') or img_file.endswith('jpg')):
            continue

        img = (torchvision.io.read_image(f'{path}/{img_file}') / 255.).unsqueeze(dim=0).to(device)
        P3, P4, P5 = yolov5(img)

        img_name = img_file.split('.')[0]
        torch.save(P3.data, f'{path}/features/{img_name}_P3.pt')
        torch.save(P4.data, f'{path}/features/{img_name}_P4.pt')
        torch.save(P5.data, f'{path}/features/{img_name}_P5.pt')


if __name__ == '__main__':
    export_yolov5_features(path='../../../assets/DOTA_sample_data/split/images')
