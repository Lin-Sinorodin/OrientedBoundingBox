<h1 align="center">Oriented Object Detection on Satellite Images</h1>

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

[![Tests](https://github.com/Lin-Sinorodin/OrientedBoundingBox/actions/workflows/tests.yaml/badge.svg)](https://github.com/Lin-Sinorodin/OrientedBoundingBox/actions/workflows/tests.yaml)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

</div>


<details>
  <summary><b> Progress Log </b></summary>
  
* 17/10/2021 (Lin): Write script for downloading DODAv1.0 dataset
* 18/10/2021 (Lin): Create a Pytorch `Dataset` and `DataLoader` for DOTA dataset
* 18/10/2021 (Lin): Add sample data and obb visualization for the data
* 21/10/2021 (Lin): Add YOLOv5 for Backbone and Neck feature extraction
* 22/10/2021 (Lin): Add code for 2d Gaussian for rotated bbox
* 22/10/2021 (Lin): Implement OLA from _'A General Gaussian Heatmap Labeling for Arbitrary-Oriented Object Detection'_ paper
* 20/11/2021 (Lin): Implement custom _Feature Map_ (backbone+neck based on YOLOv5 and TPH-YOLOv5)
* 28/11/2021 (Lin): Implement offset initialization for RepPoints paper
* 10/12/2021 (Lin): Implement head architecture for RepPoints paper
* 10/12/2021 (Lin): Add automatic testing with GitHub actions
* 14/12/2021 (Matan): Add convex hull and minimum bounding rectangle functionality (see [notebook](https://github.com/Lin-Sinorodin/OrientedBoundingBox/blob/main/notebooks/bbox.ipynb))
* 15/12/2021 (Lin): combine oriented rep points head with backbone and neck

</details>


<details>
  <summary><b> The Project Structure </b></summary>
  
```
└─ OrientedBoundingBox
   ├─ assets
   ├─ DOTA_sample_data
   │  ├─ train
   │  │  ├─ images
   │  │  └─ labelTxt
   │  └─ val
   │     ├─ images
   │     └─ labelTxt
   ├─ src
   │  ├─ model
   │  │  ├─ common.py
   │  │  ├─ custom_model.py
   │  │  ├─ feature_map.py
   │  │  ├─ gghl.py
   │  │  ├─ oriented_reppoints.py
   │  │  ├─ rep_points.py
   │  │  └─ yolov5.py
   │  └─ utils
   │     ├─ box_ops.py
   │     ├─ dataset.py
   │     ├─ gaussian.py
   │     └─ visualize.py
   ├─ pyproject.toml
   ├─ README.md
   ├─ requirements.txt
   ├─ setup.cfg
   └─ setup.py
```
</details>


<details>
  <summary><b> Theoretical Background </b></summary>

| Paper 	| Implementation  	| About 	|
|------	    |:----------------: |---------	|
| [ReDet: A Rotation-equivariant Detector for Aerial Object Detection](https://arxiv.org/abs/2103.07733) | [Official](https://github.com/csuhan/ReDet) | |
| [RepPoints: Point Set Representation for Object Detection](https://arxiv.org/abs/1904.11490) | [Official](https://github.com/microsoft/RepPoints) | |
| [Oriented RepPoints for Aerial Object Detection](https://arxiv.org/abs/2105.11111) | [Official](https://github.com/LiWentomng/OrientedRepPoints), [w. Swin Transformer](https://github.com/hukaixuan19970627/OrientedRepPoints_DOTA) | |
| [FCOSR: A Simple Anchor-free Rotated Detector for Aerial Object Detection](https://arxiv.org/abs/2111.10780) | [Official](https://github.com/lzh420202/fcosr) | |
| [A General Gaussian Heatmap Labeling for Arbitrary-Oriented Object Detection](https://arxiv.org/abs/2109.12848) | [Official](https://github.com/Shank2358/GGHL) | |
| [Learning High-Precision Bounding Box for Rotated Object Detection via Kullback-Leibler Divergence](https://arxiv.org/abs/2106.01883) | [Official](https://github.com/yangxue0827/RotationDetection) | |
| [Boosting object detection performance through ensembling on satellite imagery](https://medium.com/earthcube-stories/boosting-object-detection-performance-through-ensembling-on-satellite-imagery-949e891dfb28) | | |
| [G-Rep: Gaussian Representation for Arbitrary-Oriented Object Detection](https://arxiv.org/abs/2205.11796) | | |


* [Oriented Bounding Boxes for Small and Freely Rotated Objects](https://arxiv.org/pdf/2104.11854.pdf) - contains nice 
  figures demonstrates how the backbone and neck (multi-scale feature pyramid) works

</details>

## DOTAv1.0 Dataset

<details>
  <summary><b> The dataset structure </b></summary>
  
```
└─ DOTAv1
   ├─ train
   │  ├─ images
   │  │  ├─ file1.png
   │  │  └─ file2.png
   │  └─ labelTxt
   │     ├─ file1.txt
   │     └─ file2.txt
   └─ val
      ├─ images
      │  ├─ file3.png
      │  └─ file4.png
      └─ labelTxt
         ├─ file3.txt
         └─ file4.txt
```
</details>

<details>
  <summary><b> The OBB annotation format </b></summary>
  
```
x1, y1, x2, y2, x3, y3, x4, y4, category, difficult
x1, y1, x2, y2, x3, y3, x4, y4, category, difficult
...
```
</details>

<details>
  <summary><b> How to get the dataset? </b></summary>

* __Option 1__: Download the dataset manually from the [DOTA website](https://captain-whu.github.io/DOTA/dataset.html).
* __Option 2__: (recommended) Download the dataset using the following code:
    ```python
    from obb.utils import DatasetDownloader
    
    dataset_downloader = DatasetDownloader(path='example')
    dataset_downloader.download_data_from_drive()
    ```

</details>

<details>
  <summary><b> Use the Data </b></summary>

* Get a Pytorch `DataLoader`:
    ```python
    import numpy as np
    from torch.utils.data import DataLoader
  
    from obb.utils import Dataset, plot_obb
    
    train_dataset = Dataset(path='assets/DOTA_sample_data/train')
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
  
    # get one sample
    img, obb, object_class = next(iter(train_data_loader))
  
    # iterate over all samples
    for img, obb, object_class in train_data_loader:
        ...

    ```

* Show Annotations:
    ```python
    img, obb, object_class = next(iter(train_data_loader))
    print(img.shape)
    print(obb.shape)
    print(object_class.shape)
    
    # show an image with oriented bounding box
    img_show = img.squeeze().permute(1, 2, 0).numpy()
    obb_show = np.int16(obb.squeeze().numpy())
    plot_obb(img_show, obb_show)
    ```

</details>
