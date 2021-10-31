# OrientedBoundingBox


<details>
  <summary><b> Progress Log </b></summary>
  
* 17/10/2021 (Lin): Write script for downloading DODAv1.0 dataset.
* 18/10/2021 (Lin): Create a Pytorch `Dataset` and `DataLoader` for DOTA dataset.
* 18/10/2021 (Lin): Add sample data and obb visualization for the data.
* 21/10/2021 (Lin): Add YOLOv5 for Backbone and Neck feature extraction.
* 22/10/2021 (Lin): Add code for 2d Gaussian for rotated bbox.
* 22/10/2021 (Lin): Implement OLA from _'A General Gaussian Heatmap Labeling for Arbitrary-Oriented Object Detection'_ paper.

</details>


<details>
  <summary><b> The Project Structure </b></summary>
  
```
└─ OrientedBoundingBox
   ├─ code
   │   ├─ model
   │   │   ├─ dataset.py
   │   │   ├─ gghl.py
   │   │   └─ yolov5.py
   │   └─ utils
   │       ├─ data.py
   │       ├─ gaussian.py
   │       └─ visualize.py
   ├─ sample_data
   │   ├─ train
   │   │  ├─ images
   │   │  └─ labelTxt
   │   └─ val
   │      ├─ images
   │      └─ labelTxt
   ├─ main.py
   └─ README.md
```
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

<details open>
  <summary><b> How to get the dataset? </b></summary>

* __Option 1__: Download the dataset manually from the [DOTA website](https://captain-whu.github.io/DOTA/dataset.html).
* __Option 2__: (recommended) Download the dataset using the following code:
    ```python
    from code.utils import DatasetDownloader
    
    dataset_downloader = DatasetDownloader(path='example')
    dataset_downloader.download_data_from_drive()
    ```

</details>

<details open>
  <summary><b> Use the Data </b></summary>

* Pytorch `DataLoader`:
    ```python
    import numpy as np
    from torch.utils.data import DataLoader
    from code.model import Dataset
    from code.utils import plot_obb
    
    train_dataset = Dataset(path='sample_data/train')
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    ```

* Show Annotations:
    ```python
    for img, obb, object_class in train_data_loader:
        print(img.shape)
        print(obb.shape)
        print(object_class.shape)
    
        # show an image with oriented bounding box
        img_show = img.squeeze().permute(1, 2, 0).numpy()
        obb_show = np.int16(obb.squeeze().numpy())
        plot_obb(img_show, obb_show)
    
        break
    ```

</details>
