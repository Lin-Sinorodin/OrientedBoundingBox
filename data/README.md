# DOTA-v1.0 Dataset 

## How to Get the Dataset

* __Option 1__: Download the dataset manually from the [DOTA website](https://captain-whu.github.io/DOTA/dataset.html).
* __Option 2__: (recommended) Download the dataset using the following code:
    ```python
    from data import DOTA
    
    dataset_downloader = DOTA.DatasetDownloader(path='example')
    dataset_downloader.download_data_from_drive()
    ```

<details>
  <summary> The dataset structure:</summary>
  
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
  <summary> The OBB anotation format:</summary>
  
```
x1, y1, x2, y2, x3, y3, x4, y4, category, difficult
x1, y1, x2, y2, x3, y3, x4, y4, category, difficult
...
```
</details>