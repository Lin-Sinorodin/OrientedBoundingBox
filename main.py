import numpy as np
from torch.utils.data import DataLoader

from data import Dataset, plot_obb

train_dataset = Dataset(path='sample_data/train')
train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

for img, obb, object_class in train_data_loader:
    print(img.shape)
    print(obb.shape)
    print(object_class.shape)

    # show an image with oriented bounding box
    img_show = img.squeeze().permute(1, 2, 0).numpy()
    obb_show = np.int16(obb.squeeze().numpy())
    plot_obb(img_show, obb_show)

    break

