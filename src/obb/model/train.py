import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from obb.utils.dataset import Dataset
from obb.model.custom_model import DetectionModel
from obb.model.oriented_reppoints_loss import OrientedRepPointsLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def elapsed_time(start_time):
    # source: https://stackoverflow.com/a/27780763
    hours, rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    duration = "{:0>2}:{:0>2}:{:02d}".format(int(hours), int(minutes), int(seconds))
    return duration


if __name__ == '__main__':
    train_dataset = Dataset(path='../../../assets/DOTA/train_split')
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    model = DetectionModel().to(device)
    rep_points_loss = OrientedRepPointsLoss(strides=model.feature_map_strides)

    learning_rate = 1e-3
    whight_decay = 5e-5
    epochs = 5

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=whight_decay,
        amsgrad=True
    )

    train_accuracy_dict = {}
    valid_accuracy_dict = {}
    train_loss_dict = {}

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        cnt = 0
        log_step = 100
        epoch_time = time.time()
        for img, obb, object_class in tqdm(train_data_loader):

            img = img.to(device)
            obb = obb.squeeze(dim=0)
            object_class = object_class.squeeze(dim=0)

            rep_points_init, rep_points_refine, classification = model(img)  # forward pass
            loss, cls_loss, reg_loss_init, reg_loss_refine = rep_points_loss.get_loss(rep_points_init,
                                                                                      rep_points_refine,
                                                                                      classification,
                                                                                      obb, object_class)
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()  # backpropagation
            optimizer.step()  # update parameters

            running_loss = (cnt * running_loss + loss.data.item()) / (cnt + 1)
            cnt += 1
            if cnt % log_step == 0:
                print(classification.shape)
                print(object_class)
                print(f'{running_loss:.4f}, {loss.data.item():.4f}, \
                      {cls_loss.data.item():.4f}, {float(reg_loss_init):.4f}, {float(reg_loss_refine):.4f}')

        # calculate loss
        train_loss_dict[epoch] = running_loss
        print(f'{elapsed_time(start_time)}  |  Epoch {str(epoch).ljust(2)}/{epochs}  |  Loss: {running_loss:.02f}')
