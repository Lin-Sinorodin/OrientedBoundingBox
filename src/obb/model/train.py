import time
import torch
from torch.utils.data import DataLoader
from torch import nn
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
    weight_decay = 5e-5
    epochs = 10

    # optimizer = torch.optim.Adam(
    #     params=model.parameters(),
    #     lr=learning_rate,
    #     weight_decay=weight_decay,
    #     amsgrad=True
    # )

    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=8e-3,
                                momentum=0.9,
                                weight_decay=1e-4)

    train_accuracy_dict = {}
    valid_accuracy_dict = {}
    train_loss_dict = {}

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        running_cls_loss = 0.0
        running_reg_loss_init = 0.0
        running_reg_loss_refine = 0.0
        running_prcs_pos = 0.0
        running_prcs_neg = 0.0

        cnt = 0
        cnt_pos = 0
        log_step = 100
        epoch_time = time.time()
        for img, obb, object_class in tqdm(train_data_loader):

            img = img.to(device)
            obb = obb.squeeze(dim=0)
            object_class = object_class.squeeze(dim=0)
            if len(object_class) == 0:
                continue

            rep_points_init, rep_points_refine, classification = model(img)  # forward pass

            loss, cls_loss, reg_loss_init, reg_loss_refine, prcs_pos, prcs_neg = rep_points_loss.get_loss(
                rep_points_init,
                rep_points_refine,
                classification,
                obb, object_class)  # calculate loss
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()  # backpropagation
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=35, norm_type=2)  # clip gradient
            optimizer.step()  # update parameters

            running_loss = (cnt * running_loss + loss.data.item()) / (cnt + 1)
            running_cls_loss = (cnt * running_cls_loss + cls_loss.data.item()) / (cnt + 1)
            if len(object_class) > 0:
                running_prcs_pos = (cnt_pos * running_prcs_pos + prcs_pos.data.item()) / (cnt_pos + 1)
                running_reg_loss_init = (cnt_pos * running_reg_loss_init + reg_loss_init.data.item()) / (cnt_pos + 1)
                running_reg_loss_refine = (cnt_pos * running_reg_loss_refine + reg_loss_refine.data.item()) / (cnt_pos + 1)
                cnt_pos += 1
            running_prcs_neg = (cnt * running_prcs_neg + prcs_neg.data.item()) / (cnt + 1)

            cnt += 1
            if cnt % log_step == 0:
                print(f'{running_cls_loss:.4f}, '
                      f'{running_reg_loss_init:.4f}, {running_reg_loss_refine:.4f}, '
                      f'{running_prcs_pos:.4f}, {running_prcs_neg:.4f}\t\t\t\t'
                      f'{cls_loss.data.item():.4f}, '
                      f'{float(reg_loss_init):.4f}, {float(reg_loss_refine):.4f}, '
                      f'{float(prcs_pos):.4f}, {float(prcs_neg):.4f}, {len(object_class)}')

        # calculate loss
        train_loss_dict[epoch] = running_loss
        print(f'{elapsed_time(start_time)}  |  Epoch {str(epoch).ljust(2)}/{epochs}  |  Loss: {running_loss:.02f}')
