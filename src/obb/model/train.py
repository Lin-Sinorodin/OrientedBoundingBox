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
    epochs = 200

    # optimizer = torch.optim.Adam(
    #     params=model.parameters(),
    #     lr=learning_rate,
    #     weight_decay=weight_decay,
    #     amsgrad=True
    # )

    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=2e-3,
                                momentum=0.9,
                                weight_decay=1e-4)

    train_accuracy_dict = {}
    valid_accuracy_dict = {}
    train_loss_dict = {}

    start_time = time.time()

    # Load previous state
    last_epoch = 1
    checkpoint = torch.load(f'./checkpoints/epoch_{last_epoch}.pt')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    for epoch in range(last_epoch + 1, epochs + 1):
        model.train()

        running_loss = 0.0
        running_cls_loss = 0.0
        running_obj_loss = 0.0
        running_reg_loss_init = 0.0
        running_reg_loss_refine = 0.0

        cnt = 0
        cnt_pos = 0
        log_step = 100
        epoch_time = time.time()
        for img, obb, object_class in tqdm(train_data_loader):

            img = img.to(device)
            obb = obb.squeeze(dim=0)
            object_class = object_class.squeeze(dim=0)

            rep_points_init, rep_points_refine, classification = model(img)  # forward pass

            loss, loss_dict = rep_points_loss.get_loss(rep_points_init, rep_points_refine, classification, obb, object_class)  # calculate loss
                
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()  # backpropagation
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=35, norm_type=2)  # clip gradient
            optimizer.step()  # update parameters

            running_loss = (cnt * running_loss + loss.data.item()) / (cnt + 1)
            running_obj_loss = (cnt * running_obj_loss + loss_dict['objectness'].data.item()) / (cnt + 1)
            if len(object_class) > 0:
                running_cls_loss = (cnt_pos * running_cls_loss + loss_dict['classification'].data.item()) / (cnt_pos + 1)
                running_reg_loss_init = (cnt_pos * running_reg_loss_init + loss_dict['regression_init'].data.item()) / (cnt_pos + 1)
                running_reg_loss_refine = (cnt_pos * running_reg_loss_refine + loss_dict['regression_refine'].data.item()) / (cnt_pos + 1)
                cnt_pos += 1

            cnt += 1
            if cnt % log_step == 0:
                print()
                print(f"Classification: Running {running_cls_loss:.4f}  |  Current {float(loss_dict['classification']):.4f}")
                print(f"Objectness: Running {running_obj_loss:.4f}  |  Current {float(loss_dict['objectness']):.4f}")
                print(f"Regression init: Running {running_reg_loss_init:.4f}  |  Current {float(loss_dict['regression_init']):.4f}")
                print(f"Regression refine: Running {running_reg_loss_refine:.4f}  |  Current {float(loss_dict['regression_refine']):.4f}")
                print(f'Number of objects: {len(object_class)}')
                print('Class precisions:')
                print([f'{float(prc):.4f}' for prc in loss_dict['precision']])
                print('Class recalls:')
                print([f'{float(rcl):.4f}' for rcl in loss_dict['recall']])

        # calculate loss
        train_loss_dict[epoch] = running_loss
        print(f'{elapsed_time(start_time)}  |  Epoch {str(epoch).ljust(2)}/{epochs}  |  Loss: {running_loss:.02f}')

        # save state
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': running_loss
        }
        torch.save(state, f'./checkpoints/epoch_{epoch}.pt')
