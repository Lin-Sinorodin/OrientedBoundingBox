import time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ReduceLROnPlateau
from torch import nn
from tqdm import tqdm
from random import randint
from math import log2

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
    scales = [0.5, 1.0, 1.5]
    train_dataset_lst = [Dataset(path=f'../../../assets/DOTA_scaled/scale_{scale:.1f}/train_split') for scale in scales]
    train_data_loader_lst = [DataLoader(train_dataset, batch_size=1, shuffle=True) for train_dataset in train_dataset_lst]

    model = DetectionModel().to(device)
    rep_points_loss = OrientedRepPointsLoss(strides=model.feature_map_strides)

    epochs = 100000000
    batch_size = 8

    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=1e-3,
                                momentum=0.9,
                                weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, threshold=1e-2)

    train_accuracy_dict = {}
    valid_accuracy_dict = {}
    train_loss_dict = {}

    loss = 0
    epoch = 0

    start_time = time.time()

    # Load previous state
    last_epoch = 0
    if last_epoch > 0:
        checkpoint = torch.load(f'./checkpoints/epoch_{last_epoch}.pt', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    for epoch in range(last_epoch + 1, epochs + 1):
        # choose random scale
        idx = randint(0, len(scales) - 1)
        idx = 0
        scale = scales[idx]
        train_data_loader = train_data_loader_lst[idx]
        print(f'Scale: {scale}')
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        model.train()

        running_loss = 0.0
        running_cls_loss = 0.0
        running_loc_loss_init = 0.0
        running_spc_loss_init = 0.0
        running_reg_loss_refine = 0.0

        cnt = 0
        cnt_pos = 0
        batch_cnt = 0
        loss = torch.Tensor([0.0]).to(device)
        epoch_time = time.time()
        for img, obb, object_class in tqdm(train_data_loader):

            img = img.to(device)
            obb = obb.squeeze(dim=0).to(device)

            object_class = object_class.squeeze(dim=0).to(device)

            rep_points_init, rep_points_refine, classification = model(img)  # forward pass

            curr_loss, loss_dict = rep_points_loss.get_loss(rep_points_init, rep_points_refine, classification, obb, object_class)  # calculate loss
            loss += curr_loss

            batch_cnt += 1

            # if all samples in a single batch have been fed, perform backprop
            if batch_cnt == batch_size:
                loss /= batch_size
                optimizer.zero_grad()  # zero the parameter gradients
                loss.backward()  # backpropagation
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=35, norm_type=2)  # clip gradient
                optimizer.step()  # update parameters

                # Calculate total gradient norm
                total_norm = 0
                parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f'Gradient norm: {total_norm:.4f}')

                batch_cnt = 0

                # update running quantities
                running_loss = (cnt * running_loss + loss.data.item()) / (cnt + 1)
                running_cls_loss = (cnt * running_cls_loss + loss_dict['classification'].data.item()) / (cnt + 1)
                if len(object_class) > 0:
                    running_loc_loss_init = (cnt_pos * running_loc_loss_init + loss_dict['localization_init'].data.item()) / (cnt_pos + 1)
                    running_spc_loss_init = (cnt_pos * running_spc_loss_init + loss_dict['spatial_constraint_init'].data.item()) / (cnt_pos + 1)
                    running_reg_loss_refine = (cnt_pos * running_reg_loss_refine + loss_dict['regression_refine'].data.item()) / (cnt_pos + 1)
                    cnt_pos += 1

                cnt += 1

                print()
                print(f"Loss: Running {running_loss:.4f}  |  Current {float(loss):.4f}")
                print(f"Classification: Running {running_cls_loss:.4f}  |  Current {float(loss_dict['classification']):.4f}")
                print(f"Localization init: Running {running_loc_loss_init:.4f}  |  Current {float(loss_dict['localization_init']):.4f}")
                print(f"Spatial constraint init: Running {running_spc_loss_init:.4f}  |  Current {float(loss_dict['spatial_constraint_init']):.4f}")
                print(f"Regression refine: Running {running_reg_loss_refine:.4f}  |  Current {float(loss_dict['regression_refine']):.4f}")
                print(f'Number of objects: {len(object_class)}')
                print('Class precisions:')
                print([f'{float(prc):.4f}' for prc in loss_dict['precision']])
                print('Class recalls:')
                print([f'{float(rcl):.4f}' for rcl in loss_dict['recall']])

                loss = torch.Tensor([0.0]).to(device)  # reset loss

        # calculate loss
        train_loss_dict[epoch] = running_loss
        print(f'{elapsed_time(start_time)}  |  Epoch {str(epoch).ljust(2)}/{epochs}  |  Loss: {running_loss:.02f}')

        # save state
        state = {
            'epoch': epoch,
            'scale': scale,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': running_loss
        }

        torch.save(state, f'./checkpoints/epoch_{epoch}.pt')

        # update learning rate
        scheduler.step(running_reg_loss_refine)

