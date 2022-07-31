import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from obb.utils.dataset import Dataset
from obb.model.custom_model import DetectionModel
from obb.model.oriented_reppoints_loss import OrientedRepPointsLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_STRIDE = 32

if __name__ == '__main__':
    val_dataset = Dataset(path=f'../../../assets/DOTA_scaled/scale_1.0/test_split')
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    model = DetectionModel().to(device)
    model.eval()
    rep_points_loss = OrientedRepPointsLoss(strides=model.feature_map_strides)

    epochs = 33
    losses = {
        'loss': torch.zeros(epochs),
        'cls': torch.zeros(epochs),
        'reg_init': torch.zeros(epochs),
        'reg_refine': torch.zeros(epochs)
    }

    with torch.no_grad():
        for epoch in range(1, epochs + 1):
            # Load weights of current epoch
            checkpoint = torch.load(f'./checkpoints_rotated/epoch_{epoch}.pt')
            model.load_state_dict(checkpoint['state_dict'])

            running_loss = 0.0
            running_cls_loss = 0.0
            running_reg_init_loss = 0.0
            running_reg_refine_loss = 0.0
            log_step = 10
            cnt = 0
            cnt_pos = 0

            for img, obb, object_class in tqdm(val_data_loader):
                img = img.to(device)
                obb = obb.squeeze(dim=0).to(device)

                object_class = object_class.squeeze(dim=0).to(device)

                try:
                    rep_points_init, rep_points_refine, classification = model(img)  # forward pass
                except RuntimeError:
                    print('Image too large')
                    continue

                loss, loss_dict = rep_points_loss.get_loss(rep_points_init, rep_points_refine, classification, obb, object_class)  # calculate loss

                running_loss = (cnt * running_loss + loss) / (cnt + 1)  # update running loss
                running_cls_loss = (cnt * running_cls_loss + loss_dict['classification']) / (cnt + 1)
                cnt += 1

                if len(object_class) > 0:
                    running_reg_init_loss = (cnt_pos * running_reg_init_loss + loss_dict['regression_init']) / (cnt_pos + 1)
                    running_reg_refine_loss = (cnt_pos * running_reg_refine_loss + loss_dict['regression_refine']) / (cnt_pos + 1)
                    cnt_pos += 1

                if cnt % log_step == 0:
                    print(f'Running loss: {float(running_loss):.4f}')

            print(f'Epoch {str(epoch).ljust(2)}/{epochs}  |  Loss: {float(running_loss):.02f}')

            # save loss value
            losses['loss'][epoch - 1] = running_loss
            losses['cls'][epoch - 1] = running_cls_loss
            losses['reg_init'][epoch - 1] = running_reg_init_loss
            losses['reg_refine'][epoch - 1] = running_reg_refine_loss

        # save loss list
        torch.save(losses, f'./val_losses_rotated.pt')
