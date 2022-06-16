import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from obb.utils.dataset import Dataset
from obb.model.custom_model import DetectionModel
from obb.model.oriented_reppoints_loss import OrientedRepPointsLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    train_dataset = Dataset(path=f'../../../assets/DOTA/val')
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    model = DetectionModel().to(device)
    model.eval()
    rep_points_loss = OrientedRepPointsLoss(strides=model.feature_map_strides)

    epochs = 40
    losses = torch.zeros(epochs)

    with torch.no_grad():
        for epoch in range(1, epochs + 1):
            # Load weights of current epoch
            checkpoint = torch.load(f'./checkpoints_augmented/epoch_{epoch}.pt', map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])

            running_loss = 0.0
            log_step = 10
            cnt = 0

            for img, obb, object_class in tqdm(train_data_loader):

                img = img.to(device)
                if img.shape[1] != 3:
                    continue

                h, w = img.shape[-2:]
                img = img[:, :3, :(h//32)*32, :(w//32)*32]
                obb = obb.squeeze(dim=0).to(device)

                object_class = object_class.squeeze(dim=0).to(device)

                rep_points_init, rep_points_refine, classification = model(img)  # forward pass

                loss, loss_dict = rep_points_loss.get_loss(rep_points_init, rep_points_refine, classification, obb, object_class)  # calculate loss

                running_loss = (cnt * running_loss + loss) / (cnt + 1)  # update running loss
                cnt += 1

                if cnt % log_step == 0:
                    print(f'Running loss: {float(running_loss):.4f}')

            print(f'Epoch {str(epoch).ljust(2)}/{epochs}  |  Loss: {float(running_loss):.02f}')

            # save loss value
            losses[epoch - 1] = running_loss

        # save loss list
        torch.save(losses, f'./checkpoints_mega/val_losses.pt')