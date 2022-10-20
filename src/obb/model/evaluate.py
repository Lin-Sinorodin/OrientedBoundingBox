import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm

from obb.utils.dataset import Dataset
from obb.model.custom_model import DetectionModel
from obb.model.oriented_reppoints_loss import OrientedRepPointsLoss
from obb.utils.box_ops import rep_points_to_gaussian, kl_divergence_gaussian, gaussian_to_xywha, cs_to_angle
from obb.utils.infer_ops import flatten_head_output, nms_kl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_STRIDE = 32
NUM_CLASSES = 15
NUM_OFFSETS = 9

if __name__ == '__main__':
    eval_dataset = Dataset(path=f'../../../assets/DOTA_scaled/scale_1.0/test_split')
    eval_data_loader = DataLoader(eval_dataset, batch_size=1, shuffle=True)

    model = DetectionModel().to(device)
    model.eval()
    rep_points_loss = OrientedRepPointsLoss(strides=model.feature_map_strides)

    thresholds = torch.arange(0.05, 1, 0.05)
    print(f'Thresholds: {thresholds}')

    epoch = 12

    with torch.no_grad():
        # Load weights from chosen epoch
        checkpoint = torch.load(f'./checkpoints_ablation/new/epoch_{epoch}.pt')
        model.load_state_dict(checkpoint['state_dict'])

        running_loss = 0.0
        running_cls_loss = 0.0
        running_reg_init_loss = 0.0
        running_reg_refine_loss = 0.0
        log_step = 10
        cnt = 0
        cnt_cls = torch.zeros(NUM_CLASSES).to(device)

        confusion_mats = torch.zeros(len(thresholds), NUM_CLASSES + 1, NUM_CLASSES + 1).long().to(device)
        oes = torch.zeros(len(thresholds), NUM_CLASSES).to(device)

        for img, obb, object_class in tqdm(eval_data_loader):

            img = img.to(device)
            obb = obb.squeeze(dim=0).to(device)

            gt_mu, gt_S = rep_points_to_gaussian(obb.reshape(-1, 4, 2))  # convert gt to Gaussian distribution
            gt_S /= 9  # gts correspond to 3 standard deviations

            object_class = object_class.squeeze(dim=0).to(device)

            try:
                _, rep_points_refine, classification = model(img)  # forward pass
            except RuntimeError:
                print('Image too large')
                continue

            # flatten output
            classification_flattened, rep_points_flattened = flatten_head_output(classification, rep_points_refine)

            for thr_idx, thr in enumerate(thresholds):
                print(thr_idx)
                # perform NMS
                classification_nms, rep_points_nms = nms_kl(classification_flattened, rep_points_flattened,
                                                            cls_thr=thr, nms_thr=10)

                if len(classification_nms) == 0:
                    for cls in object_class:
                        confusion_mats[thr_idx, -1, cls - 1] += 1
                    continue
                classification_hard = torch.argmax(classification_nms, dim=1) + 1

                # convert predicted RepPoints to Gaussian distribution
                pred_mu, pred_S = rep_points_to_gaussian(rep_points_nms.reshape(-1, NUM_OFFSETS, 2))

                # compute cost matrix (where the cost is KL divergence)
                num_gt, num_pred = len(gt_mu), len(pred_mu)
                cost_mat_flattened = kl_divergence_gaussian(gt_mu.repeat(num_pred, 1),
                                                            gt_S.repeat(num_pred, 1, 1),
                                                            pred_mu.repeat_interleave(num_gt, dim=0),
                                                            pred_S.repeat_interleave(num_gt, dim=0), batched=True)
                cost_mat = cost_mat_flattened.reshape(num_pred, num_gt)

                # find optimal assignment between predictions and gts (no choice, has to be done in cpu)
                try:
                    pred_idx, gt_idx = linear_sum_assignment(cost_mat.cpu().numpy())
                except ValueError:
                    print('Cost matrix is infeasible')
                    continue
                pred_idx = torch.Tensor(pred_idx).long().to(device)
                gt_idx = torch.Tensor(gt_idx).long().to(device)
                cls_pred = torch.argmax(classification_nms[pred_idx], dim=1) + 1
                cls_gt = object_class[gt_idx]

                # update confusion matrix
                for cp, cg in zip(cls_pred, cls_gt):
                    confusion_mats[thr_idx, cp - 1, cg - 1] += 1

                # update background FP
                for cls in object_class:
                    confusion_mats[thr_idx, -1, cls - 1] += 1
                for cls in cls_gt:
                    confusion_mats[thr_idx, -1, cls - 1] -= 1

                # update background FN
                for cls in classification_hard:
                    confusion_mats[thr_idx, cls - 1, -1] += 1
                for cls in cls_pred:
                    confusion_mats[thr_idx, cls - 1, -1] -= 1

                # compute average orientation error
                if len(object_class) > 0 and len(pred_idx) > 0:
                    xywha_pred = gaussian_to_xywha(pred_mu[pred_idx], pred_S[pred_idx])
                    c_pred, s_pred = xywha_pred[:, -2], xywha_pred[:, -1]
                    xywha_gt = gaussian_to_xywha(gt_mu[gt_idx], gt_S[gt_idx])
                    c_gt, s_gt = xywha_gt[:, -2], xywha_gt[:, -1]
                    alpha_pred, alpha_gt = cs_to_angle(c_pred, s_pred), cs_to_angle(c_gt, s_gt)
                    angle_diff = (torch.abs(alpha_pred - alpha_gt) * (180 / torch.pi)) % 360
                    for cls in range(1, NUM_CLASSES + 1):
                        if (object_class == cls).any():
                            oes[thr_idx, cls - 1] = (cnt_cls[cls - 1] * oes[thr_idx, cls - 1]
                                                     + angle_diff[cls_gt == cls].sum()) / \
                                                    (cnt_cls[cls - 1] + torch.count_nonzero(object_class == cls))

            if len(object_class) > 0:
                for cls in range(1, NUM_CLASSES + 1):
                    cnt_cls[cls - 1] += torch.count_nonzero(object_class == cls)

            cnt += 1

    confusion_mat_dict = {
        'thrs': thresholds,
        'mats': confusion_mats,
        'oes': oes
    }

    print(confusion_mat_dict['mats'][10])
    print(confusion_mat_dict['oes'][10])

    # save loss list
    torch.save(confusion_mat_dict, f'./confusion_mats_new.pt')
