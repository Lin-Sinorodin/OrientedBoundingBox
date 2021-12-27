import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from torch.autograd import Variable


class FocalLoss(nn.Module):
    """https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py"""
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha if alpha is not None else 1
        self.size_average = size_average

        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

    def forward(self, x_in, x_target):
        x_target = x_target.view(-1, 1)
        logpt = log_softmax(x_in, dim=1).gather(1, x_target).view(-1)
        loss = - self.alpha * (1 - torch.exp(logpt)) ** self.gamma * logpt
        return loss.mean() if self.size_average else loss.sum()
