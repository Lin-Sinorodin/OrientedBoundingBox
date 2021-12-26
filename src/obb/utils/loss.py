import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input).gather(1, target).view(-1)
        pt = Variable(logpt.data.exp())
        loss = - self.alpha * (1 - pt) ** self.gamma * logpt
        return loss.mean() if self.size_average else loss.sum()
