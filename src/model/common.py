"""
This script contains common blocks for Pytorch and general neural network models.
Many of the functions here inspired by the following sources (Thanks for they great work!):
    1. https://github.com/ultralytics/yolov5
    2. https://github.com/tmp-iclr/convmixer
"""

import warnings
import torch
import torch.nn as nn
from typing import Union, Tuple


def same_padding(kernel_size: Union[int, Tuple[int, int]]) -> Union[int, Tuple[int, int]]:
    """Get same padding from kernel_size, Supports both int and (int, int) format"""
    return kernel_size//2 if isinstance(kernel_size, int) else [x//2 for x in kernel_size]


class Conv(nn.Module):
    """Convolution block with batch normalization and activation"""
    def __init__(self, channel_in: int, channel_out: int, kernel_size: int = 1, stride: int = 1,
                 padding=None, activation=True):
        super().__init__()
        if padding is None:
            padding = same_padding(kernel_size)
        self.conv = nn.Conv2d(channel_in, channel_out, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(channel_out)
        self.act = nn.SiLU() if activation is True else (activation if isinstance(activation, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    """Transformer layer (https://arxiv.org/abs/2010.11929). LayerNorm layers removed for better performance"""
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    """Vision Transformer (https://arxiv.org/abs/2010.11929)"""
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    """Bottleneck layer. TODO add explanation about it"""
    def __init__(self, channel_in, channel_out, shortcut=True, expansion=0.5):
        super().__init__()
        channel_hidden = int(channel_out * expansion)
        self.cv1 = Conv(channel_in, channel_hidden, 1, 1)
        self.cv2 = Conv(channel_hidden, channel_out, 3, 1)
        self.add = shortcut and channel_in == channel_out

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class CSP3(nn.Module):
    """CSP (Cross Stage Partial) Bottleneck with 3 convolutions"""
    def __init__(self, channel_in, channel_out, number=1, shortcut=True, expansion=1.):
        super().__init__()
        channel_hidden = int(channel_out * expansion)
        self.cv1 = Conv(channel_in, channel_hidden, 1, 1)
        self.cv2 = Conv(channel_in, channel_hidden, 1, 1)
        self.cv3 = Conv(2 * channel_hidden, channel_out, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(channel_hidden, channel_hidden, shortcut, expansion=1.) for _ in range(number)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class CSP3Transformer(CSP3):
    """C3 module with TransformerBlock()"""
    def __init__(self, channel_in, channel_out, number=1, shortcut=True, expansion=1.):
        super().__init__(channel_in, channel_out, number, shortcut, expansion)
        channel_hidden = int(channel_out * expansion)
        self.m = TransformerBlock(channel_hidden, channel_hidden, 4, number)


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher"""
    def __init__(self, channel_in, channel_out, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        channel_hidden = channel_in//2
        self.conv1 = Conv(channel_in, channel_hidden, 1, 1)
        self.conv2 = Conv(channel_hidden * 4, channel_out, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)

    def forward(self, x):
        x = self.conv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.conv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension"""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim, depth, kernel_size=7):
    """https://github.com/tmp-iclr/convmixer"""
    return nn.Sequential(
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
    )
