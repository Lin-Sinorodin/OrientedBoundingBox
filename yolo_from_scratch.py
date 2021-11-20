
import warnings
import torch
import torch.nn as nn


def same_padding(kernel_size):
    """Get same padding from kernel_size, Supports both int and (int, int) format"""
    return kernel_size//2 if isinstance(kernel_size, int) else [x//2 for x in kernel_size]


class Conv(nn.Module):
    """Convolution block with batch normalization and activation"""
    def __init__(self, channel_in, channel_out, kernel_size=1, stride=1, padding=None, activation=True):
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
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
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
    # Vision Transformer https://arxiv.org/abs/2010.11929
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
    # C3 module with TransformerBlock()
    def __init__(self, channel_in, channel_out, number=1, shortcut=True, expansion=1.):
        super().__init__(channel_in, channel_out, number, shortcut, expansion)
        channel_hidden = int(channel_out * expansion)
        self.m = TransformerBlock(channel_hidden, channel_hidden, 4, number)


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
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
    # Concatenate a list of tensors along dimension
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


if __name__ == '__main__':
    backbone = [
        ['Conv', {'channel_out': 64, 'kernel_size': 6, 'stride': 2, 'padding': 2}],   # 0 - 3xWxH -> 64x(W/2)x(H/2)
        ['Conv', {'channel_out': 128, 'kernel_size': 3, 'stride': 2}],                # 1 -
        ['CSP3', {'channel_out': 128, 'number': 3, 'shortcut': True}],                # 2 -
        ['Conv', {'channel_out': 256, 'kernel_size': 3, 'stride': 2}],                # 3 - P3/8
        ['CSP3', {'channel_out': 256, 'number': 6, 'shortcut': True}],                # 4 - P3/8
        ['Conv', {'channel_out': 512, 'kernel_size': 3, 'stride': 2}],                # 5 - P4/16
        ['CSP3', {'channel_out': 512, 'number': 9, 'shortcut': True}],                # 6 - P4/16
        ['Conv', {'channel_out': 1024, 'kernel_size': 3, 'stride': 2}],               # 7 - P5/32
        ['CSP3Transformer', {'channel_out': 1024, 'number': 3, 'shortcut': True}],    # 8 - P5/32
        ['SPPF', {'channel_out': 1024, 'k': 5}],
    ]

    neck = [
        # pyramid up
        ['Conv', {'channel_out': 512, 'kernel_size': 1, 'stride': 1}],                # 10
        ['Upsample', {'scale_factor': 2, 'mode': 'nearest'}],
        ['Concat', {'layer_idx': 6}],   # cat backbone P4
        ['CSP3', {'channel_out': 512, 'number': 3, 'shortcut': False}],
        ['ConvMixer', {'dim': 512, 'depth': 1, 'kernel_size': 7}],

        ['Conv', {'channel_out': 256, 'kernel_size': 1, 'stride': 1}],                # 15
        ['Upsample', {'scale_factor': 2, 'mode': 'nearest'}],
        ['Concat', {'layer_idx': 4}],   # cat backbone P3
        ['CSP3', {'channel_out': 256, 'number': 3, 'shortcut': False}],
        ['ConvMixer', {'dim': 256, 'depth': 1, 'kernel_size': 7}],

        ['Conv', {'channel_out': 128, 'kernel_size': 1, 'stride': 1}],                # 20
        ['Upsample', {'scale_factor': 2, 'mode': 'nearest'}],
        ['Concat', {'layer_idx': 2}],   # cat backbone P2
        ['CSP3Transformer', {'channel_out': 128, 'number': 1, 'shortcut': False}],    # 23
        ['ConvMixer', {'dim': 128, 'depth': 1, 'kernel_size': 7}],

        # pyramid down
        ['Conv', {'channel_out': 128, 'kernel_size': 3, 'stride': 2}],                # 25
        ['Concat', {'layer_idx': 20}],  # cat backbone P2
        ['CSP3Transformer', {'channel_out': 256, 'number': 1, 'shortcut': False}],    # 27
        ['ConvMixer', {'dim': 256, 'depth': 1, 'kernel_size': 7}],

        ['Conv', {'channel_out': 256, 'kernel_size': 3, 'stride': 2}],
        ['Concat', {'layer_idx': 15}],  # cat backbone P2
        ['CSP3Transformer', {'channel_out': 512, 'number': 2, 'shortcut': False}],    # 31
        ['ConvMixer', {'dim': 512, 'depth': 1, 'kernel_size': 7}],

        ['Conv', {'channel_out': 512, 'kernel_size': 3, 'stride': 2}],
        ['Concat', {'layer_idx': 10}],  # cat backbone P2
        ['CSP3Transformer', {'channel_out': 1024, 'number': 3, 'shortcut': False}],   # 35

    ]

    layers = []
    prev_layers = []
    next_channels_in = [3]
    for layer_idx, layer in enumerate(backbone + neck):
        layer_type, layer_kwargs = layer[0], layer[1]
        channel_in = next_channels_in[-1]
        prev_layer = -1

        if layer_type == 'Conv':
            layer = Conv(channel_in, **layer_kwargs)

        if layer_type == 'CSP3':
            layer = CSP3(channel_in, **layer_kwargs)

        if layer_type == 'CSP3Transformer':
            layer = CSP3Transformer(channel_in, **layer_kwargs)

        if layer_type == 'SPPF':
            layer = SPPF(channel_in, **layer_kwargs)

        if layer_type == 'Upsample':
            layer = nn.Upsample(size=None, **layer_kwargs)
            layer_kwargs['channel_out'] = channel_in

        if layer_type == 'Concat':
            layer = Concat()
            layer_kwargs['channel_out'] = channel_in * 2
            prev_layer = [-1, layer_kwargs['layer_idx']]

        if layer_type == 'ConvMixer':
            layer = ConvMixer(**layer_kwargs)
            layer_kwargs['channel_out'] = layer_kwargs['dim']

        layer.num_params = sum([x.numel() for x in layer.parameters()])
        layers.append(layer)
        prev_layers.append(prev_layer)
        next_channels_in.append(layer_kwargs['channel_out'])


class YOLOv5FromScratch(nn.Module):
    def __init__(self, layers, prev_layers):
        super().__init__()
        self.layers = layers
        self.prev_layers = prev_layers

    def forward(self, img):
        """Get features from YOLOv5 Backbone+Neck on the given image"""
        x, y = img, []
        for layer_idx, (layer, prev_layer) in enumerate(zip(self.layers, self.prev_layers)):
            if prev_layer != -1:  # if current layer is concat or SPP
                if isinstance(prev_layer, int):
                    x = y[prev_layer]
                else:
                    x = [x if j == -1 else y[j] for j in prev_layer]

            x = layer(x)
            y.append(x if layer_idx in [2, 4, 6, 10, 15, 20, 23, 27, 31, 35] else None)
            print(f'{layer_idx=}, {prev_layer=}, {x.shape=}')

        P2, P3, P4, P5 = [i for i in y if i is not None][-4:]
        return P2, P3, P4, P5


x = torch.rand(1, 3, 160, 160)
yolo_from_scratch = YOLOv5FromScratch(layers, prev_layers)
P2, P3, P4, P5 = yolo_from_scratch(x)
print(P2.shape, P3.shape, P4.shape, P5.shape)
