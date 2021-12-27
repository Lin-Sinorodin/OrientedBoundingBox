import numpy as np
import torch
import torch.nn as nn

from obb.model import common
from typing import List

BACKBONE = [
    ['Conv', {'channel_out': 64, 'kernel_size': 6, 'stride': 2, 'padding': 2}],   # 0
    ['Conv', {'channel_out': 128, 'kernel_size': 3, 'stride': 2}],                # 1
    ['CSP3', {'channel_out': 128, 'number': 3, 'shortcut': True}],                # 2
    ['Conv', {'channel_out': 256, 'kernel_size': 3, 'stride': 2}],                # 3
    ['CSP3', {'channel_out': 256, 'number': 6, 'shortcut': True}],                # 4
    ['Conv', {'channel_out': 512, 'kernel_size': 3, 'stride': 2}],                # 5
    ['CSP3', {'channel_out': 512, 'number': 9, 'shortcut': True}],                # 6
    ['Conv', {'channel_out': 1024, 'kernel_size': 3, 'stride': 2}],               # 7
    ['CSP3Transformer', {'channel_out': 1024, 'number': 3, 'shortcut': True}],    # 8
    ['SPPF', {'channel_out': 1024, 'k': 5}],
]

NECK = [
    # pyramid up
    ['Conv', {'channel_out': 512, 'kernel_size': 1, 'stride': 1}],                # 10
    ['nn.Upsample', {'scale_factor': 2, 'mode': 'nearest'}],
    ['Concat', {'layer_idx': 6}],   # cat backbone P4
    ['CSP3', {'channel_out': 512, 'number': 3, 'shortcut': False}],
    ['ConvMixer', {'dim': 512, 'depth': 1, 'kernel_size': 7}],

    ['Conv', {'channel_out': 256, 'kernel_size': 1, 'stride': 1}],                # 15
    ['nn.Upsample', {'scale_factor': 2, 'mode': 'nearest'}],
    ['Concat', {'layer_idx': 4}],   # cat backbone P3
    ['CSP3', {'channel_out': 256, 'number': 3, 'shortcut': False}],
    ['ConvMixer', {'dim': 256, 'depth': 1, 'kernel_size': 7}],

    ['Conv', {'channel_out': 128, 'kernel_size': 1, 'stride': 1}],                # 20
    ['nn.Upsample', {'scale_factor': 2, 'mode': 'nearest'}],
    ['Concat', {'layer_idx': 2}],   # cat backbone P2
    ['CSP3Transformer', {'channel_out': 256, 'number': 1, 'shortcut': False}],    # 23
    ['ConvMixer', {'dim': 256, 'depth': 1, 'kernel_size': 7}],

    # pyramid down
    ['Conv', {'channel_out': 128, 'kernel_size': 3, 'stride': 2}],                # 25
    ['Concat', {'layer_idx': 20}],  # cat backbone P2
    ['CSP3Transformer', {'channel_out': 256, 'number': 1, 'shortcut': False}],    # 27
    ['ConvMixer', {'dim': 256, 'depth': 1, 'kernel_size': 7}],

    ['Conv', {'channel_out': 256, 'kernel_size': 3, 'stride': 2}],
    ['Concat', {'layer_idx': 15}],  # cat backbone P2
    ['CSP3Transformer', {'channel_out': 256, 'number': 2, 'shortcut': False}],    # 31
    ['ConvMixer', {'dim': 256, 'depth': 1, 'kernel_size': 7}],

    ['Conv', {'channel_out': 512, 'kernel_size': 3, 'stride': 2}],
    ['Concat', {'layer_idx': 10}],  # cat backbone P2
    ['CSP3Transformer', {'channel_out': 256, 'number': 3, 'shortcut': False}],   # 35
]

REMEMBER_LAYERS = [2, 4, 6, 10, 15, 20, 23, 27, 31, 35]
NUM_FEATURE_MAPS = 4


class FeatureMap(nn.Module):
    def __init__(
            self,
            backbone: List[List[str, dict]],
            neck: List[List[str, dict]],
            remember_layers: List[int],
            num_feature_maps: int
    ):
        super().__init__()

        self.backbone = backbone
        self.neck = neck
        self.remember_layers = remember_layers
        self.num_feature_maps = num_feature_maps
        self.model = self._get_feature_map()

    def _get_feature_map(self) -> nn.Sequential:
        layers = []
        next_channels_in = 3
        for layer_idx, (layer_str, layer_kwargs) in enumerate(self.backbone + self.neck):
            assert isinstance(layer_str, str)
            layer_obj = eval(f'common.{layer_str}')

            channels_in = next_channels_in
            prev_layer = -1

            if layer_str in ['Conv', 'CSP3', 'CSP3Transformer', 'SPPF']:
                layer = layer_obj(channels_in, **layer_kwargs)
            elif layer_obj is nn.Upsample:
                layer = layer_obj(size=None, **layer_kwargs)
                layer_kwargs['channel_out'] = channels_in
            elif layer_obj is common.Concat:
                layer = layer_obj()
                layer_kwargs['channel_out'] = channels_in * 2
                prev_layer = [-1, layer_kwargs['layer_idx']]
            elif layer_obj is common.ConvMixer:
                layer = layer_obj(**layer_kwargs)
                layer_kwargs['channel_out'] = layer_kwargs['dim']
            else:
                raise ValueError('Unknown layer type')

            layer.idx = layer_idx
            layer.prev_layer_idxs = prev_layer
            layer.channels_in = channels_in
            layer.channels_out = layer_kwargs['channel_out']
            layer.num_params = sum([sub_layer.numel() for sub_layer in layer.parameters()])

            layers.append(layer)
            next_channels_in = layer_kwargs['channel_out']

        return nn.Sequential(*layers)

    def forward(self, x) -> List[torch.tensor]:
        curr_feature = x
        prev_features = []
        for layer in self.model:
            prev_layer_idxs = layer.prev_layer_idxs
            if prev_layer_idxs != -1:
                if isinstance(prev_layer_idxs, int):
                    curr_feature = prev_features[prev_layer_idxs]
                else:
                    curr_feature = [curr_feature if idx == -1 else prev_features[idx] for idx in prev_layer_idxs]

            curr_feature = layer(curr_feature)
            prev_features.append(curr_feature if layer.idx in self.remember_layers else None)

        # add stride information for feature maps
        feature_maps = []
        for feature_map in [feature for feature in prev_features if feature is not None][-self.num_feature_maps:]:
            feature_map.stride = x.shape[-1]//feature_map.shape[-1]
            feature_maps.append(feature_map)

        return feature_maps


if __name__ == '__main__':
    img_in = torch.rand(2, 3, 256, 256)
    print('\n\t'.join(['Input:', f'img_in.shape = {img_in.shape}']))

    feature_map = FeatureMap(BACKBONE, NECK, REMEMBER_LAYERS, NUM_FEATURE_MAPS)
    P2, P3, P4, P5 = feature_map(img_in)
    print('\n\t'.join([
        'Output (Feature Maps):',
        f'P2.shape = {P2.shape}  |  P2.stride = {P2.stride}',
        f'P3.shape = {P3.shape}  |  P3.stride = {P3.stride}',
        f'P4.shape = {P4.shape}  |  P4.stride = {P4.stride}',
        f'P5.shape = {P5.shape}  |  P5.stride = {P5.stride}',
    ]))

    total_parameters = sum([np.prod(p.size()) for p in feature_map.model.parameters()])
    trainable_parameters = sum([np.prod(p.size()) for p in feature_map.model.parameters() if p.requires_grad])
    print('\n\t'.join(['Model parameters:',
                       f'total_parameters = {total_parameters:,}',
                       f'trainable_parameters = {trainable_parameters:,}']))

    """
    Input:
        img_in.shape = torch.Size([1, 3, 256, 256])
    Output (Feature Maps):
        P2.shape = torch.Size([1, 128, 64, 64])  |  P2.stride = 4
        P3.shape = torch.Size([1, 256, 32, 32])  |  P3.stride = 8
        P4.shape = torch.Size([1, 512, 16, 16])  |  P4.stride = 16
        P5.shape = torch.Size([1, 1024, 8, 8])  |  P5.stride = 32
    Model parameters:
        total_parameters = 129057152
        trainable_parameters = 129057152
    """
