import torch
import torch.nn as nn
import numpy as np
import torch.functional as F
from functools import partial

from .blocks import Bottleneck

BN_MOMENTUM = 0.1
blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

"""
    Something to research: what is this block.expansion value?
"""


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class PoseHighResolutionNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        # the number of outchannels more or less
        self.inplanes = 64
        super(PoseHighResolutionNet, self).__init__()
        # our standard conv block and batch normalisation block
        conv = partial(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                                 bias=False))
        bn = partial(nn.BatchNorm2d(64, momentum=BN_MOMENTUM))

        # Stage 1 of the model
        self.conv1 = conv
        self.bn1 = bn
        self.conv2 = conv
        self.layer1 = self._make_first_layer(Bottleneck, 64, 4)

        # Stage 2 of the model
        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg,
                                                           num_channels)

        # Stage 3 of the model
        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]

    def _make_first_layer(self, block, planes, blocks, stride=1):
        """
        Gives us n_blocks of layers of Bottlenecks, with the first Bottlenecks
        containing containing an extra conv block if this layer looks like it
        is downsampling
        """

        # If we are downsampling for this layer, we will add this extra conv
        # block
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM)
            )
        else:
            downsample = None

        unit_block = block(self.inplanes, planes, stride, downsample)
        layers = [unit_block]
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transition_layer(
        self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    block = nn.Sequential(
                        nn.Conv2d(
                            num_channels_pre_layer[i],
                            num_channels_cur_layer[i],
                            3, 1, 1, bias=False
                        ),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)
                    )
                    transition_layers.append(block)
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_branches_pre[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_channels_pre_layer else inchannels
                    conv = nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False
                        ),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)
                    )
                    conv3x3s.append(conv)
                    transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)
