import torch
import torch.nn as nn
import numpy as np
import torch.functional as F
from functools import partial
import logging
import os

from .blocks import Bottleneck, BasicBlock

BN_MOMENTUM = 0.1
blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}
logger = logging.getLogger(__name__)

"""
    Something to research: what is this block.expansion value?
    Also, what is multi_scale_output?
"""


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class PoseHighResolutionNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        # the number of outchannels more or less
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
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
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels
        )

        # Stage 4 of the model
        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels
        )
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True
        )

        #### modification to output keypoints and mask prediction ####
        last_inp_channels = np.int(np.sum(pre_stage_channels))

        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=cfg.MODEL.NUM_JOINTS,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
            )
        )

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # This seems to get three different resolution images and then does
        # some interpolation to make them equal the same size as the largest
        # resolution map, which is then concatenated together and fed into the
        # final layer to get the mask
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear',
                           align_corners=False)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear',
                           align_corners=False)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear',
                           align_corners=False)

        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.final_layer(x)

        return x

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
                # Given that the number of channels in and the number of
                # channels out are not equal, we will make a conv block as we
                # have for previous layers
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    block = nn.Sequential(
                        nn.Conv2d(
                            num_channels_pre_layer[i],
                            num_channels_cur_layer[i],
                            kernel_size=3, stride=1, padding=1, bias=False
                        ),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)
                    )
                    transition_layers.append(block)
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                # If i is greater than num_branches_pre, then we will
                # then iterate from 0 to the equivalent of the difference
                # between i and num_branches_pre
                for j in range(i+1-num_branches_pre):
                    # Note: the above for loop looks like a bug because it
                    # seems to not iterate throught the channels that surpass
                    # pre_stage_channels

                    # These are going to connect to the last block from the
                    # previous layer, so we will take the number of channels
                    # of that block as the inchannels
                    inchannels = num_branches_pre[-1]
                    # The number of outchannels will equal the same as
                    # inchannels for every convblock except for the last one,
                    # which will equal the current outchannel that j is sitting
                    # on
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_channels_pre_layer else inchannels
                    # We will be downsizing the image by increasing the stride
                    # to 2, so it will progressively downsample the image
                    conv = nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels,
                            kernel_size=3, stride=2, padding=1, bias=False
                        ),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)
                    )
                    conv3x3s.append(conv)
                # Append the conv blocks to the end of the list
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = layer_config['BLOCK']
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )

            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, m.Conv2d):
                nn.init.noral_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=format)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} does not exist!'.format(pretrained))


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        # set up the amount of branches specified
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        # If downsizing is present, then add this particular conv block to act
        # as an residual conv block at the start
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )
        else:
            downsample = None

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )

        # add the next conv blocks according to the number of blocks for this
        # branch
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []

        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    # make a bunch of upsampling conv blocks that connect to
                    # various branches from the previous layer
                    block = nn.Sequential(
                        nn.Conv2d(
                            num_inchannels[j],
                            num_inchannels[i],
                            kernel_size=1, stride=1, padding=0, bias=False
                        ),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                    )
                    fuse_layer.append(block)
                elif j == i:
                    fuse_layer.append(None)
                else:
                    # make a bunch of downsampling conv blocks for the other
                    # branches from the previous layer
                    conv3x3s = []
                    for k in range(i-j):
                        if (k == i - j - 1):
                            # If it's the final block of this loop, don't
                            # include relu module
                            num_outchannels_conv3x3 = num_inchannels[i]
                            block = nn.Sequential(
                                nn.Conv2d(
                                    num_inchannels[j],
                                    num_outchannels_conv3x3,
                                    kernel_size=3, stride=2, padding=1,
                                    bias=False
                                ),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                            )
                            conv3x3s.append(block)
                        else:
                            # If it is not the last block of this loop, include
                            # the relu module
                            num_outchannels_conv3x3 = num_inchannels[j]
                            block = nn.Sequential(
                                nn.Conv2d(
                                    num_inchannels[j],
                                    num_outchannels_conv3x3,
                                    kernel_size=3, stride=2, padding=1,
                                    bias=False
                                ),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(True)
                            )
                            conv3x3s.append(block)
                # basically put all these upsampling and downsampling conv
                # blocks into an indexable 2D module list
                fuse_layer.append(block)
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels
