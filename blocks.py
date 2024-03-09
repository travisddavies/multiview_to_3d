import torch.nn as nn
from functools import partial

BN_MOMENTUM = 0.1


class Bottleneck(nn.Module):
    """
    The block basically is a three conv block combination, which contains
    a residual which if downsampling occurs will use a residual conv block
    to remember information from the previous layer, otherwise uses the
    original output from the previous layer.

    Bottlenecking occurs in the middle conv block, where the stride and the
    kernel size are larger
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    """
    A general conv block which can downsample the image if the stride is larger
    than one. A residual block is also present to add the data from the previous layer
    to the output
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        conv = partial(nn.Conv2d(inplanes, planes, kernel_size=3,
                                 stride=stride, padding=1, bias=False))
        bn = partial(nn.BatchNorm2d(inplace=True))
        self.conv1 = conv
        self.bn1 = bn
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv
        self.bn2 = bn
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
