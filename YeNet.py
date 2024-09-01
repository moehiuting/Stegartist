import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

SRM_npy = np.load('SRM_Kernels.npy')


# 8 filters in class “1st”
# 4 filters in class “2nd”
# 8 filters in class “3rd”
# 1 filters in class “SQUARE 3 × 3”
# 4 filters in class “EDGE 3 × 3”
# 1 filters in class “SQUARE 5 × 5”
# 4 filters in class “EDGE 5 × 5”
# print(type(SRM_npy), SRM_npy.shape, SRM_npy)


class SRM_conv2d(nn.Module):
    """conduct 30 SRM on 256*256*1 gray-level image to produce 252*252*30 residual"""

    def __init__(self, stride=1, padding=0):
        super(SRM_conv2d, self).__init__()
        self.in_channels = 1
        self.out_channels = 30
        self.kernel_size = (5, 5)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        self.dilation = (1, 1)
        self.transpose = False
        self.output_padding = (0,)
        self.groups = 1
        self.weight = Parameter(torch.Tensor(30, 1, 5, 5),
                                requires_grad=True)
        self.bias = Parameter(torch.Tensor(30),
                              requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias,
                        self.stride, self.padding, self.dilation,
                        self.groups)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, with_bn=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride)
        self.relu = nn.ReLU()
        self.with_bn = with_bn
        if with_bn:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = lambda x: x
        self.reset_parameters()

    def forward(self, x):
        return self.norm(self.relu(self.conv(x)))

    def reset_parameters(self):
        nn.init.xavier_uniform(self.conv.weight)
        self.conv.bias.data.fill_(0.2)
        if self.with_bn:
            self.norm.reset_parameters()


class YeNet(nn.Module):
    def __init__(self, with_bn=False, threshold=3):
        super(YeNet, self).__init__()
        self.with_bn = with_bn
        self.preprocessing = SRM_conv2d(1, 0)
        self.TLU = nn.Hardtanh(-threshold, threshold, True)
        if with_bn:
            self.norm1 = nn.BatchNorm2d(30)
        else:
            self.norm1 = lambda x: x
        # x:256*256*1
        # layer2,input:252*252*30,ouutput:250*250*30
        self.block2 = ConvBlock(30, 30, 3, with_bn=self.with_bn)
        # layer3,input:250*250*30,output:248*248*30
        self.block3 = ConvBlock(30, 30, 3, with_bn=self.with_bn)
        # layer4,input:248*248*30,output:246*246*30
        self.block4 = ConvBlock(30, 30, 3, with_bn=self.with_bn)
        # input:246*246*30,output:123*123*30
        self.pool1 = nn.AvgPool2d(2, 2)
        # layer5,input:123*123*30,output:118*118*32
        self.block5 = ConvBlock(30, 32, 5, with_bn=self.with_bn)
        # input:118*118*32,output:59*59*32
        self.pool2 = nn.AvgPool2d(3, 2)
        # layer6,input:59*59*32,output:55*55*32
        self.block6 = ConvBlock(32, 32, 5, with_bn=self.with_bn)
        # input:55*55*32,output:27*27*32
        self.pool3 = nn.AvgPool2d(3, 2)
        # layer7,input:27*27*32,output:23*23*32
        self.block7 = ConvBlock(32, 32, 5, with_bn=self.with_bn)
        # input:23*23*32,output:11*11*32
        self.pool4 = nn.AvgPool2d(3, 2)
        # layer8,input:11*11*32,output:9*9*16
        self.block8 = ConvBlock(32, 16, 3, with_bn=self.with_bn)
        # layer9,input:9*9*16,output:3*3*16
        self.block9 = ConvBlock(16, 16, 3, 3, with_bn=self.with_bn)
        self.ip1 = nn.Linear(3 * 3 * 16, 2)
        self.reset_parameters()

    def forward(self, x):
        x = x.float()
        # layer1,input:256*256*1,output:252*252*30
        x = self.preprocessing(x)
        # print("after processsing", x)
        x = self.TLU(x)
        x = self.norm1(x)
        # layer2,input:252*252*30,ouutput:250*250*30
        x = self.block2(x)
        # layer3,input:250*250*30,output:248*248*30
        x = self.block3(x)
        # layer4,input:248*248*30,output:246*246*30
        x = self.block4(x)
        # input:246*246*30,output:123*123*30
        x = self.pool1(x)
        # layer5,input:123*123*30,output:118*118*32
        x = self.block5(x)
        # input:118*118*32,output:59*59*32
        x = self.pool2(x)
        # layer6,input:59*59*32,output:55*55*32
        x = self.block6(x)
        # input:55*55*32,output:27*27*32
        x = self.pool3(x)
        # layer7,input:27*27*32,output:23*23*32
        x = self.block7(x)
        # input:27*27*32,output:11*11*32
        x = self.pool4(x)
        # layer8,input:11*11*32,output:9*9*16
        x = self.block8(x)
        # print("block8:", x)
        # layer9,input:9*9*16,output:3*3*16
        x = self.block9(x)
        x = x.view(x.size(0), -1)
        x = self.ip1(x)
        # print("outputs:", x)
        return x

    def reset_parameters(self):
        for mod in self.modules():
            if isinstance(mod, SRM_conv2d) or isinstance(mod, nn.BatchNorm2d) or isinstance(mod, ConvBlock):
                mod.reset_parameters()
            elif isinstance(mod, nn.Linear):
                nn.init.normal(mod.weight, 0., 0.01)
                mod.bias.data.zero_()


def accuracy(outputs, labels):
    _, argmax = torch.max(outputs, 1)
    return (labels == argmax.squeeze()).float().mean()
