import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Squeeze_excitation_layer(nn.Module):
    def __init__(self, filters, se_ratio=4):
        super(Squeeze_excitation_layer, self).__init__()
        reduction = filters // se_ratio
        self.se = nn.Sequential(nn.Conv2d(filters, reduction, kernel_size=1, bias=True),
                                nn.SiLU(),
                                nn.Conv2d(reduction, filters, kernel_size=1, bias=True),
                                nn.Sigmoid())
    def forward(self, inputs):
        x = torch.mean(inputs, [2, 3], keepdim=True)
        x = self.se(x)
        return torch.multiply(inputs, x)

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class MBConv2d_block(nn.Module):
    def __init__(self, in_channels, out_channels, k=1):
        super(MBConv2d_block, self).__init__()

        self.net = nn.Sequential(nn.Conv2d(in_channels, out_channels * k, kernel_size=(1, 1), stride=(1, 1),  bias=False),
               nn.BatchNorm2d(out_channels * k),
               nn.SiLU(),
               depthwise_separable_conv(out_channels * k, out_channels * k, kernel_size = 3, bias=False),
               nn.BatchNorm2d(out_channels * k),
               nn.SiLU(),
               Squeeze_excitation_layer(filters=out_channels * k, se_ratio=4),
               nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
               nn.BatchNorm2d(out_channels * k),
               nn.Dropout(p=0.2))

    def forward(self, inputs):
        x = self.net(inputs)
        return torch.add(inputs, x)
  
