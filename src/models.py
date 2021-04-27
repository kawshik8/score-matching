# This file implements models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import numpy as np
import math
import logging as log
import torchvision.models.resnet as resnet
from torchvision.models.inception import inception_v3

class Inception(nn.Module):
    def __init__(self, layer=-1):
        super(Inception, self).__init__()
        model = inception_v3(pretrained=True, transform_input=False, aux_logits=False)
        modules1 = list(model.children())[:-2]
        self.model = nn.Sequential(*modules1)
        modules2 = list(model.children())[-2:]
        self.rest = nn.Sequential(*modules2)


    def forward(self,x):
        features = self.model(x).view(x.size(0),-1)
        logits = self.rest(features)

        return features, logits


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, transpose = False, norm_type='batch'):
        super(Conv, self).__init__()
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels)


        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, norm_type='batch'):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            Conv(in_channels, mid_channels, kernel_size=3, padding=1, norm_type=norm_type),
            Conv(mid_channels, out_channels, kernel_size=3, padding=1, norm_type=norm_type)
        )

    def forward(self, x):
        return self.double_conv(x)

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_channels, out_channels, mid_channels=None, norm_type='batch'):
        super(PreActBlock, self).__init__()

        if not mid_channels:
            mid_channels = out_channels

        if norm_type == 'batch':
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(mid_channels)
        elif norm_type == 'instance':
            self.bn1 = nn.InstanceNorm2d(in_channels)
            self.bn2 = nn.InstanceNorm2d(mid_channels)
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if in_channels!=out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            )

    def forward(self, x):
        
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, maxpool=True, norm_type='batch', block=None):
        super().__init__()
        if maxpool:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = Conv(in_channels, out_channels, kernel_size = 2, stride = 2, norm_type=norm_type)

        if block == 'conv_block':
            self.conv = DoubleConv(in_channels if maxpool else out_channels, out_channels, norm_type=norm_type)
        else:
            self.conv = PreActBlock(in_channels if maxpool else out_channels, out_channels, norm_type=norm_type)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)

        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False, norm_type='batch',block=None):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if block == 'conv_block':
                self.conv = DoubleConv(in_channels + in_channels // 2, out_channels, in_channels // 2, norm_type=norm_type)
            else:
                self.conv = PreActBlock(in_channels + in_channels // 2, out_channels, in_channels // 2, norm_type=norm_type)
        else:
            self.up = Conv(in_channels , in_channels // 2, kernel_size=2, stride=2, transpose=True, norm_type=norm_type)
            if block == 'conv_block':
                self.conv = DoubleConv(in_channels, out_channels, norm_type=norm_type)
            else:
                self.conv = PreActBlock(in_channels, out_channels, norm_type=norm_type)


    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class In_OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(In_OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, depth = 3, bilinear=False, init_channels = 64, factor = 2, block='conv_block', norm_type='batch'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.depth = depth

        self.down_block = nn.ModuleList([])
        self.up_block = nn.ModuleList([])
        # print(block)

        for d in range(depth+1):
            if d==0:
                self.down_block.append(DoubleConv(n_channels, init_channels, norm_type=norm_type) if block=='conv_block' else nn.Sequential(In_OutConv(n_channels, init_channels), PreActBlock(init_channels, init_channels, norm_type=norm_type)))
                self.up_block.append(In_OutConv(init_channels, n_channels))
            else:
                self.down_block.append(Down(init_channels, init_channels * 2, maxpool=bilinear, norm_type=norm_type, block=block))
                self.up_block = nn.ModuleList([Up(init_channels * 2, init_channels, bilinear=bilinear, norm_type=norm_type, block=block)]).extend(self.up_block)
                init_channels = init_channels*2


    def forward(self, x):
        
        skip_connections = []
        # print("input: ", x.shape)
        for i in range(self.depth+1):
            x = self.down_block[i](x)
            skip_connections.append(x)
            # print("down " + str(i) + ": ", x.shape)

        for i in range(self.depth+1):
            # print("up-in " + str(i) + ": ", x.shape, skip_connections[self.depth-i-1].shape)
            if i==self.depth:
                x = self.up_block[i](x)
            else:
                
                x = self.up_block[i](x,skip_connections[self.depth-i-1])
                # print("up " + str(i) + ": ", x.shape)

        return x

class Encoder(nn.Module):
    def __init__(self, n_channels=3, bilinear=False, depth = 3, init_channels = 64, factor = 2, norm_type='batch'):
        super(Encoder, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.depth = depth

        self.down_block = nn.ModuleList([])

        for d in range(depth+1):
            if d==0:
                self.down_block.append(DoubleConv(n_channels, init_channels, norm_type=norm_type))
            else:
                self.down_block.append(Down(init_channels, init_channels * 2, maxpool=bilinear, norm_type=norm_type))
                init_channels = init_channels*2

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear = nn.Linear(init_channels, 128)
        self.out = nn.Linear(128, 1)


    def forward(self, x):
        
        # print("input: ", x.shape)
        for i in range(self.depth+1):
            x = self.down_block[i](x)
            # print("down " + str(i) + ": ", x.shape)

        x = self.avg_pool(x).view(x.size(0),-1)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.out(x)

        return x

        


if __name__ == "__main__":
    model = UNet(3, depth=3)
    print(model)
    print(torch.rand(64,3,32,32).shape, model(torch.rand(64,3,32,32)).shape)

    # model = Encoder(3)
    # print(model)
    # print(torch.rand(64,3,32,32).shape, model(torch.rand(64,3,32,32)).shape)

    # model = Inception()
    # print(model)
    # print(torch.rand(64,3,299,299).shape, model(torch.rand(64,3,299,299)).shape)
    # print(model)
    