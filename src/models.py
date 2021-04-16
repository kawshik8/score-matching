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

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + in_channels // 2, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, bilinear=False, depth = 3, init_channels = 64, factor = 2):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.depth = depth

        self.down_block = nn.ModuleList([])
        self.up_block = nn.ModuleList([])

        for d in range(depth+1):
            if d==0:
                self.down_block.append(DoubleConv(n_channels, init_channels))
                self.up_block.append(OutConv(init_channels, n_channels))
            else:
                self.down_block.append(Down(init_channels, init_channels * 2))
                self.up_block = nn.ModuleList([Up(init_channels * 2, init_channels, bilinear)]).extend(self.up_block)
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
    def __init__(self, n_channels=3, bilinear=False, depth = 3, init_channels = 64, factor = 2):
        super(Encoder, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.depth = depth

        self.down_block = nn.ModuleList([])

        for d in range(depth+1):
            if d==0:
                self.down_block.append(DoubleConv(n_channels, init_channels))
            else:
                self.down_block.append(Down(init_channels, init_channels * 2))
                init_channels = init_channels*2

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.dropout = torch.nn.Dropout(p=0.5, inplace=True)
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
    model = UNet(3, 3)
    print(torch.rand(64,3,32,32).shape, model(torch.rand(64,3,32,32)).shape)

    model = Encoder(3)
    print(torch.rand(64,3,32,32).shape, model(torch.rand(64,3,32,32)).shape)
    # print(model)
    