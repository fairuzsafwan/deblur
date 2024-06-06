import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# FPN MODEL
class Down_fpn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down_fpn, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up_fpn(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up_fpn, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class FPN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(FPN, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down_fpn(64, 128)
        self.down2 = Down_fpn(128, 256)
        self.down3 = Down_fpn(256, 512)
        self.down4 = Down_fpn(512, 512)

        self.lateral4 = nn.Conv2d(512, 256, kernel_size=1)
        self.lateral3 = nn.Conv2d(512, 256, kernel_size=1)
        self.lateral2 = nn.Conv2d(256, 256, kernel_size=1)
        self.lateral1 = nn.Conv2d(128, 256, kernel_size=1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_fuse3 = DoubleConv(256, 256)
        self.conv_fuse2 = DoubleConv(256, 256)
        self.conv_fuse1 = DoubleConv(256, 256)

        self.outc = nn.Conv2d(256, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        p5 = self.lateral4(x5)
        p4 = self.up1(p5) + self.lateral3(x4)
        p4 = self.conv_fuse3(p4)
        p3 = self.up2(p4) + self.lateral2(x3)
        p3 = self.conv_fuse2(p3)
        p2 = self.up3(p3) + self.lateral1(x2)
        p2 = self.conv_fuse1(p2)

        out = self.up4(p2)
        return self.outc(out)


# UNET MODEL
class Down_unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down_unet, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up_unet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up_unet, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down_unet(64, 128)
        self.down2 = Down_unet(128, 256)
        self.down3 = Down_unet(256, 512)
        self.down4 = Down_unet(512, 512)
        self.up1 = Up_unet(1024, 256)
        self.up2 = Up_unet(512, 128)
        self.up3 = Up_unet(256, 64)
        self.up4 = Up_unet(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)