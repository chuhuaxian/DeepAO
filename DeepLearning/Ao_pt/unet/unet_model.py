# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, mode='maxpooling'):
        super(UNet, self).__init__()
        # base 64
        # self.inc = inconv(n_channels, 64)
        # self.down1 = down(64, 128)
        # self.down2 = down(128, 256)
        # self.down3 = down(256, 512)
        # self.down4 = down(512, 512)
        # self.up1 = up(1024, 256)
        # self.up2 = up(512, 128)
        # self.up3 = up(256, 64)
        # self.up4 = up(128, 64)
        # self.outc = outconv(64, n_classes)

        # base 16
        # self.inc = inconv(n_channels, 16)
        # self.down1 = down(16, 32)
        # self.down2 = down(32, 64)
        # self.down3 = down(64, 128)
        # self.down4 = down(128, 128)
        # self.up1 = up(256, 64)
        # self.up2 = up(128, 32)
        # self.up3 = up(64, 16)
        # self.up4 = up(32, 8)
        # self.outc = outconv(8, n_classes)

        # base 8
        # self.inc = inconv(n_channels, 8)
        # self.down1 = down(8, 16)
        # self.down2 = down(16, 32)
        # self.down3 = down(32, 64)
        # self.down4 = down(64, 64)
        # # self.down5 = down(128, 256)
        #
        # # self.up0 = up(256+128, 64)
        # self.up1 = up(128, 16)
        # self.up2 = up(64+16, 8)
        # self.up3 = up(32+8, 4)
        # self.up4 = up(16+4, 2)
        # self.outc = outconv(2, n_classes)

        # base 4
        self.inc = inconv(n_channels, 4)

        self.down1 = down(4, 8)
        self.down2 = down(8, 16)
        self.down3 = down(16, 32)
        self.down4 = down(32, 32)

        self.up1 = up(64, 16)
        self.up2 = up(32, 8)
        self.up3 = up(16, 4)
        self.up4 = up(8, 4)
        self.outc = outconv(4, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        dx1 = self.up1(x5, x4)
        dx2 = self.up2(dx1, x3)
        dx3 = self.up3(dx2, x2)
        dx4 = self.up4(dx3, x1)
        dx5 = self.outc(dx4)
        return dx5


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return self.net(x).view(batch_size)

