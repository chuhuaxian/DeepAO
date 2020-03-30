# sub-parts of the U-Net model
import torch
import torch.nn as nn

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, group=1):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=group),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, group=1):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, group=group)

    def forward(self, x):
        x = self.conv(x)

        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, group=1):
        super(down, self).__init__()
        # if mode == 'maxpooling':
        self.mpconv = nn.Sequential(
            # nn.MaxPool2d(2),
            nn.AvgPool2d(2),
            double_conv(in_ch, out_ch, group=group))
        #     )
        # else:
        # self.mpconv = double_conv(in_ch, out_ch, group=group)

    def forward(self, x):
        x = self.mpconv(x)
        return x


class down_stride(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_stride, self).__init__()
        self.mpconv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, group=1, size=None):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            if size == None:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            else:
                self.up = nn.Upsample(size=size, mode='bilinear', align_corners=False)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch, group=group)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x

class DSUNet(nn.Module):
    def __init__(self, n_channels, n_classes, mode='maxpooling'):
        super(DSUNet, self).__init__()

        # base 8
        self.inc = inconv(n_channels, 8)
        self.down1 = down(8, 16, group=2)
        self.down2 = down(16, 32, group=4)
        self.down3 = down(32, 64, group=8)
        self.down4 = down(64, 128, group=16)
        self.down5 = down(128, 256, group=32)

        self.up1 = up(256+128, 128, group=16, size=None)
        self.up2 = up(128+64, 64, group=8)
        self.up3 = up(64+32, 32, group=4)
        self.up4 = up(32+16, 16, group=2)
        self.up5 = up(16+8, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        dx1 = self.up1(x6, x5)
        dx2 = self.up2(dx1, x4)
        dx3 = self.up3(dx2, x3)
        dx4 = self.up4(dx3, x2)
        dx5 = self.up5(dx4, x1)
        return dx5
