# sub-parts of the U-Net model
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x*(torch.tanh(F.softplus(x)))


def mish(x):
    return x * (torch.tanh(F.softplus(x)))


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, mode='maxpooling'):
        super(double_conv, self).__init__()

        # stride_ = 1 if mode == 'maxpooling' else 2
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_ch, in_ch, 1),
        #     Mish(),
        #     nn.Conv2d(in_ch, out_ch, 3, padding=1),
        #     Mish(),
        #     nn.Conv2d(out_ch, out_ch, 1),
        #     # Mish(),
        #
        #
        #     # nn.Conv2d(out_ch, out_ch, 3, stride=stride_, padding=1),
        #     # nn.BatchNorm2d(out_ch),
        #     # nn.LeakyReLU(inplace=True),
        # )

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            # nn.InstanceNorm2d(out_ch),
            # nn.BatchNorm2d(out_ch),
            # nn.GroupNorm(4, out_ch),
            nn.LeakyReLU(inplace=True),
            # Mish(),
            # nn.BatchNorm2d(out_ch),

            # nn.Conv2d(out_ch, out_ch, 3, stride=stride_, padding=1),
            # nn.BatchNorm2d(out_ch),
            # nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, mode='maxpooling'):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, mode=mode)

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
    def __init__(self, in_ch, out_ch, mode='maxpooling'):
        super(down, self).__init__()
        if mode == 'maxpooling':
            self.mpconv = nn.Sequential(
                # nn.MaxPool2d(2),
                nn.AvgPool2d(2),
                double_conv(in_ch, out_ch)
            )
        else:
            self.mpconv = double_conv(in_ch, out_ch, mode=mode)

    def forward(self, x):
        x = self.mpconv(x)
        return x


class down_multi_scale(nn.Module):
    def __init__(self, in_ch, out_ch, down_scale):
        super(down_multi_scale, self).__init__()

        self.conv = double_conv(4, in_ch)
        self.pool = nn.AvgPool2d(down_scale)
        # self.pool = nn.Upsample(scale_factor=1/down_scale)

        self.avg_pool = nn.AvgPool2d(2)
        self.dconv = double_conv(in_ch*2, out_ch)

    def forward(self, x, input):
        input = self.pool(self.conv(input))
        x = self.avg_pool(x)
        x = torch.cat([input, x], dim=1)
        x = self.dconv(x)
        return x


class down_stride(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_stride, self).__init__()
        self.mpconv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x



