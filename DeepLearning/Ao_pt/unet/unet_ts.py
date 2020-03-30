# full assembly of the sub-parts to form the complete net
from .unet_parts_ts import *


# class UNet_teacher(nn.Module):
#     def __init__(self, n_channels, n_classes, mode='maxpooling'):
#         super(UNet_teacher, self).__init__()
#         # self.inc = inconv(n_channels, 32, mode='maxpooling')
#         # self.down1 = down(32, 64, mode='maxpooling')
#         # self.down2 = down(64, 128, mode='maxpooling')
#         # self.down3 = down(128, 256, mode='maxpooling')
#         # self.down4 = down(256, 256, mode='maxpooling')
#         # self.up1 = up(512, 128)
#         # self.up2 = up(256, 64)
#         # self.up3 = up(128, 32)
#         # self.up4 = up(64, 16)
#         # self.outc = outconv(16, n_classes)
#
#         self.inc = inconv(n_channels, 64)
#         self.down1 = down(64, 128)
#         self.down2 = down(128, 256)
#         self.down3 = down(256, 512)
#         self.down4 = down(512, 512)
#         self.up1 = up(1024, 256)
#         self.up2 = up(512, 128)
#         self.up3 = up(256, 64)
#         self.up4 = up(128, 64)
#         self.outc = outconv(64, n_classes)
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#
#         dx1 = self.up1(x5, x4)
#         dx2 = self.up2(dx1, x3)
#         dx3 = self.up3(dx2, x2)
#         dx4 = self.up4(dx3, x1)
#         dx5 = self.outc(dx4)
#         return [x1, x2, x3, x4, x5, dx1, dx2, dx3, dx4, dx5]


class UNet_student(nn.Module):
    def __init__(self, n_channels, n_classes, mode='maxpooling'):
        super(UNet_student, self).__init__()
        # self.inc = inconv(n_channels, 32, mode='maxpooling')
        # self.down1 = down(32, 64, mode='maxpooling')
        # self.down2 = down(64, 128, mode='maxpooling')
        # self.down3 = down(128, 256, mode='maxpooling')
        # self.down4 = down(256, 256, mode='maxpooling')
        # self.up1 = up(512, 128)
        # self.up2 = up(256, 64)
        # self.up3 = up(128, 32)
        # self.up4 = up(64, 16)
        # self.outc = outconv(16, n_classes)

        self.inc = inconv(n_channels, 8, mode=mode)
        self.down1 = down(8, 16, mode=mode)
        self.down2 = down(16, 32, mode=mode)
        self.down3 = down(32, 64, mode=mode)
        self.down4 = down(64, 64, mode=mode)
        self.up1 = up(128, 32)
        self.up2 = up(64, 16)
        self.up3 = up(32, 8)
        self.up4 = up(16, 4)
        self.outc = outconv(4, n_classes)

        # self.inc = inconv(6, 4, mode=mode)
        # self.down1 = down(4, 8, mode=mode)
        # self.down2 = down(8, 16, mode=mode)
        # self.down3 = down(16, 32, mode=mode)
        # self.down4 = down(32, 32, mode=mode)
        # self.up1 = up(64, 16)
        # self.up2 = up(32, 8)
        # self.up3 = up(16, 4)
        # self.up4 = up(8, 2)
        # self.outc = outconv(2, n_classes)

        # self.inc = inconv(n_channels, 16, mode=mode)
        # self.down1 = down(16, 32, mode=mode)
        # self.down2 = down(32, 64, mode=mode)
        # self.down3 = down(64, 128, mode=mode)
        # self.down4 = down(128, 128, mode=mode)
        # self.up1 = up(256, 64)
        # self.up2 = up(128, 32)
        # self.up3 = up(64, 16)
        # self.up4 = up(32, 8)
        # self.outc = outconv(8, n_classes)

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
        return [x1, x2, x3, x4, x5, dx1, dx2, dx3, dx4, dx5]

