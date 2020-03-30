import numpy as np
import matplotlib.pyplot as plt
import pyexr
import cv2
import datetime
# grayImg = pyexr.open('D:\\Projects\\Datasets\\3dsmax\\evaluate\\Test\\scene06\\Position\\pokoj_Camera001_VRaySamplerInfo.exr').get()[:, :, 2]
# grayImg = cv2.imread('E:\\005586-rawDepth.png', flags=-1)
grayImg = cv2.imread('C:\\Users\\39796\Desktop\\background_1_1_depth.png', flags=-1)

# plt.imshow(grayImg)
# plt.show()
# print(np.max(grayImg))
import torch
import torch.nn as nn
from torch.autograd import Variable
# d_im /=1000.
# d_im = (1500 + d_im) / 1500.0
# zy, zx = np.gradient(grayImg)
# You may also consider using Sobel to get a joint Gaussian smoothing and differentation
# to reduce noise
win_size = 5
sobel_operator = {3:np.array([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                  5:-1*np.array([[1,2,0,-2,-1],[4,8,0,-8,-4],[6,12,0,-12,-6],[4,8,0,-8,-4],[1,2,0,-2,-1]], dtype=np.float32)}
class Gradient(nn.Module):
    def __init__(self):
        super(Gradient, self).__init__()
        self.conv_x = nn.Conv2d(1, 1, kernel_size=win_size, stride=1, padding=win_size//2)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=win_size, stride=1, padding=win_size//2)
        conv_x = sobel_operator[win_size]
        conv_y = conv_x.transpose()
        # conv_y = np.array([[-1.0, -2, -1.0], [0, 0, 0], [1, 2, 1]])
        self.conv_x.weight = nn.Parameter(torch.from_numpy(conv_x).float().unsqueeze(0).unsqueeze(0))
        self.conv_y.weight = nn.Parameter(torch.from_numpy(conv_y).float().unsqueeze(0).unsqueeze(0))

    def getGrd(self, input):
        grd_x = self.conv_x(input)
        grd_y = self.conv_y(input)
        return -grd_x, -grd_y

model = Gradient().cuda()
one_arr = torch.ones((grayImg.shape[0], grayImg.shape[1], 1), dtype=torch.float32).cuda()
print()

def depth2normal(depth):
    depth = np.expand_dims(np.expand_dims(depth, axis=0), axis=0)
    depth = torch.from_numpy(depth).float().cuda()
    gx, gy = model.getGrd(depth)
    gx = torch.squeeze(gx)
    gy = torch.squeeze(gy)

    print(gx.shape, gy.shape, one_arr.shape)
    normal = torch.cat([torch.unsqueeze(gx, dim=-1), torch.unsqueeze(gy, dim=-1), one_arr], dim=-1)
    norm = torch.unsqueeze(torch.sqrt(torch.sum(torch.pow(normal, 2.0), dim=-1)), dim=-1)
    normal /= norm
    return normal


# grayImg = (1500+ grayImg)/1500.
Img = grayImg.astype(np.float32)
res = depth2normal(Img)
res += 1
res /= 2
res *= 255
res = res.cpu().detach().numpy()
plt.subplot(131)
plt.imshow(res.astype(np.uint8))
# plt.show()


# img = np.expand_dims(np.expand_dims(grayImg, axis=0), axis=0)
# input = torch.from_numpy(img)
# input = Variable(input.float()).cuda()
#
# gx, gy = model.getGrd(input)
# gx = np.squeeze(gx.cpu().detach().numpy())
# gy = np.squeeze(gy.cpu().detach().numpy())
# zy, zx = np.gradient(grayImg)
zx = cv2.Sobel(grayImg, cv2.CV_64F, 1, 0, ksize=15)
zy = cv2.Sobel(grayImg, cv2.CV_64F, 0, 1, ksize=15)

normal = np.dstack((-zx, -zy, np.ones_like(grayImg)))
n = np.linalg.norm(normal, axis=2)

# n = np.sqrt(np.sum(np.square(normal), axis=-1))
normal[:, :, 0] /= n
normal[:, :, 1] /= n
normal[:, :, 2] /= n

# offset and rescale values to be in 0-255
normal += 1
normal /= 2
normal *= 255
bias = np.abs(normal-res)
bias = bias.astype(np.uint8)
nm = normal.astype('uint8')
#
plt.subplot(132)
plt.imshow(nm)



plt.subplot(133)
plt.imshow(bias)

plt.show()

# h, w = np.shape(grayImg)
# normals = np.zeros((h, w, 3))
#
#
# def normalizeVector(v):
#     length = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
#     v = v/length
#     return v
#
# def depthValue(x):
#     return x
#
#
# for x in range(1, h-1):
#     for y in range(1, w-1):
#
#         dzdx = (float(depthValue(grayImg[x+1, y])) - float(depthValue(grayImg[x-1, y]))) / 2.0
#         dzdy = (float(depthValue(grayImg[x, y+1])) - float(depthValue(grayImg[x, y-1]))) / 2.0
#
#         d = (-dzdx, -dzdy, 1.0)
#
#         n = normalizeVector(d)
#
#         normals[x, y] = n * 0.5 + 0.5
#
# normals *= 255
#
# normals = normals.astype('uint8')
#
# plt.imshow(normals)
# plt.show()

# normals = cv2.cvtColor(normals, cv2.COLOR_BGR2RGB)


# plt.imshow(normals)
# plt.show()
# cv2.imwrite("normal.png", normal[:, :, ::-1])

# plt.subplot(221), plt.imshow(zx)
# plt.subplot(222), plt.imshow(zy)
#
# plt.subplot(223), plt.imshow(gx)
# plt.subplot(224), plt.imshow(gy)
# # plt.show()
#
# plt.show()
#
#
# print()