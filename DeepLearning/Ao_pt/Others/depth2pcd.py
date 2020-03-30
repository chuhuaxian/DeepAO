import os
import pyexr
import matplotlib.pyplot as plt
import numpy as np
import math
# normal_path = 'C:\\Users\\39796\Desktop\\0000000061_normal.exr'
# position_path = 'C:\\Users\zhangdongjiu\\test_01\screenshot\\2-position.exr'
# position_path = 'D:\Projects\Datasets\scene_08\\scene_08_Perspective_position_View150111.exr'
# position_path = 'C:\\Users\zhangdongjiu\Desktop\\3dresource\\test_img\\Untitled_Perspective_Z Depth.exr'
# position_path1 = 'C:\\Users\zhangdongjiu\Desktop\\3dresource\\test_img\\Untitled_Perspective_position.exr'
position_path = 'D:\\Projects\\Datasets\\3dsmax\\evaluate\\Test\\scene000\\Position\\0000000002.exr'
# position_path1 = 'D:\\Projects\\Datasets\\3dsmax\\evaluate\\Test\\scene000\\Position\\scene_08_Camera019_Z Depth.exr'


positions = pyexr.open(position_path).get()


# plt.imshow(positions[:, :, 2])
# plt.show()
# positions1= pyexr.open(position_path1).get()
camera_cx = 256.0
camera_cy = 256.0
camera_fx = 548.9937
camera_fy = 548.9937

cam_proj_ours = np.array([
    [2.1, 0.000000, -0.000000,  0.000000],
    [0.000000, 2.1,  0.000000,  0.000000],
    [0.000000, 0.000000, -1.000200, -0.02],
    [0.000000, 0.000000, -1.000000,  0.000000]])

cam_inv_proj_unity = np.linalg.inv(cam_proj_ours)


clip_near = 0.1
clip_far = 1000.0
fov = 50
a = -(clip_far+clip_near)/(clip_far-clip_near)
b = -(2*clip_far*clip_near/(clip_far-clip_near))

right = clip_near*math.tan(fov/2*math.pi/180)
left = -right
aspect = 1280/720
aspect = 1
top = right/aspect
bottom = -top
# print(math.atan(1/0.715138)*2*180/math.pi, math.atan(1/1.271356)*2*180/math.pi)

# Project_Matrix = np.array([[2*clip_near/(right-left), 0.0,                      (right+left)/(right-left), 0.0],
#                           [0.0,                      2*clip_near/(top-bottom), (top+bottom)/(top-bottom), 0.0],
#                           [0.0,                      0.0,                       a,                        b],
#                           [0.0,                      0.0,                       -1.0,                     0.0]])

Project_Matrix = np.array([[2*clip_near/(right-left), 0.0,                      0.0, 0.0],
                          [0.0,                      2*clip_near/(top-bottom), 0.0, 0.0],
                          [0.0,                      0.0,                       a,                        b],
                          [0.0,                      0.0,                       -1.0,                     0.0]])

Project_Inv_Matrix = np.array([[(right-left)/(2*clip_near),  0.0,                        0.0,    0.0],
                               [0.0,                        (top-bottom)/(2*clip_near),  0.0,    0.0],
                               [0.0,                         0.0,                        0.0,    -1],
                               [0.0,                         0.0,                        1/b,    a/b]])


print(np.matmul(Project_Matrix, Project_Inv_Matrix))
cam_proj_unity = np.array(
[[2.14451,0.00000,0.00000,0.00000],
[0.00000,2.14451,0.00000,0.00000],
[0.00000,0.00000,-1.00001,-0.02000],
[0.00000,0.00000,-1.00000,0.00000]])
cam_inv_proj_unity = np.linalg.inv(cam_proj_unity)

cam_inv_proj_ours = np.linalg.inv(cam_proj_ours)


cam_inv_proj = np.array([
    [-1.398332, -0.000000,  0.000000, -0.000000],
    [-0.000000,  0.786562, -0.000000,  0.000000],
    [ 0.000000, -0.000000,  0.000000, -1.000000],
    [-0.000000,  0.000000, -4.999500,  5.000499]])
# fov: 108.86000058017584
cam_proj = np.array([
    [-0.715138, 0.000000, -0.000000,  0.000000],
    [ 0.000000, 1.271356,  0.000000,  0.000000],
    [ 0.000000, 0.000000, -1.000200, -0.200020],
    [ 0.000000, 0.000000, -1.000000,  0.000000]])

# print(cam_proj)
# print(Project_Matrix)

def perspective_depth(d, n, f):
    return -((2 * n) / d - f - n) / (f - n)

def camera_space(D):
    D = perspective_depth(1-D, clip_near, clip_far) * 2.0 - 1.0
    U = np.empty((D.shape[0], D.shape[1], 2))
    U[:,:,0] = (((np.arange(D.shape[0])+0.5)/D.shape[0]) * 2.0 - 1.0)[:,np.newaxis]
    U[:,:,1] = (((np.arange(D.shape[1])+0.5)/D.shape[1]) * 2.0 - 1.0)[np.newaxis,:]
    P = np.concatenate([U[:,:,0:1], U[:,:,1:2], D, np.ones(D.shape)], axis=-1)
    P = cam_inv_proj.dot(P.reshape(-1,4).T).T.reshape(P.shape)
    return P[:,:,:3] / P[:,:,3:4]


_ZBufferParams = [1-clip_far/clip_near, clip_far/clip_near, (1-clip_far/clip_near)/clip_far, 1/clip_near]
_ProjectionParams = [1, clip_near, clip_far, 1/clip_far]


def Linear01Depth(z):
    return 1.0 / (_ZBufferParams[0] * z + _ZBufferParams[1])


def ReverseLinear01Depth(z):
    return (1.0/z-_ZBufferParams[1])/_ZBufferParams[0]


def LinearEyeDepth(z):
    return 1.0 / (_ZBufferParams[2] * z + _ZBufferParams[3])


def ReconstructViewPosition(D):
    U = np.empty((D.shape[0], D.shape[1], 2))
    U[:, :, 0] = (((np.arange(D.shape[0]-1, -1, -1)+0.5)/D.shape[0]) * 2.0 - 1.0)[:,np.newaxis]
    U[:, :, 1] = (((np.arange(D.shape[1])+0.5)/D.shape[1]) * 2.0 - 1.0)[np.newaxis, :]
    # U = U/2.14451*D

    U[:, :, 0:1] = U[:, :, 0:1] / Project_Inv_Matrix[0, 0] * D
    U[:, :, 1:2] = U[:, :, 1:2] / Project_Inv_Matrix[1, 1] * D
    P = np.concatenate([U[:, :, 0:1], U[:, :, 1:2], D, np.ones(D.shape)], axis=-1)
    # P = cam_proj_unity.dot(P.reshape(-1, 4).T).T.reshape(P.shape)
    return P[:, :, :3]


def HBAO(depth, normal):
    pass
nnao_positions = camera_space(positions[:, :, 2:3])
our_positions = ReconstructViewPosition((1-positions[:, :, 2:3])*_ProjectionParams[2])
our_positions = our_positions/aspect
our_positions[:, :, 2:3] = 1-our_positions[:, :, 2:3]

# nnao_positions = (nnao_positions+500)/1000
# our_positions = (our_positions+500)/1000

plt.subplot(331), plt.imshow(nnao_positions[:, :, 0])
plt.subplot(332), plt.imshow(nnao_positions[:, :, 1])
plt.subplot(333), plt.imshow(nnao_positions[:, :, 2])

plt.subplot(334), plt.imshow(our_positions[:, :, 0])
plt.subplot(335), plt.imshow(our_positions[:, :, 1])
plt.subplot(336), plt.imshow(our_positions[:, :, 2])

error = positions[:, :, (1, 0, 2)]
plt.subplot(337), plt.imshow(error[:, :, 0])
plt.subplot(338), plt.imshow(error[:, :, 1])
plt.subplot(339), plt.imshow(error[:, :, 2])

print(np.max(error), np.min(error), np.mean(error))
plt.show()




# positions1 = camera_space(positions1[:, :, 2:3])
positions = positions[:, :, (1, 0, 2)]

plt.subplot(231), plt.imshow(positions[:, :, 0])
plt.subplot(232), plt.imshow(positions[:, :, 1])
plt.subplot(233), plt.imshow(positions[:, :, 2])

depth = (1-positions[:, :, 2])*_ProjectionParams[2]
# depth = ReverseLinear01Depth(depth/_ProjectionParams[2])
depth = ReconstructViewPosition(np.expand_dims(depth, axis=-1))

# print(np.mean(np.abs(depth[:, :, :3]-positions[:, :, :3])))
xx = (depth[:, :, 0]+750.0)/_ProjectionParams[2]
yy = (depth[:, :, 1]+750.0)/_ProjectionParams[2]
zz = 1-depth[:, :, 2]/_ProjectionParams[2]

# plt.imshow(depth-positions[:, :, :3])
# plt.show()
# xx = depth[:, :, 0]
# yy = depth[:, :, 1]
# zz = depth[:, :, 2]

plt.subplot(234), plt.imshow(xx)
plt.subplot(235), plt.imshow(yy)
plt.subplot(236), plt.imshow(zz)
plt.show()
# plt.show()
# camera_cx = 992.6047957710321
# camera_fx = -425.16151464980805
# positions = np.where(positions>1, 1, positions)
# positions = np.where(positions<-1, -1, positions)
# print(np.max(positions[:, :, 0]),np.min(positions[:, :, 0]), np.mean(positions[:, :, 0]))
# print(np.max(positions[:, :, 1]),np.min(positions[:, :, 1]), np.mean(positions[:, :, 1]))
# print(np.max(positions[:, :, 2]),np.min(positions[:, :, 2]), np.mean(positions[:, :, 2]))
# positions1 *= 1500

# xx = positions1[:, :, 0]-750.0
# yy = positions1[:, :, 1]-750.0
# zz = 1-positions1[:, :, 2]
# plt.imshow(positions1)
# plt.show()
# xx = (750+positions[:, :, 0])/1500
# yy = (750+positions[:, :, 1])/1500
# zz = (1500+positions[:, :, 2])/1500
# cx = (positions[:, :, 0]-np.min(positions[:, :, 0]))/(np.max(positions[:, :, 0])-np.min(positions[:, :, 0]))
# plt.subplot(234)
# plt.imshow(xx)
#
# plt.subplot(235)
# plt.imshow(yy)
#
# plt.subplot(236)
# plt.imshow(zz)
# plt.show()
# position = positions[:, :, 0]
# # def  GammaToLinearSpace (sRGB):
# #     return sRGB * (sRGB * (sRGB * 0.305306011 + 0.682171111) + 0.012522878)
#
#
# # def GammaToLinearSpaceExact(value):
# #
# #     if value <= 0.04045:
# #         return value / 12.92
# #     elif value < 1.0:
# #         return pow((value + 0.055)/1.055, 2.4)
# #     else:
# #         return pow(value, 2.2)
# # camera
# f = open('D:\\Projects\\Datasets\\3dsmax\\evaluate\\Test\\scene000\\Position\\s2222.obj', mode='w')
#
# row_map = np.ones(shape=(512, 512), dtype=np.float)*np.arange(0, 512)
# col_map = np.transpose(row_map)
#
# row_map = np.expand_dims(row_map, axis=-1)-camera_cx
# col_map = np.expand_dims(col_map, axis=-1)-camera_cy
#
# # camera_fx -= 420
# def depth2position(depth_map):
#
#     xx = np.multiply(row_map, 1-depth_map)/camera_fx
#     yy = np.multiply(col_map, 1-depth_map) / camera_fx
#     plt.subplot(234)
#     plt.imshow(np.squeeze(xx))
#     plt.subplot(233)
#     plt.imshow(zz)
#     plt.subplot(235)
#     plt.imshow(1-np.squeeze(yy))
#
#     plt.subplot(236)
#     plt.imshow(positions1[:, :, 2])
#
#     # plt.subplot(236)
#     # plt.imshow(np.squeeze(depth_map))
#     plt.show()
#     return np.concatenate([yy, xx, depth_map], axis=-1)
#
# res = depth2position(np.expand_dims(position, axis=-1))
# # print()
# for n in range(position.shape[0]):
#     for m in range(position.shape[1]):
#         d = position[n, m]
#         z = 1-d
#         x = (n - camera_cx) * z / camera_fx
#         y = (m - camera_cy) * z / camera_fy
#
#         f.write('v %s %s %s\n' % (x, y, z))
#         # f.write('v %s %s %s\n' % (res[n, m, 0], res[n, m, 1], res[n, m, 2]))
#
# f.close()
