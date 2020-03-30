# for i in range(1800):
#     str_ = ''
#     for j in range(3, -1, -1):
#         k = i-j if (i//100*100)==(i-j)//100*100 else (i//100*100)
#         str_ += '%s,' % k
#         # print(i, (i//100*100)==(i-j)//100*100, k)
#     print(str_)

# xx = 6*8 + 8*16 + 16*32 + 32*64 + 64*128 + 128*256
# xx *= 9
# print(xx)

# r1, c1, x1, y1, z1 = 262, 203, 0.55126953, 0.4482422, 0.32080078
# r2, c2, x2, y2, z2 = 312, 259, 0.5756836, 0.43041992, 0.35961914
# r3, c3, x3, y3, z3 = 383, 179, 0.55078125, 0.4116211, 0.39233398
# r4, c4, x4, y4, z4 = 368, 231, 0.56640625, 0.41479492, 0.38549805
# r5, c5, x5, y5, z5 = 343, 81,  0.51953125, 0.42382812, 0.39111328
# r6, c6, x6, y6, z6 = 0, 0, 0.38793975, 0.6376953, 0.16186523
# def compute_cx_fx(r1, x1, z1, r2, x2, z2):
#     c_x = (r1*z1*x2-r2*z2*x1)/(z1*x2-z2*x1)
#     f_x = (r1-c_x)*z1/x1
#     return c_x, f_x
# 
# ccx, cfx = compute_cx_fx(r1, x1, z1, r2, x2, z2)
# ccy, cfy = compute_cx_fx(c3, y3, z3, c2, y2, z2)
# 
# print(ccx, cfx, ccy, cfy)
# def compute_x(r, d):
#     return (r - ccx) * d / cfx
# 
# def compute_y(c, d):
#     return (c - ccy) * d / cfy
# 
# 
# print(compute_x(r6, z6))
# print(compute_y(c6, z6))
import numpy as np
import matplotlib.pyplot as plt
import pyexr
# dp = 'C:\\Users\\39796\Desktop\\Touareg_Camera001.exr'
# depth = pyexr.open(dp).get()
#
# plt.imshow(depth[:, :, 2], cmap='gray')
# plt.show()
#
# SSIM_hbao = 0.814 +0.679+0.845+0.746+0.819+0.874+0.892+0.919+0.742+0.754+0.825+0.904+0.870
# # print(SSIM_hbao/13)
# SSIM_nnao = 0.296+0.312+0.595+0.245+0.258+0.461+0.291+0.389+0.231+0.263+0.295+0.462+0.589
# # print(SSIM_nnao/13)
#
# print(np.sqrt(0.666))
# print(np.sqrt(0.510))
# SSIM_deepshading=0.739
# SSIM_ours = 0.833+0.764+0.887+0.749+0.853+0.898+0.939+0.941+0.750+0.765+0.829+0.957+0.884
# # print(SSIM_ours/13)
# MSE_hbao = 0.074+0.089+0.040+0.081+0.073+0.049+0.046+0.031+0.077+0.063+0.045+0.037+0.049
# # print(MSE_hbao/13)
# MSE_nnao = 0.136+0.150+0.097+0.144+0.125+0.086+0.092+0.079+0.150+0.188+0.125+0.070+0.080
# # print(MSE_nnao/13)
#
#
#
# MSE_deepshading=0.133
# MSE_ours = 0.061+0.064+0.032+0.069+0.042

dp_ao = 'C:\\Users\\39796\Desktop\\scene_151_Camera001.exr'
dp_rgb = 'C:\\Users\\39796\Desktop\\sponza_Camera001_rgb.exr'

def blender(rgb, ao):
    ao = np.squeeze(ao)
    ao = np.expand_dims(ao, axis=-1)
    ao = np.concatenate([ao, ao, ao], axis=-1)
    return ao*rgb

ours = pyexr.open(dp_ao).get()[:, :, 0]

plt.imsave('C:\\Users\\39796\Desktop\\scene_151_Camera001_ao.png', ours, cmap='gray')
print()
rgb = pyexr.open(dp_rgb).get()[:, :, :3]

plt.imsave('C:\\Users\\39796\Desktop\\sponza_rgb.png', ours, cmap='gray')
# plt.imsave('C:\\Users\\39796\Desktop\\2-rgb_out.png', rgb)

rgb*=4
rgb = np.clip(rgb, 0.0, 1.0)
plt.imsave('C:\\Users\\39796\Desktop\\sponza_rgb_out.png', rgb)
res = blender(rgb, ours)
# print(np.max(res))
res = np.clip(res, 0.0, 1.0)
plt.imsave('C:\\Users\\39796\Desktop\\sponza-rgb_blend.png', res)
plt.imshow(res)
plt.show()






