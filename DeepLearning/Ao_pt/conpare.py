import pyexr
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import cv2
from Myloss import MY_SSIM
from pytorch_msssim import SSIM
# fp_hbao = 'C:\\Users\\39796\\Desktop\\compare\\1-hbao.exr'
# fp_ours = 'C:\\Users\\39796\\Desktop\\compare\\1-ours.exr'
# fp_gt = 'C:\\Users\\39796\\Desktop\\compare\\1-gt.exr'
# fp_nnao = 'C:\\Users\\39796\\Desktop\\compare\\1-nnao.exr'
# fp_deepshading = 'C:\\Users\\39796\\Desktop\\compare\\1-deepshading.exr'

order = 3
base_dir = 'D:\\Projects\\Python\\PycharmProjects\\Ao_pt\\Logs\\test\\i%s'%order
fp_hbao = os.path.join(base_dir,'%s-hbao.exr'%order)
fp_ours = os.path.join(base_dir,'%s-ours.exr'%order)
fp_gt = os.path.join(base_dir,'%s-gt.exr'%order)
fp_nnao = os.path.join(base_dir,'%s-nnao.exr'%order)
fp_deepshading = os.path.join(base_dir,'%s-deepshading.exr'%order)
fp_vao = os.path.join(base_dir,'%s-vao++.exr'%order)

fp_dgi = os.path.join(base_dir,'%s-dgi.png'%order)

gt = pyexr.open(fp_gt).get()[:, :, 0]
ours = pyexr.open(fp_ours).get()[:, :, 0]
# hbao = pyexr.open(fp_hbao).get()[:, :, 0]
# nnao = pyexr.open(fp_nnao).get()[:, :, 0]
# deepshading = pyexr.open(fp_deepshading).get()[:, :, 0]
vao = pyexr.open(fp_vao).get()[:, :, 0]
# dgi = cv2.imread(fp_dgi)
# dgi = dgi[:, :, (2, 1, 0)]
plt.imsave('C:\\Users\\39796\Desktop\\Ambient Occlosion Paper\\Experiment\\%s-gt.png' % order, gt, cmap='gray')
# plt.imsave('C:\\Users\\39796\Desktop\\Ambient Occlosion Paper\\Experiment\\%s-hbao.png' % order, hbao, cmap='gray')
# plt.imsave('C:\\Users\\39796\Desktop\\Ambient Occlosion Paper\\Experiment\\%s-nnao.png' % order, nnao, cmap='gray')
# plt.imsave('C:\\Users\\39796\Desktop\\Ambient Occlosion Paper\\Experiment\\%s-deepshading.png' % order, deepshading, cmap='gray')
plt.imsave('C:\\Users\\39796\Desktop\\Ambient Occlosion Paper\\Experiment\\%s-ours.png' % order, ours, cmap='gray')
plt.imsave('C:\\Users\\39796\Desktop\\Ambient Occlosion Paper\\Experiment\\%s-vao.png' % order, vao, cmap='gray')

# gt1 = gt
# ours1 = ours
# hbao1 = hbao
# nnao1 = nnao
# deepshading1 = deepshading

gt = np.expand_dims(np.expand_dims(gt, 0), axis=0)
ours = np.expand_dims(np.expand_dims(ours, 0), axis=0)
# hbao = np.expand_dims(np.expand_dims(hbao, 0), axis=0)
# nnao = np.expand_dims(np.expand_dims(nnao, 0), axis=0)
# deepshading = np.expand_dims(np.expand_dims(deepshading, 0), axis=0)
vao = np.expand_dims(np.expand_dims(vao, 0), axis=0)




gt = torch.from_numpy(gt).cuda()
ours = torch.from_numpy(ours).cuda()
# hbao = torch.from_numpy(hbao).cuda()
# nnao = torch.from_numpy(nnao).cuda()
# deepshading = torch.from_numpy(deepshading).cuda()
vao = torch.from_numpy(vao).cuda()

# ssim_hbao = SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=1)(gt, hbao).item()
ssim_ours = SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=1)(gt, ours).item()
# ssim_nnao = SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=1)(gt, nnao).item()
# ssim_deepshading = SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=1)(gt, deepshading).item()
ssim_vao = SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=1)(gt, vao).item()

print(ssim_ours, ssim_vao)
# mse_hbao = torch.nn.L1Loss()(gt, hbao).item()
mse_ours = torch.nn.L1Loss()(gt, ours).item()
# mse_nnao = torch.nn.L1Loss()(gt, nnao).item()
# mse_deepshading = torch.nn.L1Loss()(gt, deepshading).item()
mse_vao = torch.nn.L1Loss()(gt, vao).item()
print(mse_ours, mse_vao)
# def blender(rgb, ao):
#     ao = np.squeeze(ao)
#     ao = np.expand_dims(ao, axis=-1)
#     ao = np.concatenate([ao, ao, ao], axis=-1)
#     return ao*rgb
# plt.subplot(231)
# plt.title('HBAO: SSIM=%04f, MSE=%04f' %(ssim_hbao,mse_hbao )), plt.imshow(hbao1, cmap='gray')
# # plt.show()
# plt.subplot(232)
# plt.title('NNAO: SSIM=%04f, MSE=%04f' %(ssim_nnao,mse_nnao )),plt.imshow(nnao1, cmap='gray')
# plt.subplot(235)
# plt.title('OURS: SSIM=%04f, MSE=%04f' %(ssim_ours,mse_ours )),plt.imshow(ours1, cmap='gray')
# plt.subplot(234)
# plt.title('DEEPSHADING: SSIM=%04f, MSE=%04f' %(ssim_deepshading,mse_deepshading )),plt.imshow(deepshading1, cmap='gray')
# plt.subplot(236)
# plt.imshow(gt1, cmap='gray')
# plt.show()



# print('SSIM_hbao = %.3f, SSIM_nnao = %.3f, SSIM_deepshading=%.3f, SSIM_ours = %.3f' %(ssim_hbao, ssim_nnao, ssim_deepshading, ssim_ours) )
# print('MSE_hbao = %.3f, MSE_nnao = %.3f, MSE_deepshading=%.3f, MSE_ours = %.3f' %(mse_hbao, mse_nnao, mse_deepshading, mse_ours) )