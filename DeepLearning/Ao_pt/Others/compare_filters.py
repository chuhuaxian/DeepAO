import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from Myloss import SSIM
import os


fnk4_2_lst = [os.path.join('Logs\\K4_2', i) for i in os.listdir('Logs\\K4_2')]
count_ = 0
for i in range(2249):
    # fnk4 = 'Logs\\K4\\%05d-result.png' % i
    fnk4 = fnk4_2_lst[count_]
    # print(fnk4)
    fnk8 = 'Logs\\K8\\%05d-result.png' % i
    # print(fnk8)
    fnk16 = 'Logs\\K16\\%05d-result.png' % i
    fnkgt = 'Logs\\GT\\%05d-gt.png' % i

    if os.path.exists(fnk8) == False:
        continue
    K4 = cv2.imread(fnk4)
    K8 = cv2.imread(fnk8)
    K16 = cv2.imread(fnk16)
    Kgt = cv2.imread(fnkgt)
    count_ += 1

    # plt.subplot(221)
    # plt.imshow(K4)
    # plt.subplot(222)
    # plt.imshow(K8)
    # plt.subplot(223)
    # plt.imshow(K16)
    # plt.subplot(224)
    # plt.imshow(Kgt)
    # plt.show()
    gap = np.zeros((K4.shape[0], 5, 3), dtype=np.uint8)
    res = np.concatenate((K4, gap, K8, gap, K16, gap, Kgt), axis=1)

    K4 = K4.astype(np.float32)
    K4 = K4[:, :, 0]/255.

    K8 = K8.astype(np.float32)
    K8 = K8[:, :, 0]/255.

    K16 = K16.astype(np.float32)
    K16 = K16[:, :, 0]/255.

    Kgt = Kgt.astype(np.float32)
    Kgt = Kgt[:, :, 0]/255.
    Kgt = torch.from_numpy(np.expand_dims(np.expand_dims(Kgt, 0), 0)).cuda()

    ssim_k4 = SSIM()(Kgt, torch.from_numpy(np.expand_dims(np.expand_dims(K4, 0), 0)).cuda()).item()
    ssim_k8 = SSIM()(Kgt, torch.from_numpy(np.expand_dims(np.expand_dims(K8, 0), 0)).cuda()).item()
    ssim_k16 = SSIM()(Kgt, torch.from_numpy(np.expand_dims(np.expand_dims(K16, 0), 0)).cuda()).item()

    cv2.imwrite('Logs\\COMPARE_1\\%05d-%.4f-%.4f-%.4f-res.png' % (i, ssim_k4, ssim_k8, ssim_k16), res)
    # plt.imshow(res)
    # plt.show()
