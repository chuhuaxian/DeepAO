import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from Myloss import MY_SSIM
import os
import pyexr

fp_hbao = 'C:\\Users\\39796\\Desktop\\1-hbao.exr'
fp_ours = 'C:\\Users\\39796\\Desktop\\1-ours.exr'
fp_gt = 'C:\\Users\\39796\\Desktop\\guest room_Camera001.exr'

gt = torch.from_numpy(pyexr.open(fp_gt).get())
ours = torch.from_numpy(pyexr.open(fp_ours).get())
hbao = torch.from_numpy(pyexr.open(fp_hbao).get())

ssim_hbao = MY_SSIM()(gt, hbao).item()
ssim_ours = MY_SSIM()(gt, ours).item()

print(ssim_hbao, '\n', ssim_ours)



