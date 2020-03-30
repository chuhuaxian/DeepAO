import pyexr
import matplotlib.pyplot as plt
import numpy as np
import cv2
normal_path = 'D:\\Projects\\Unity\\test_03\screenshot\\scene_00_Perspective_normal_View010005.exr'
depth_path = 'D:\\Projects\\Unity\\test_03\screenshot\\scene_00_Perspective_Z Depth_View010005.exr'

normal = pyexr.open(normal_path).get()

normal = normal*255
normal = normal.astype(np.uint8)
normal = normal[:, :, (2, 1, 0)]
depth = pyexr.open(depth_path).get()[:, :, 0:1]*65535
xx = depth / 255
yy = depth % 255

res = np.concatenate([xx, yy, np.ones((512, 512, 1) )], axis=-1)
res = res.astype(np.uint8)
res = res[:, :, (2, 1, 0)]
cv2.imwrite('D:\\Projects\\Unity\\test_03\screenshot\\normal-3.png', normal)
cv2.imwrite('D:\\Projects\\Unity\\test_03\screenshot\\depth-3.png', res)
print(np.sum(np.unique(normal[:, :, 0])!=0))

