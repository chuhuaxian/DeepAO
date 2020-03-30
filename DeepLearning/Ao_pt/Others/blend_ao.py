import pyexr
import cv2
import matplotlib.pyplot as plt
import numpy as np

ours_128 = pyexr.open('C:\\Users\\zhangdongjiu\\Desktop\\s06-c3_128-exp4-ours.exr').get()[:, :, 0]
ours_2048 = pyexr.open('C:\\Users\\zhangdongjiu\\Desktop\\scene_15_Camera003_2048.exr').get()[:, :, 0]

hbao = pyexr.open('C:\\Users\\zhangdongjiu\\Desktop\\S06-C3_2048-1-hbao.exr').get()[:, :, 0]

blend_ = cv2.resize(ours_128, (0, 0), fx=16, fy=16, interpolation=cv2.INTER_CUBIC)
plt.subplot(221)
plt.imshow(np.clip(ours_128*255., 0, 255).astype(np.uint8), cmap='gray')

plt.subplot(222)
plt.imshow(np.clip(hbao*255., 0, 255).astype(np.uint8), cmap='gray')

plt.subplot(223)
plt.imshow(np.clip((blend_+hbao)*255./2., 0, 255).astype(np.uint8), cmap='gray')

plt.subplot(224)
# plt.imshow(np.clip(cv2.resize(ours_128, (0, 0), fx=16, fy=16, interpolation=cv2.INTER_CUBIC)*255., 0, 255).astype(np.uint8), cmap='gray')
plt.imshow(np.clip(ours_2048*255., 0, 255).astype(np.uint8), cmap='gray')
plt.show()