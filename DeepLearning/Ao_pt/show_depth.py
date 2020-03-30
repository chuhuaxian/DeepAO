import pyexr
import os
import matplotlib.pyplot as plt
import cv2
base_dir = 'C:\\Users\\39796\\Desktop'

depth_path = os.path.join(base_dir, 'scene_06_Camera021_depth.exr')

depth = pyexr.open(depth_path).get()[:, :, 2]
# depth_path =

plt.imsave('C:\\Users\\39796\\Desktop\\depth.png', depth)
plt.imshow(depth)
plt.show()