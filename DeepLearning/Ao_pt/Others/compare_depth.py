import numpy as np
import pyexr
import matplotlib.pyplot as plt
import os

base_dir = 'C:\\Users\\39796\\Desktop\\Ambient Occlosion Paper\\scene_image'
# 'scene_18_Perspective_View180014'
position_path = os.path.join(base_dir, 'scene_18_Perspective_View130013.exr')
# depth_path = os.path.join(base_dir, 'scene_19_Camera001_Z Depth.exr')
# 'scene_18_Perspective_rgb_Z Depth_View090049.exr'
# position = 1+pyexr.open(position_path).get()[:, :, 2]/1500.0
position = pyexr.open(position_path).get()[:, :, :]
print(np.mean(position[:,:, 0]))
# depth = pyexr.open(depth_path).get()[:, :, 2]
# plt.imsave('%s\\scene_18_Perspective_rgb_Z Depth_View090049.png'%base_dir, position)

# plt.subplot(131)
plt.imshow(position)
# plt.subplot(132)
# plt.imshow(depth)
# plt.subplot(133)
# plt.imshow(position-depth)
plt.show()