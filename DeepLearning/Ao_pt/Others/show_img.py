import matplotlib.pyplot as plt
import pyexr
import numpy as np
import math
img = pyexr.open('C:\\Users\zhangdongjiu\\test_01\screenshot\\0-out_lk_1.exr').get()[:, :, 0]
img1 = pyexr.open('C:\\Users\zhangdongjiu\\test_01\screenshot\\0-out_maxpool_1.exr').get()[:, :, 0]


res = np.zeros(shape=(img1.shape), dtype=np.float32)
for i in range(res.shape[0]):
    for j in range(res.shape[1]):
        temp = img[i*2, j*2]
        temp = max(temp, img[i*2+1, j*2])
        temp = max(temp, img[i * 2, j * 2+1])
        temp = max(temp, img[i * 2 + 1, j * 2+1])
        res[i, j] = temp
# print(img.shape)
# print(np.unique(img).shape)
plt.subplot(131)

# gamma, beta, mean, var = 0.1709389,0.39395604,-0.16607302,0.0060356096
# img = gamma*((img-mean)/(math.sqrt(var+1e-5)))+beta
plt.imshow(res)


plt.subplot(132)
plt.imshow(img1)

plt.subplot(133)
plt.imshow(img1-res)

print(np.mean(img1-res))
plt.show()