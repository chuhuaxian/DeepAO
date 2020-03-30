import os
import cv2
import pyexr
import numpy as np
import matplotlib.pyplot as plt
# from xml.etree.ElementTree import ElementTree
# tree = ElementTree()
# res = tree.parse('C:\\Users\\39796\\Desktop\\test.xaf').findall('Node')
#
# for i in res:
#     print(i)
# node = res.find('Samples')

camera_cx = 256.0
camera_cy = 256.0
camera_fx = 548.9937
camera_fy = 548.9937


# print(res)
def analysis_xaf(filename):
    f = open(filename)
    lines = f.readlines()
    Samples = [i for i in lines if '<S t=' in i]
    Samples = [i.split('"')[3].strip().split(' ') for i in Samples]
    Samples = [[float(j) for j in i] for i in Samples]

    Samples1 = [i for i in lines if '<MVal' in i]
    Samples1 = [i.split('"')[3].strip().split(' ') for i in Samples1]
    Samples1 = [[float(j) for j in i] for i in Samples1]
    return Samples, Samples1


def depth2obj(position, filename, ts_matrix):
    f = open(filename, mode='w')
    for n in range(position.shape[0]):
        for m in range(position.shape[1]):
            d = position[n, m]
            z = 1 - d
            x = (n - camera_cx) * z / camera_fx
            y = (m - camera_cy) * z / camera_fy
            f.write('v %s %s %s\n' % (x, y, z))
    f.close()

s, s1 = analysis_xaf('C:\\Users\\39796\\Desktop\\c1.xaf')


files = [os.path.join('D:\\Projects\\Datasets\\3dsmax\scene\\scene14', i) for i in os.listdir('D:\\Projects\\Datasets\\3dsmax\scene\\scene14') if 'View01' in i]
depth = [i for i in files if 'Z D'in i]
fn1, fn2 = depth[0], depth[1]

d1 = pyexr.open(fn1).get()[:, :, 0]
d2 = pyexr.open(fn2).get()[:, :, 0]

depth2obj(d2, 'C:\\Users\\39796\\Desktop\\d2.obj', np.array(s[1]).reshape((4, 3)))
# plt.imshow(d1)
# plt.show()
# img =

print()




