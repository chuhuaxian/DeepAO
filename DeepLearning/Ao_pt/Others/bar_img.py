import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# import seaborn as sns
import numpy as np

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()-0.125, height+0.1, '%s' % float(height), fontdict={'fontsize':25})


# font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)

wid = 0.1
scale = 2
# a = plt.bar([1/scale], [2.1], width=wid, label='4x')
#
# b = plt.bar([2/scale], [10.8], width=wid, label='8x')
#
# c = plt.bar([3/scale], [42.5], width=wid, label='16x')

# d = plt.bar([4/scale], [33.6], width=wid, label='1920x1080')
lst = np.array([1.74,10.32,8.6,
                4.66,10.16,8.3,
5.19,10.15,8.9,
5.51,10.17,9.3,
6.28,10.7,9.4,
6.72,11.21,9.2,
8.29,13.11,10.5,
12.98,21.49,11.7,
16.08,21.06,11.8,
20.60,25.52,16.5,
24.21,29.02,17.5,
29.31,34.18,25.3,
34.93,39.49,26.6,
40.36,45.52,25.3,
40.53,45.92,24.9]).reshape((15, 3))
print(lst[1:, :]-lst[:-1, :])



# a = plt.bar([1/scale, 2/scale, 3/scale], [1.0,1.1,15.6], width=wid, label='720p')
#
# b = plt.bar([5/scale, 6/scale, 7/scale], [1.8,2.3,38.7], width=wid, label='2k')
#
# c = plt.bar([9/scale, 10/scale, 11/scale], [7.3,8.8,153.9], width=wid, label='4k')
#
# # d = plt.bar([4/scale], [33.6], width=wid, label='1920x1080')
#
# # a.spines['top'].set_visible(False)
# # params
#
# # x: 条形图x轴
# # y：条形图的高度
# # width：条形图的宽度 默认是0.8
# # bottom：条形底部的y坐标值 默认是0
# # align：center / edge 条形图是否以x轴坐标为中心点或者是以x轴坐标为边缘
# autolabel(a)
# autolabel(b)
# autolabel(c)
# # autolabel(d)
#
# plt.legend(fontsize=20)
#
# plt.xlabel('Resolution', fontdict={'fontsize':15})
# plt.ylabel('Time(ms)', fontdict={'fontsize':15})
# plt.xticks([])
# plt.title(u'Time of 4x Filters', fontdict={'fontsize':15})
#
# plt.show()