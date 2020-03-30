# # import numpy as np
# # import os
# # import cv2
# # # # import mat
# # # # datapath = os.path.join('Datasets', "dataset_full.npy")
# # # #
# # # # # data = np.load(datapath)
# # # # import torch
# # # #
# # # # inp = torch.randn(1, 3, 10, 12)
# # # # w = torch.randn(2, 3, 3, 3)
# # # # inp_unf = torch.nn.functional.unfold(inp, (3, 3), padding=1)
# # # # out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
# # # # out = torch.nn.functional.fold(out_unf, (10, 12), (1, 1))
# # # #
# # # # # or equivalently (and avoiding a copy),
# # # # # out = out_unf.view(1, 2, 7, 8)
# # # # a = torch.nn.functional.conv2d(inp, w, padding=1)
# # # # xx = (a - out).abs()
# # # # print(torch.max(xx))
# # #
# # # img_dir = 'C:\\Users\\39796\\PycharmProjects\\neterase\\nnao_pt\\Logs\\result'
# # img_dir = 'C:\\Users\\39796\\PycharmProjects\\neterase\Datasets\\train\Cars\Position'
# # img_lst = [os.path.join(img_dir, i) for i in os.listdir(img_dir)]
# # import pyexr
# #
# # import matplotlib.pyplot as plt
# # for i in img_lst:
# #     img = pyexr.open(i).get()*65535
# #     plt.subplot(131)
# #     plt.imshow(img[:, :, 0])
# #     plt.subplot(132)
# #     plt.imshow(img[:, :, 1])
# #     plt.subplot(133)
# #     plt.imshow(img[:, :, 2])
# #     plt.show()
# #     # img = np.where(img < 10, 255, img)
# #     # cv2.imwrite(i, img)
# #     # cv2.imshow('', img)
# #     # cv2.waitKey(0)
# #     # print()
# #
# #
# #
# #
#
#
import matplotlib.pyplot as plt
#
import numpy as np

# 'net_params_20_2019-08-26_09-38.pkl' # 4
# 'net_params_26_2019-08-23_12-27.pkl' # 8
# 'net_params_24_2019-08-23_17-06.pkl' #16
convs = [1.3, 0.8, 0.6, 0.5, 1.1, 3.5, 3.4, 2.9, 2.9, 1.4]

pools = [0.1, 0.1, 0.1, 0.1]
ups = [0.1, 1.0, 1.0, 2.5]
cats = [1.7, 1.9, 2.8, 2.9]
total = sum(convs)+ sum(pools)+ sum(ups)+ sum(cats)
print(sum(ups))
print(sum(cats))
# data = []
# data = [int(100*sum(convs)/33.6)+3, int(100*sum(pools)/33.6), int(100*sum(ups)/33.6)+1, 1+int(100*sum(cats)/33.6)]
# data = [100*i/sum(convs) for i in convs]
data = [sum(convs)/total, sum(pools)/total, sum(ups)/total, sum(cats)/total]
# data = [100*i/sum(convs) for i in convs]
# print(data)
# data = [15, 30, 45, 10]
lables = 'Conv', 'Pool', 'UpSample', 'Cat'
# lables = 'up_1', 'up_2', 'up_3', 'up_4'
# lables = 'cat_1', 'cat_2', 'cat_3', 'cat_4'
# lables = 'Conv_in', 'Conv_down_1', 'Conv_down_2', 'Conv_down_3', 'Conv_down_4', 'Conv_up_1', 'Conv_up_2','Conv_up_3','Conv_up_4', 'Conv_out'
plt.axes(aspect=1)
patches,l_text,p_text = plt.pie(x=data, labels=lables, autopct='%.0f%%', textprops={'fontsize':15,'color':'black'})
# plt.legend(props={'fontsize':20,'color':'black'})
# for t in l_text:
#     t.set_size=(100)
# for t in p_text:
#     t.set_size=(20)

plt.show()
#
# x = np.linspace(-10, 10, 60)
#
#
# def elu(x, a):
#
#     y = []
#     for i in x:
#         if i >=0:
#             y.append(i)
#         else:
#
#              y.append(a * np.exp(i) -1)
#
#     return y
#
# relu = np.maximum(x, [0] *60)
#
# relu6 = np.minimum(np.maximum(x, [0] *60), [6] *60)
#
# softplus = np.log(np.exp(x) +1)
#
# elu = elu(x, 1)
#
# softsign = x / (np.abs(x) +1)
#
# sigmoid =1 / (1 + np.exp(-x))
#
# tanh = np.tanh(x)
#
# lrelu = np.maximum(0.1 * x, x)
#
# plt.figure()
#
# # plt.plot(x, relu6, label='relu6', linewidth=3.0)
#
# plt.plot(x, relu, label='relu', color='black', linestyle='--', linewidth=2.0)
#
# # plt.plot(x, elu, label='elu', linewidth=2.0)
#
# plt.plot(x, lrelu, label='lrelu', linewidth=1.0)
#
# ax = plt.gca()
#
# ax.spines['right'].set_color('none')
#
# ax.spines['top'].set_color('none')
#
# ax.xaxis.set_ticks_position('bottom')
#
# ax.spines['bottom'].set_position(('data', 0))
#
# ax.yaxis.set_ticks_position('left')
#
# ax.spines['left'].set_position(('data', 0))
#
# plt.legend(loc='best')
#
# plt.figure()
#
# plt.ylim((-1.2, 1.2))
#
# # plt.plot(x, softsign, label='softsign', linewidth=2.0)
#
# plt.plot(x, sigmoid, label='sigmoid', linewidth=2.0)
#
# plt.plot(x, tanh, label='tanh', linewidth=2.0)
#
# # plt.plot(x, softplus, label='softplus', linewidth=2.0)
#
# # plt.plot(x, hyperbolic_tangent,label='hyperbolic_tangent',linewidth=2.0)
#
# ax = plt.gca()
#
# ax.spines['right'].set_color('none')
#
# ax.spines['top'].set_color('none')
#
# ax.xaxis.set_ticks_position('bottom')
#
# ax.spines['bottom'].set_position(('data', 0))
#
# ax.yaxis.set_ticks_position('left')
#
# ax.spines['left'].set_position(('data', 0))
#
# plt.legend(loc='best')
#
# plt.show()
# # import torch
# # from Models.NANO import NANO, NANO_cnn
# #
# #
# # class inputLayer(object):
# #     def __init__(self,
# #                  InputShape=(0, 0, 0),
# #                  OutputShape=(0, 0, 0),
# #                  WeightShape=(0.0, 0.0, 0.0, 0.0),
# #                  InputChannels=4,
# #                  src=0):
# #         self.Name = 'input_normalize_1'
# #         self.InputShape = {"x": InputShape[0], "y": InputShape[1], "z": InputShape[2]}
# #         self.OutputShape = {"x": OutputShape[0], "y": OutputShape[1], "z": OutputShape[2]}
# #         self.WeightShape = {"x": WeightShape[0], "y": WeightShape[1], "z": WeightShape[2], "w": WeightShape[3]}
# #         self.InputChannels = InputChannels
# #         self.src = {"instanceID": src}
# #
# #
# # class outputLayer(object):
# #     def __init__(self,
# #                  InputShape=(0, 0, 0),
# #                  OutputShape=(0, 0, 0),
# #                  WeightShape=(0.0, 0.0, 0.0, 0.0)):
# #         self.Name = 'transform_output'
# #         self.InputShape = {"x": InputShape[0], "y": InputShape[1], "z": InputShape[2]}
# #         self.OutputShape = {"x": OutputShape[0], "y": OutputShape[1], "z": OutputShape[2]}
# #         self.WeightShape = {"x": WeightShape[0], "y": WeightShape[1], "z": WeightShape[2], "w": WeightShape[3]}
# #
# #
# # class conv2dLayer(object):
# #     def __init__(self,
# #                  name,
# #                  InputShape=(0, 0, 0),
# #                  OutputShape=(0, 0, 0),
# #                  WeightShape=(3.0, 3.0, 16.0, 4.0),
# #                  weightcache=None,
# #                  Filters=3,
# #                  KernalSize=(3, 3),
# #                  Stride=(1, 1)):
# #         self.Name = 'conv2d_%s' % name
# #         self.InputShape = {"x": InputShape[0], "y": InputShape[1], "z": InputShape[2]}
# #         self.OutputShape = {"x": OutputShape[0], "y": OutputShape[1], "z": OutputShape[2]}
# #         self.WeightShape = {"x": WeightShape[0], "y": WeightShape[1], "z": WeightShape[2], "w": WeightShape[3]}
# #         self.weightcache = weightcache
# #         self.Filters = Filters
# #         self.KernalSize = {"x": KernalSize[0], "y": KernalSize[1]}
# #         self.Stride = {"x": Stride[0], "y": Stride[1]}
# #
# #
# # class batchnormLayer(object):
# #     def __init__(self,
# #                  name,
# #                  InputShape=(0, 0, 0),
# #                  OutputShape=(0, 0, 0),
# #                  WeightShape=(16.0, 0.0, 0.0, 0.0),
# #                  weightcache=None):
# #         self.Name = name
# #         self.InputShape = {"x": InputShape[0], "y": InputShape[1], "z": InputShape[2]}
# #         self.OutputShape = {"x": OutputShape[0], "y": OutputShape[1], "z": OutputShape[2]}
# #         self.WeightShape = {"x": WeightShape[0], "y": WeightShape[1], "z": WeightShape[2], "w": WeightShape[3]}
# #         self.weightcache = weightcache
# #
# #
# # class activationLayer(object):
# #     def __init__(self,
# #                  name,
# #                  InputShape=(0, 0, 0),
# #                  OutputShape=(0, 0, 0),
# #                  WeightShape=(0.0, 0.0, 0.0, 0.0)):
# #         self.Name = 'activation_%s' % name
# #         self.InputShape = {"x": InputShape[0], "y": InputShape[1], "z": InputShape[2]}
# #         self.OutputShape = {"x": OutputShape[0], "y": OutputShape[1], "z": OutputShape[2]}
# #         self.WeightShape = {"x": WeightShape[0], "y": WeightShape[1], "z": WeightShape[2], "w": WeightShape[3]}
# #
# #
# # class addLayer(object):
# #     def __init__(self,
# #                  name,
# #                  InputShape=(0, 0, 0),
# #                  OutputShape=(0, 0, 0),
# #                  WeightShape=(0.0, 0.0, 0.0, 0.0),
# #                  AlternativeInputId=None):
# #         self.Name = 'add_%s' % name
# #         self.InputShape = {"x": InputShape[0], "y": InputShape[1], "z": InputShape[2]}
# #         self.OutputShape = {"x": OutputShape[0], "y": OutputShape[1], "z": OutputShape[2]}
# #         self.WeightShape = {"x": WeightShape[0], "y": WeightShape[1], "z": WeightShape[2], "w": WeightShape[3]}
# #         self.AlternativeInputId = AlternativeInputId
# #
# #
# # class upsample2dLayer(object):
# #     def __init__(self,
# #                  name,
# #                  InputShape=(0, 0, 0),
# #                  OutputShape=(0, 0, 0),
# #                  WeightShape=(0.0, 0.0, 0.0, 0.0),
# #                  Size=(2, 2)):
# #         self.Name = 'up_sampling2d_%s' % name
# #         self.InputShape = {"x": InputShape[0], "y": InputShape[1], "z": InputShape[2]}
# #         self.OutputShape = {"x": OutputShape[0], "y": OutputShape[1], "z": OutputShape[2]}
# #         self.WeightShape = {"x": WeightShape[0], "y": WeightShape[1], "z": WeightShape[2], "w": WeightShape[3]}
# #         self.Size = {"x": Size[0], "y": Size[1]}
# #
# #
# # class maxpoolingLayer(object):
# #     def __init__(self,
# #                  name,
# #                  InputShape=(0, 0, 0),
# #                  OutputShape=(0, 0, 0),
# #                  WeightShape=(0.0, 0.0, 0.0, 0.0),
# #                  Size=(2, 2)):
# #         self.Name = 'maxpooling_%s' % name
# #         self.InputShape = {"x": InputShape[0], "y": InputShape[1], "z": InputShape[2]}
# #         self.OutputShape = {"x": OutputShape[0], "y": OutputShape[1], "z": OutputShape[2]}
# #         self.WeightShape = {"x": WeightShape[0], "y": WeightShape[1], "z": WeightShape[2], "w": WeightShape[3]}
# #         self.Size = {"x": Size[0], "y": Size[1]}
# #
# #
# # class concatLayer(object):
# #     def __init__(self,
# #                  name,
# #                  InputShape=(0, 0, 0),
# #                  OutputShape=(0, 0, 0),
# #                  WeightShape=(0.0, 0.0, 0.0, 0.0),
# #                  AlternativeInputId=None):
# #         self.Name = 'concat_%s' % name
# #         self.InputShape = {"x": InputShape[0], "y": InputShape[1], "z": InputShape[2]}
# #         self.OutputShape = {"x": OutputShape[0], "y": OutputShape[1], "z": OutputShape[2]}
# #         self.WeightShape = {"x": WeightShape[0], "y": WeightShape[1], "z": WeightShape[2], "w": WeightShape[3]}
# #         self.AlternativeInputId = AlternativeInputId
# #
# # # xx = conv2dLayer('111', weightcache=[i for i in range(32)])
# # # bn = batchnorm('bnnn', myInputShape, myOutputShape, myWeightShape)
# # import json
# # # d1 = json.dumps(xx, default=lambda obj:obj.__dict__, indent=1)
# # # print(d1)
# #
# #
# # model_name = 'net_params_30_2019-07-27_14-22.pkl'
# # train_loader = None
# # from config import config
# #
# # mlp = NANO_cnn(train_loader, config)
# #
# # # from torch
# # # g = mlp.generator.to(device)
# # xxx = torch.load('Checkpoints\\Unet+wgan\\%s' % model_name)
# # xx = [(k, v) for (k, v) in torch.load('Checkpoints\\Unet+wgan\\%s' % model_name).items() if 'generator' in k and 'num' not in k]
# # # mlp.load_state_dict(torch.load('Checkpoints\\Unet+wgan\\%s' % model_name))
# # LayerTypes_ = [i[0] for i in xx]
# # LayerJson_ = [i[1] for i in xx]
# #
# # inc = [(k, v) for (k, v) in xx if 'inc' in k]
# # inc_conv = [j for i, j in enumerate(inc) if i % 6 < 2]
# # inc_bn = [j for i, j in enumerate(inc) if i % 6 >= 2]
# #
# # up = [(k, v) for (k, v) in xx if 'up' in k]
# # up_conv = [j for i, j in enumerate(up) if i % 6 < 2]
# # up_bn = [j for i, j in enumerate(up) if i % 6 >= 2]
# #
# # down = [(k, v) for (k, v) in xx if 'down' in k]
# # down_conv = [j for i, j in enumerate(down) if i % 6 < 2]
# # down_bn = [j for i, j in enumerate(down) if i % 6 >= 2]
# #
# # out = [(k, v) for (k, v) in xx if 'out' in k]
# #
# # print()
