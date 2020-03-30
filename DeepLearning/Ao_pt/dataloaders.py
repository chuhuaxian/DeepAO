from torch.utils.data import Dataset
import numpy as np
import torch
import pyexr
import matplotlib.pyplot as plt


camera_cx = 256.0
camera_cy = 256.0
camera_fx = 548.9937
camera_fy = 548.9937


row_map = np.ones(shape=(512, 512), dtype=np.float)*np.arange(0, 512)
col_map = np.transpose(row_map)
row_map = np.expand_dims(row_map, axis=-1)-camera_cx
col_map = np.expand_dims(col_map, axis=-1) - camera_cy


def depth2position(depth_map):
    depth_map = 1 - depth_map
    xx = 0.5+np.multiply(row_map, depth_map) / camera_fx
    yy = 0.35-np.multiply(col_map, depth_map) / camera_fy
    return np.concatenate([yy, xx, depth_map], axis=-1)


class MyAoDataset(Dataset):
    # 读取存储image路径的txt文件
    def __init__(self, one_batch, x_mean, x_var, y_mean, y_var):
        self.one_batch = one_batch
        self.x_mean = x_mean
        self.x_var = x_var
        self.y_mean = y_mean
        self.y_var = y_var

    # 读取存储image路径的txt文件
    def __getitem__(self, index):
        input = np.array(self.one_batch['X'][index])
        label = np.array(self.one_batch['Y'][index])

        input = (input - self.x_mean) / self.x_var
        label = (label - self.y_mean) / self.y_var
        input = np.reshape(input, (-1, ))

        return torch.from_numpy(input),  torch.from_numpy(label)  # 最后一定要return tensor类型不然会报错

    def __len__(self):
        return len(self.one_batch['Y'])


class MyAoDataset_full(Dataset):
    # 读取存储image路径的txt文件
    def __init__(self, one_batch):
        self.one_batch = one_batch

    # 读取存储image路径的txt文件
    def __getitem__(self, index):
        input = self.one_batch[index, :, :, :4]
        label = self.one_batch[index, :, :, 4]
        input = np.transpose(input, (2, 0, 1))

        return torch.from_numpy(input),  torch.from_numpy(label)  # 最后一定要return tensor类型不然会报错

    def __len__(self):
        return len(self.one_batch)


class MyAoDataset_DS(Dataset):
    # 读取存储image路径的txt文件
    def __init__(self, one_batch, model='ours', mode='Train'):
        self.one_batch = one_batch
        self.model = model
        self.mode = mode
    # 读取存储image路径的txt文件
    def __getitem__(self, index):

        if self.model == 'ours':
            if self.mode == 'Train':
                depth = pyexr.open(self.one_batch[index][2]).get()[:, :, 2:3]
            else:
                depth = 1 + pyexr.open(self.one_batch[index][2]).get()[:, :, 2:3] / 1500.0
            # depth /=5
        else:
            depth = pyexr.open(self.one_batch[index][2]).get()
            xx = (750 + depth[:, :, 0:1]) / 1500.0
            yy = (750 + depth[:, :, 1:2]) / 1500.0
            zz = (1500 + depth[:, :, 2:3]) / 1500.0
            #
            depth = np.concatenate([xx, yy, zz], axis=-1)
            depth /= 3

        # print('depth', np.min(depth))
        # print(depth.shape)
        # plt.imshow(depth[:, :, :3])
        # plt.show()
        # depth *= 1.49
        # depth = depth2position(depth)

        # plt.subplot(131)
        # plt.imshow(xx)
        # plt.subplot(132)
        # plt.imshow(yy)
        # plt.subplot(133)
        # plt.imshow(zz)
        # plt.show()
        # depth = np.concatenate([xx, yy, zz], axis=-1)
        # depth /=3
        # print(np.max(depth), np.min(depth), np.mean(depth))
        # depth = (depth-np.min(depth))/(np.max(depth)-np.min(depth))
        # depth = depth/1.8
        # if np.max(depth) >1:
        #     depth -= 1
        normal = pyexr.open(self.one_batch[index][1]).get()[:, :, :3]

        label = pyexr.open(self.one_batch[index][0]).get()[:, :, 0:1]

        input = np.transpose(np.concatenate((normal, depth), axis=-1), (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))

        # print(np.max(normal), np.min(normal))
        # plt.imshow(np.squeeze(depth))skkkkkkkkk75752
        # plt.show()

        return torch.from_numpy(input),  torch.from_numpy(label)  # 最后一定要return tensor类型不然会报错

        # return torch.from_numpy(np.expand_dims(input, axis=0)), torch.from_numpy(
        #     np.expand_dims(label, axis=0))  # 最后一定要return tensor类型不然会报错

    def __len__(self):
        return len(self.one_batch)


class MyAoDataset_LTSM(Dataset):
    # 读取存储image路径的txt文件
    def __init__(self, one_batch,):
        self.one_batch = one_batch

    # 读取存储image路径的txt文件
    def __getitem__(self, index):

        inputs = []
        labels = []
        for i in range(4):
            depth = pyexr.open(self.one_batch[index][i*3+2]).get()[:, :, 2:3]
            normal = pyexr.open(self.one_batch[index][i*3+1]).get()[:, :, :3]
            label = pyexr.open(self.one_batch[index][i*3+0]).get()[:, :, 0:1]
            input = np.transpose(np.concatenate((normal, depth), axis=-1), (2, 0, 1))
            label = np.transpose(label, (2, 0, 1))
            inputs.append(np.expand_dims(input, axis=0))
            labels.append(label)
        inputs = np.concatenate(inputs, axis=0)
        labels = np.concatenate(labels, axis=0)

        return torch.from_numpy(inputs),  torch.from_numpy(labels)  # 最后一定要return tensor类型不然会报错

    def __len__(self):
        return len(self.one_batch)
