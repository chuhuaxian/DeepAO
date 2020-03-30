import os
from dataloaders import MyAoDataset, MyAoDataset_DS
from Models.NANO import NANO, NANO_cnn, NANO_cnn_ts
from torch.utils.data import DataLoader
from config import config
import h5py
import numpy as np
from torch.utils.data import random_split

CHECKPOINT_DIR = './Checkpoints/NNAO_cnn'

# Loading datasets
print("\nLoading DATA...")


DATA_DIR = config.DATAPATH
epochs = config.TRAIN_EPOCH
learning_rate = config.learning_rate

# datapath = os.path.join(DATA_DIR, "dataset_full.npy")

# data = np.load(datapath)
# f = h5py.File(datapath, "r")


# N, config.width, config.height = f['X'].shape[:3]
# print(N)
# Ymean, Ystd = np.array(f['Ymean']), np.array(f['Ystd'])
# Xmean, Xstd = np.array(f['Xmean']), np.array(f['Xstd'])
f = open('..\\Datasets\\train_lst.txt', mode='r')
train_lst = [i.strip().split(',') for i in f.readlines() if 'scene' not in i]

f = open('..\\Datasets\\test_lst.txt', mode='r')
test_lst = [[j for j in i.strip().split(',')] for i in f.readlines()]

# val_lst = [[os.path.join('Datasets\\test', i)] for i in os.listdir('Datasets\\test') if '20017' in i]

train_dataset = MyAoDataset_DS(train_lst)
test_dataset = MyAoDataset_DS(test_lst, mode='test')
# val_dataset = MyAoDataset_DS(val_lst)


train_loader = DataLoader(dataset=train_dataset, batch_size=config.bs,  shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.bs, shuffle=True)
# val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

mlp = NANO_cnn_ts(train_loader, config, ds_test=test_loader).cuda()
mlp.fit_ts(epochs, learning_rate, tn_path='Unet+wgan\\net_params_30_2019-07-27_14-22.pkl')

# model_name = 'Unet+wgan\\net_params_30_2019-07-27_14-22.pkl'
# mlp.predict_gan(model_name=model_name)
print('-------------finished!-------------')

