from dataloaders import MyAoDataset_DS
from Models.NANO import NANO_cnn
from torch.utils.data import DataLoader
from config import config
from generate_train_file_lst import generate_file_list
import os

DATA_DIR = config.DATAPATH
epochs = config.TRAIN_EPOCH
learning_rate = config.learning_rate

if not os.path.exists(os.path.join('Logs\\result', config.save_name)):
    os.makedirs(os.path.join('Logs\\result', config.save_name))

if not os.path.exists(os.path.join('Checkpoints', config.save_name)):
    os.makedirs(os.path.join('Checkpoints', config.save_name))

if not os.path.exists('train_lst.txt') or not os.path.exists('test_lst.txt'):
    print('begin generate train and test file list')
    generate_file_list(config.DATAPATH)

print('load  train and test file list')
f = open('train_lst.txt', mode='r')
train_lst = [i.strip().split(',') for i in f.readlines() if 'scene_07_Back_View06' not in i]

f = open('test_lst.txt', mode='r')
test_lst = [[j for j in i.strip().split(',')] for i in f.readlines() if 'scene_07_Back_View06' not in i]


train_dataset = MyAoDataset_DS(train_lst)
test_dataset = MyAoDataset_DS(test_lst)


train_loader = DataLoader(dataset=train_dataset, batch_size=config.bs,  shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.bs, shuffle=True)


mlp = NANO_cnn(train_loader, config, ds_test=test_loader).cuda()
mlp.fit(epochs, learning_rate)

print('-------------finished!-------------')

