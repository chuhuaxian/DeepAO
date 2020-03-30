from dataloaders import MyAoDataset_LTSM
from Models.NANO import NANO_LSTM
from torch.utils.data import DataLoader
from config import config
from generate_train_file_lst import generate_deepshading_file_list
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
    generate_deepshading_file_list(config.DATAPATH)

print('load  train and test file list')
f = open('train_lst.txt', mode='r')
train_lst = [i.strip().split(',') for i in f.readlines()]
# train_lst = []

f = open('test_lst.txt', mode='r')
test_lst = [i.strip().split(',') for i in f.readlines()]


train_dataset = MyAoDataset_LTSM(train_lst)
test_dataset = MyAoDataset_LTSM(test_lst)


train_loader = DataLoader(dataset=train_dataset, batch_size=config.bs,  shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.bs, shuffle=True)


mlp = NANO_LSTM(train_loader, config, ds_test=test_loader).cuda()
mlp.fit(epochs, learning_rate)

print('-------------finished!-------------')

