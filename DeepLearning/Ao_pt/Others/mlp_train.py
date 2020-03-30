import os
from dataloaders import  MyAoDataset_full
from Models.NANO import  NANO_cnn
from torch.utils.data import DataLoader
from config import config
import numpy as np
from torch.utils.data import random_split

CHECKPOINT_DIR = './Checkpoints/NNAO_cnn'

# Loading datasets
# print("\nLoading DATA...")


DATA_DIR = config.DATAPATH
epochs = config.TRAIN_EPOCH
learning_rate = config.learning_rate

datapath = os.path.join(DATA_DIR, "dataset_full.npy")

data = np.load(datapath)


full_dataset = MyAoDataset_full(data)

print(len(full_dataset))
train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=config.bs,  shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.bs, shuffle=True)

mlp = NANO_cnn(train_loader, config, ds_test=test_loader).cuda()


mlp.fit(epochs, learning_rate)

print('-------------finished!-------------')

