from dataloaders import MyAoDataset_DS
from Models.NANO import NANO_cnn
from torch.utils.data import DataLoader
from config import config
import os
from generate_train_file_lst import generate_deepshading_file_list
CHECKPOINT_DIR = './Checkpoints/NNAO_cnn'

DATA_DIR = config.DATAPATH

# if not os.path.exists('Filelists\\evaluate\\val_lst.txt'):
#     print('begin generate evaluate file list')
generate_deepshading_file_list(config.DATAPATH)

print('load  evaluate file list')
f = open('test_lst.txt', mode='r')
val_lst = [[j for j in i.strip().split(',')] for i in f.readlines()]


val_dataset = MyAoDataset_DS(val_lst,  model=config.model, mode='Test')
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

train_loader = None
mlp = NANO_cnn(train_loader, config, ds_test=val_loader).cuda()

mlp.predict(model_name=config.model_name, model=config.model)
print('-------------finished!-------------')

