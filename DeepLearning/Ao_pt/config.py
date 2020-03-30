from easydict import EasyDict as edict

config = edict()

# config.DATAPATH = 'D:\\Projects\\Datasets\\train'
config.DATAPATH = 'D:\Projects\Datasets\\3dsmax\\evaluate'
# config.DATAPATH = 'D:\\Projects\Datasets\\3dsmax\scene'
# config.DATAPATH = 'D:\Projects\Datasets\\3dsmax\\evaluate\\deepshading\Test'

config.VALDATAPATH = 'Datastes\\Unity_sceenshot\\test1'
config.train_save_dir = 'Logs\\result'
# config.save_name = '9-1-Unet-4'
config.save_name = '12-22-Unet-4'


# Settings
config.dropout = 0.5
config.learning_rate = 1e-3
config.TRAIN_EPOCH = 100000
config.save_interval = 1

# config.fc_lst = [128, 64, config.n_classes]

# config.pooling_ratio = 0.8
config.load_model = False
# config.load_model = True
config.model = 'ours'
# config.model = 'deepshading'
# 'Unet+wgan\\net_params_30_2019-07-27_14-22.pkl'
if config.model == 'ours':
    # config.model_name = '10-18-LSTM-8\\net_params_18_2019-11-21_12-19.pkl'
    # config.model_name = '12-10-Unet-4\\net_params_16_2019-12-12_01-57.pkl'
    # config.model_name = '12-15-Unet-4\\net_params_8_2019-12-17_12-44.pkl'
    config.model_name = '12-20-Unet-4\\net_params_12_2019-12-22_16-30.pkl'

else:
    config.model_name = 'DeepShading\\net_params_60_2019-11-21_14-09.pkl'
# Hyperparameters
config.epochs = 100000
bs = 2
config.bs = bs

# config.decay_every = int(665600/bs*20)
# print()

config.clip_value = 0.01
config.n_critic = 1