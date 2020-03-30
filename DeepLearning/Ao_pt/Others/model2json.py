import numpy as np
import torch
import json


class InputLayer(object):
    def __init__(self,
                 name='input_normalize_1',
                 input_shape=(0, 0, 0),
                 output_shape=(0, 0, 0),
                 weight_shape=(0.0, 0.0, 0.0, 0.0),
                 input_channels=4,
                 src=0):
        self.Name = name
        self.InputShape = {"x": input_shape[0], "y": input_shape[1], "z": input_shape[2]}
        self.OutputShape = {"x": output_shape[0], "y": output_shape[1], "z": output_shape[2]}
        self.WeightShape = {"x": weight_shape[0], "y": weight_shape[1], "z": weight_shape[2], "w": weight_shape[3]}
        self.InputChannels = input_channels
        self.src = {"instanceID": src}


class Conv2dLayer(object):
    def __init__(self,
                 name,
                 input_shape=(0, 0, 0),
                 output_shape=(0, 0, 0),
                 weight_shape=(3.0, 3.0, 4.0, 4.0),
                 weight_cache=None,
                 filters=3.0,
                 kernal_size=(3, 3),
                 stride=(1, 1)):
        self.Name = 'conv2d_%s' % name
        self.InputShape = {"x": input_shape[0], "y": input_shape[1], "z": input_shape[2]}
        self.OutputShape = {"x": output_shape[0], "y": output_shape[1], "z": output_shape[2]}
        self.WeightShape = {"x": weight_shape[0], "y": weight_shape[1], "z": weight_shape[2], "w": weight_shape[3]}
        print(len(weight_cache))
        self.weightcache = weight_cache
        self.Filters = filters
        self.KernalSize = {"x": kernal_size[0], "y": kernal_size[1]}
        self.Stride = {"x": stride[0], "y": stride[1]}


class BatchNormLayer(object):
    def __init__(self,
                 name,
                 input_shape=(0, 0, 0),
                 output_shape=(0, 0, 0),
                 weight_shape=16.0,
                 weight_cache=None):
        self.Name = 'batch_norm2d_%s' % name
        self.InputShape = {"x": input_shape[0], "y": input_shape[1], "z": input_shape[2]}
        self.OutputShape = {"x": output_shape[0], "y": output_shape[1], "z": output_shape[2]}
        self.WeightShape = {"x": weight_shape, "y": 0.0, "z": 0.0, "w": 0.0}
        self.weightcache = weight_cache


class ActivationLayer(object):
    def __init__(self,
                 name,
                 input_shape=(0, 0, 0),
                 output_shape=(0, 0, 0),
                 weight_shape=(0.0, 0.0, 0.0, 0.0)):
        self.Name = 'activation_%s' % name
        self.InputShape = {"x": input_shape[0], "y": input_shape[1], "z": input_shape[2]}
        self.OutputShape = {"x": output_shape[0], "y": output_shape[1], "z": output_shape[2]}
        self.WeightShape = {"x": weight_shape[0], "y": weight_shape[1], "z": weight_shape[2], "w": weight_shape[3]}


class LeakyReluLayer(object):
    def __init__(self,
                 name,
                 input_shape=(0, 0, 0),
                 output_shape=(0, 0, 0),
                 weight_shape=(0.0, 0.0, 0.0, 0.0),
                 alpha=0.01):
        self.Name = 'activation_%s' % name
        self.InputShape = {"x": input_shape[0], "y": input_shape[1], "z": input_shape[2]}
        self.OutputShape = {"x": output_shape[0], "y": output_shape[1], "z": output_shape[2]}
        self.WeightShape = {"x": weight_shape[0], "y": weight_shape[1], "z": weight_shape[2], "w": weight_shape[3]}
        self.Alpha = alpha


class AddLayer(object):
    def __init__(self,
                 name,
                 input_shape=(0, 0, 0),
                 output_shape=(0, 0, 0),
                 weight_shape=(0.0, 0.0, 0.0, 0.0),
                 alternative_input_id=None):
        self.Name = 'add_%s' % name
        self.InputShape = {"x": input_shape[0], "y": input_shape[1], "z": input_shape[2]}
        self.OutputShape = {"x": output_shape[0], "y": output_shape[1], "z": output_shape[2]}
        self.WeightShape = {"x": weight_shape[0], "y": weight_shape[1], "z": weight_shape[2], "w": weight_shape[3]}
        self.AlternativeInputId = alternative_input_id


class UpSample2dLayer(object):
    def __init__(self,
                 name,
                 input_shape=(0, 0, 0),
                 output_shape=(0, 0, 0),
                 weight_shape=(0.0, 0.0, 0.0, 0.0),
                 size=(2, 2),
                 alternative_input_id=-1):
        self.Name = 'up_sampling2d_%s' % name
        self.InputShape = {"x": input_shape[0], "y": input_shape[1], "z": input_shape[2]}
        self.OutputShape = {"x": output_shape[0], "y": output_shape[1], "z": output_shape[2]}
        self.WeightShape = {"x": weight_shape[0], "y": weight_shape[1], "z": weight_shape[2], "w": weight_shape[3]}
        self.Size = {"x": size[0], "y": size[1]}
        self.AlternativeInputId = alternative_input_id


class MaxPoolingLayer(object):
    def __init__(self,
                 name,
                 input_shape=(0, 0, 0),
                 output_shape=(0, 0, 0),
                 weight_shape=(0.0, 0.0, 0.0, 0.0),
                 size=(2, 2)):
        self.Name = 'max_pooling_%s' % name
        self.InputShape = {"x": input_shape[0], "y": input_shape[1], "z": input_shape[2]}
        self.OutputShape = {"x": output_shape[0], "y": output_shape[1], "z": output_shape[2]}
        self.WeightShape = {"x": weight_shape[0], "y": weight_shape[1], "z": weight_shape[2], "w": weight_shape[3]}
        self.Size = {"x": size[0], "y": size[1]}


class ConcatLayer(object):
    def __init__(self,
                 name,
                 input_shape=(0, 0, 0),
                 output_shape=(0, 0, 0),
                 weight_shape=(0.0, 0.0, 0.0, 0.0),
                 alternative_input_id=None):
        self.Name = 'concat_%s' % name
        self.InputShape = {"x": input_shape[0], "y": input_shape[1], "z": input_shape[2]}
        self.OutputShape = {"x": output_shape[0], "y": output_shape[1], "z": output_shape[2]}
        self.WeightShape = {"x": weight_shape[0], "y": weight_shape[1], "z": weight_shape[2], "w": weight_shape[3]}
        self.AlternativeInputId = alternative_input_id


def deal_conv_dict(conv_dict, name=''):
    weight = np.array([v for k, v in conv_dict.items() if 'weight' in k][0].cpu())

    weight = weight.transpose((2, 3, 1, 0))
    shape = [float(i) for i in list(weight.shape)]
    print(shape)
    weight = np.reshape(weight, -1)
    bias = np.array([v for k, v in conv_dict.items() if 'bias' in k][0].cpu())

    weightcache = np.concatenate((weight, bias))
    # weightcache = np.ones(shape=(weightcache.shape[0]), dtype=np.float)
    weightcache = [float(str(i)) for i in list(weightcache)]
    return Conv2dLayer(name=name, weight_shape=shape, weight_cache=weightcache, filters=shape[-1])


def deal_bn_dict(conv_dict, name=''):

    weight = [v for k, v in conv_dict.items() if 'weight' in k][0].cpu()
    bias = [v for k, v in conv_dict.items() if 'bias' in k][0].cpu()
    run_mean = [v for k, v in conv_dict.items() if 'running_mean' in k][0].cpu()
    run_var = [v for k, v in conv_dict.items() if 'running_var' in k][0].cpu()
    weightcache = np.vstack((weight, bias, run_mean, run_var)).transpose()
    weightcache = np.reshape(weightcache, -1)
    weightcache = [float(str(i)) for i in list(weightcache)]
    shape = weight.shape[0]

    return BatchNormLayer(name=name, weight_shape=float(shape), weight_cache=weightcache)


def deal_conv_block(block, name='', bn=True, act_type='leky'):

    if bn is True:
        conv_layer_num = len(block) // 6
        count = 0
        for crr_layer in range(conv_layer_num):
            conv = {k: v for k, v in block.items() if 'conv.%s' % (crr_layer+count) in k}
            bn_1 = {k: v for k, v in block.items() if 'conv.%s' % (crr_layer+1+count) in k}
            model_lst.append([LayerTypes['conv'], deal_conv_dict(conv, '%s_%s' % (name, crr_layer))])
            down_lst.append(len(model_lst) - 1)
            model_lst.append([LayerTypes['bn'], deal_bn_dict(bn_1, '%s_%s' % (name, crr_layer))])
            # if act_type is not None:
            #     model_lst.append([LayerTypes['leky'], ActivationLayer(name='%s_%s' % (name, crr_layer))])
            count += 1

    else:
        conv_layer_num = len(block) // 2
        count = 0
        for crr_layer in range(conv_layer_num):
            conv = {k: v for k, v in block.items() if 'conv.%s' % (crr_layer + count) in k}
            model_lst.append([LayerTypes['conv'], deal_conv_dict(conv, '%s_%s' % (name, crr_layer))])
            down_lst.append(len(model_lst) - 1)
            # if act_type is not None:
            #     model_lst.append([LayerTypes['leky'], ActivationLayer(name='%s_%s' % (name, crr_layer))])
            count += 1


LayerTypes = {'input': "NNPP.InputLayer", 'conv': "NNPP.Conv2D", 'bn': "NNPP.BatchNormalization",
              'leky': "NNPP.LeakyReLU", 'cat': "NNPP.Concatenate", 'maxpool': "NNPP.MaxPooling2D",
              'up': 'NNPP.UpSampling2D'}

model_name = 'net_params_22_2019-08-10_20-29.pkl'

pretrain = torch.load('Checkpoints\\test\\%s' % model_name)
pretrained_dict = {k[10:]: v for k, v in pretrain.items() if 'dis' not in k and 'num' not in k}

model_lst = []
cat_lst = []
down_lst = []

model_lst.append([LayerTypes['input'], InputLayer()])

inc = {k: v for k, v in pretrained_dict.items() if 'inc' in k}
deal_conv_block(inc, '1')
cat_lst.append(len(model_lst)-1)

for i in range(1, 5):
    down = {k[13:]: v for k, v in pretrained_dict.items() if 'down%s' % i in k}
    model_lst.append([LayerTypes['maxpool'], MaxPoolingLayer(name='maxpooling2d_%s' % i)])
    deal_conv_block(down, name='%s' % (i+1))
    if i < 4:
        cat_lst.append(len(model_lst)-1)

down_lst = down_lst[:-1]

for i in range(1, 5):
    up = {k: v for k, v in pretrained_dict.items() if 'up%s' % i in k}
    model_lst.append([LayerTypes['up'], UpSample2dLayer(name='up_sampling2d_%s' % i,
                                                        alternative_input_id=down_lst[4-i])])
    model_lst.append([LayerTypes['cat'],
                      ConcatLayer(name='concatenate_%s' % i, alternative_input_id=cat_lst[4-i])])
    deal_conv_block(up, name='%s' % (i+5))

out = {k: v for k, v in pretrained_dict.items() if 'out' in k}
model_lst.append([LayerTypes['conv'], deal_conv_dict(out, name='output_conv')])


model_type = [i[0] for i in model_lst]
model_json = [json.dumps(i[1], default=lambda obj: obj.__dict__, indent=1) for i in model_lst]

out = {'LayerTypes': model_type, 'LayerJson': model_json}
f = open('Checkpoints\\test\\model.json', mode='w', encoding='utf-8')
json.dump(out, f, ensure_ascii=False)
f.close()
print()
