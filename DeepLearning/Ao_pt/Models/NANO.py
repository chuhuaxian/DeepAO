import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parameter
import numpy as np
import cv2
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from unet import UNet, Discriminator, UNet_multi_scale
from Models import ConvLSTM
from unet import UNet_student
from unet import DSUNet

# from Myloss import MY_SSIM
import torch.autograd as autograd
import matplotlib.pyplot as plt
import math
from torch.optim.optimizer import Optimizer
import datetime
from pytorch_msssim import MS_SSIM, SSIM

# This version of Adam keeps an fp32 copy of the parameters and
# does all of the parameter updates in fp32, while still doing the
# forwards and backwards passes using fp16 (i.e. fp16 copies of the
# parameters and fp16 activations).
#
# Note that this calls .float().cuda() on the params such that it
# moves them to gpu 0--if you're using a different GPU or want to
# do multi-GPU you may need to deal with this.


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                            N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class Adam16(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        params = list(params)
        super(Adam16, self).__init__(params, defaults)
        # for group in self.param_groups:
        # for p in group['params']:

        self.fp32_param_groups = [p.data.float().cuda() for p in params]
        if not isinstance(self.fp32_param_groups[0], dict):
            self.fp32_param_groups = [{'params': self.fp32_param_groups}]

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, fp32_group in zip(self.param_groups, self.fp32_param_groups):
            for p, fp32_p in zip(group['params'], fp32_group['params']):
                if p.grad is None:
                    continue

                grad = p.grad.data.float()
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], fp32_p)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # print(type(fp32_p))
                fp32_p.addcdiv_(-step_size, exp_avg, denom)
                p.data = fp32_p.half()

        return loss


class RMSprop16(Optimizer):
    """Implements RMSprop algorithm.

    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-4, weight_decay=0, momentum=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        params = list(params)
        super(RMSprop16, self).__init__(params, defaults)

        self.fp32_param_groups = [p.data.float().cuda() for p in params]
        if not isinstance(self.fp32_param_groups[0], dict):
            self.fp32_param_groups = [{'params': self.fp32_param_groups}]

    def __setstate__(self, state):
        super(RMSprop16, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, fp32_group in zip(self.param_groups, self.fp32_param_groups):
            for p, f32_p in zip(group['params'], fp32_group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = grad.new().resize_as_(grad).zero_()
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = grad.new().resize_as_(grad).zero_()
                    if group['centered']:
                        state['grad_avg'] = grad.new().resize_as_(grad).zero_()

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], f32_p)
                    # grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    f32_p.add_(-group['lr'], buf)
                else:
                    f32_p.addcdiv_(-group['lr'], grad, avg)
                p.data = f32_p.half()

        return loss


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, rng=np.random):
        super(MyLinear, self).__init__()

        W_bound = np.sqrt(6. / (in_features*out_features))
        init_weight = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(in_features, out_features)), dtype=np.float32)
        self.weight = nn.Parameter(torch.from_numpy(init_weight))

        init_beta = torch.from_numpy(np.zeros((out_features, ), dtype=np.float32))
        self.bias = nn.Parameter(init_beta)

    def forward(self, x):
        return torch.mm(x, self.weight)+self.bias


class MyPrelu(nn.Module):
    def __init__(self, num_parameters, init_alpha=0.25, init_beta=1.0):
        super(MyPrelu, self).__init__()

        self.alpha = nn.Parameter(torch.Tensor(num_parameters).fill_(init_alpha))
        self.beta = nn.Parameter(torch.Tensor(num_parameters).fill_(init_beta))

    def forward(self, x):
        neg = x < 0
        pos = x > 0

        return x*neg.float()*self.alpha + x*pos.float()*self.beta


class NANO(nn.Module):
    def __init__(self, ds_train, config, ds_test=None):
        super(NANO, self).__init__()

        self.ds_train = ds_train
        self.ds_test = ds_test
        self.config = config
        self.rng = np.random.RandomState(123)

        self.fcs = torch.nn.Sequential(
            torch.nn.Dropout(config.dropout),  # drop 50% of the neuron
            MyLinear(4*31*31, 4, rng=self.rng),
            MyPrelu(4),

            MyLinear(4, 4, rng=self.rng),
            MyPrelu(4),

            MyLinear(4, 4, rng=self.rng),
            MyPrelu(4),

            MyLinear(4, 1, rng=self.rng),
            MyPrelu(1),
        )

    def forward(self, x):

        res = self.fcs(x)
        return res

    def fit(self, epoches, lr):

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)


        # self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.decay_every, gamma=0.5)

        self.loss = nn.MSELoss().cuda()

        if self.config.load_model and self.config.model_name != '':
            self.load_state_dict(torch.load('Checkpoints\\%s' % self.config.model_name))
            print('load success')

        # writer = SummaryWriter()

        for epoch in range(epoches):

            if epoch % self.config.save_interval == 0:
                torch.save(self.state_dict(), 'Checkpoints\\net_params_%s_%s.pkl' % (
                    epoch, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))

            # scheduler.step()
            # Train
            loss_train, loss_train_exp, acc_train, = 0., 0., 0.
            count_train = 0

            for i, j in self.ds_train:
                self.train()
                self.optimizer.zero_grad()

                count_train += 1
                inputs = Variable(i.float()).cuda()
                labels = Variable(j.float()).cuda()

                outputs = self.forward(inputs)

                loss = self.loss(outputs, labels)
                loss_train += loss.item()

                l1_regularization = 0

                params = [i for i in self.named_parameters() if len(i[1].shape) == 2]

                for param in params:
                    l1_regularization += torch.mean(torch.abs(param[1]))

                loss += 0.01*l1_regularization

                loss.backward()

                self.optimizer.step()

                # loss_train += loss.item()

            print('######################################  TRAIN  ######################################')
            print('Train: epoch = %s, loss = %s' % (epoch, loss_train / count_train))

            # evaluate
            count_test = 0
            loss_test = 0.
            with torch.no_grad():
                for i, j in self.ds_test:

                    self.eval()
                    count_test += 1
                    inputs = Variable(i.float()).cuda()
                    labels = Variable(j.float()).cuda()

                    outputs = self.forward(inputs)
                    loss = self.loss(outputs, labels)

                    loss_test += loss.item()
                    # print(loss.item())
                    # print('Epoch: %d, Test: %.4f' % (epoch, loss.item()))

            print('######################################  TEST  ######################################')
            print('Test: epoch = %s, loss = %s' % (epoch, loss_test / count_test))


class MyConvolutional(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=True, padding_mode='full'):
        super(MyConvolutional, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight.data)
        nn.init.kaiming_normal_(self.bias.data)

    def forward(self, x):
        # inp = torch.randn(1, 3, 10, 12)
        # w = torch.randn(2, 3, 4, 5)
        # inp_unf = torch.nn.functional.unfold(inp, (4, 5))
        # out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        # out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))

        res = F.unfold(x, self.kernel_size, self.dilation, self.padding, self.stride)
        res = res.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()).transpose(1, 2)
        # out = torch.nn.functional.fold(res, (, 8), (1, 1))
        # res = torch.conv2d(x, weight=self.weight, bias=self.bias, stride=self.kernel_size, padding=self.padding)
        return res


class NANO_cnn_ts(nn.Module):
    def __init__(self, ds_train, config, ds_test=None, ds_val=None):
        super(NANO_cnn_ts, self).__init__()

        self.ds_train = ds_train
        self.ds_test = ds_test
        self.ds_val = ds_val
        self.config = config
        self.tnet = UNet(4, 1).cuda()
        self.generator = UNet_student(4, 1).cuda()
        # self.discriminator = Discriminator().cuda()

    # def fit_ts(self, epoches, lr=1e-3, tn_path=''):
    #     # lr:5e-5
    #     # xx = [i for i in  self.generator.parameters()]
    #     print('# generator parameters:', sum(param.numel() for param in self.generator.parameters()))
    #
    #     self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr)
    #     # self.optimizer_G = torch.optim.Adadelta(self.generator.parameters())
    #
    #     self.ssimloss = SSIM().cuda()
    #     # self.mseloss = torch.nn.MSELoss().cuda()
    #
    #     # pretrain = torch.load('Checkpoints\\%s' % tn_path)
    #     # pretrained_dict = {k[10:]: v for k, v in pretrain.items() if 'dis' not in k}
    #
    #     # self.tnet.load_state_dict(pretrained_dict)
    #     # print('Teacher net load success')
    #
    #     # for k, v in self.tnet.named_parameters():
    #     #     v.requires_grad = False  # 固定参数
    #
    #     if self.config.load_model and self.config.model_name != '':
    #         self.generator.load_state_dict(torch.load('Checkpoints\\%s' % self.config.model_name))
    #         print('student load success')
    #
    #     writer = SummaryWriter()
    #     gs = 0
    #     for epoch in range(0, epoches):
    #
    #         if epoch % self.config.save_interval == 0:
    #             torch.save(self.state_dict(), 'Checkpoints\\8-9-TSNet\\net_params_%s_%s.pkl' % (
    #                 epoch, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))
    #
    #         # scheduler.step()
    #         # Train
    #         count_train = 0
    #
    #         for train_input, train_label in self.ds_train:
    #             self.train()
    #             self.optimizer_G.zero_grad()
    #
    #             count_train += 1
    #             inputs = Variable(train_input.float()).cuda()
    #             labels = Variable(train_label.float()).cuda()
    #
    #             # t_outputs = self.tnet(inputs)
    #             s_outputs = self.generator(inputs)
    #
    #             # loss_Distillation_lst = [self.mseloss(torch.mean(t_outputs[i], dim=1), torch.mean(s_outputs[i], dim=1))
    #             #                      for i in range(len(t_outputs)-1)]
    #             # loss_Distillation = 0.0
    #             # for i in loss_Distillation_lst:
    #             #     loss_Distillation += i
    #
    #             loss = 1 - self.ssimloss(labels, s_outputs)
    #             # total_loss = loss
    #             # total_loss = loss + 0.01 * loss_Distillation
    #
    #             writer.add_scalar('train SSIM', 1-loss.item(), global_step=gs)
    #             # writer.add_scalar('train Distillation', loss_Distillation.item(), global_step=gs)
    #
    #             gs += 1
    #             # loss_train += loss.item()
    #
    #             # print(loss.item())
    #             loss.backward()
    #             self.optimizer_G.step()
    #             if gs % 100 == 0 and gs != 0:
    #                 count_test = 0
    #                 loss_test = 0.
    #                 batch = 0
    #                 with torch.no_grad():
    #                     for test_input, test_label, hbao in self.ds_test:
    #                         self.eval()
    #                         count_test += 1
    #                         inputs = Variable(test_input.float()).cuda()
    #                         labels = Variable(test_label.float()).cuda()
    #
    #                         outputs = self.generator(inputs)
    #                         loss = self.ssimloss(labels, outputs)
    #
    #                         if batch % 400 == 0:
    #
    #                             a = torch.squeeze(outputs[0, :, :, :]).cpu().detach().numpy()
    #
    #                             b = torch.squeeze(test_label[0, :, :]).numpy()
    #                             c = torch.squeeze(hbao[0, :, :]).numpy()
    #
    #                             gap = np.zeros((512, 5), dtype=np.float32)
    #                             outputs = np.concatenate((a, gap, b, gap, c), axis=1) * 255.
    #                             outputs = np.clip(outputs, 0, 255)
    #
    #                             outputs = outputs.astype(np.uint8)
    #
    #                             cv2.imwrite('Logs\\result\\8-9-TSNet\\%d_%d_%d-result.png' % (epoch, gs, batch), outputs)
    #                         batch += 1
    #
    #                         loss_test += loss.item()
    #
    #                     writer.add_scalar('test SSIM', loss_test / count_test, global_step=gs)
    #                     # print('######################################  TEST  ######################################')
    #                     # print('Test: epoch = %s, loss = %s' % (epoch, loss_test / count_test))

    def fit_ts(self, epoches, lr=1e-4, tn_path=''):
        # lr:5e-5
        xx = [i for i in self.generator.parameters()]
        print('# generator parameters:', sum(param.numel() for param in self.generator.parameters()))

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, eps=5e-4)

        self.ssimloss = MY_SSIM().cuda()
        self.mseloss = torch.nn.MSELoss().cuda()

        pretrain = torch.load('Checkpoints\\%s' % tn_path)
        pretrained_dict = {k[10:]: v for k, v in pretrain.items() if 'dis' not in k}

        self.tnet.load_state_dict(pretrained_dict)
        print('Teacher net load success')

        for k, v in self.tnet.named_parameters():
            v.requires_grad = False  # 固定参数

        if self.config.load_model and self.config.model_name != '':
            self.generator.load_state_dict(torch.load('Checkpoints\\%s' % self.config.model_name))
            print('student load success')

        # writer = SummaryWriter()
        gs = 0
        for epoch in range(0, epoches):

            if epoch % self.config.save_interval == 0:
                torch.save(self.state_dict(), 'Checkpoints\\8-9-TSNet\\net_params_%s_%s.pkl' % (
                    epoch, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))

            # scheduler.step()
            # Train
            count_train = 0

            for train_input, train_label in self.ds_train:
                self.train()
                self.optimizer_G.zero_grad()

                count_train += 1
                inputs = Variable(train_input.float()).cuda()
                labels = Variable(train_label.float()).cuda()

                t_outputs = self.tnet(inputs)
                s_outputs = self.generator(inputs)

                loss_Distillation_lst = [self.mseloss(torch.mean(t_outputs[i], dim=1), torch.mean(s_outputs[i], dim=1))
                                         for i in range(len(t_outputs) - 1)]
                loss_Distillation = 0.0
                for i in loss_Distillation_lst:
                    loss_Distillation += i

                loss = 1 - self.ssimloss(labels, s_outputs[-1])
                total_loss = loss + 0.01 * loss_Distillation

                # writer.add_scalar('train SSIM', 1 - loss.item(), global_step=gs)
                # writer.add_scalar('train Distillation', loss_Distillation.item(), global_step=gs)

                gs += 1
                # loss_train += loss.item()

                # print(loss.item())
                total_loss.backward()
                self.optimizer_G.step()
                if gs % 500 == 0 and gs != 0:
                    count_test = 0
                    loss_test = 0.
                    batch = 0
                    with torch.no_grad():
                        times = 0
                        times_count = 0
                        for test_input, test_label, hbao in self.ds_test:
                            self.eval()
                            count_test += 1
                            inputs = Variable(test_input.float()).cuda()
                            labels = Variable(test_label.float()).cuda()

                            start = datetime.datetime.now()
                            outputs = self.generator(inputs)
                            end = datetime.datetime.now()

                            print((end-start).microseconds)
                            loss = self.ssimloss(labels, outputs[-1])

                            if batch % 200 == 0:
                                a = torch.squeeze(outputs[-1][0, :, :, :]).cpu().detach().numpy()

                                b = torch.squeeze(test_label[0, :, :]).numpy()
                                c = torch.squeeze(hbao[0, :, :]).numpy()

                                gap = np.zeros((512, 5), dtype=np.float32)
                                outputs = np.concatenate((a, gap, b, gap, c), axis=1) * 255.
                                outputs = np.clip(outputs, 0, 255)

                                outputs = outputs.astype(np.uint8)

                                cv2.imwrite('Logs\\result\\8-9-TSNet\\%d_%d_%d-result.png' % (epoch, gs, batch),
                                            outputs)
                            batch += 1

                            loss_test += loss.item()

                        # writer.add_scalar('test SSIM', loss_test / count_test, global_step=gs)
                        # print('######################################  TEST  ######################################')
                        # print('Test: epoch = %s, loss = %s' % (epoch, loss_test / count_test))

    def fit_ts_wgan(self, epoches, lr, tn_path=''):
        lr = 5e-5
        # self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=lr, eps=1e-4)
        # self.optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=lr, eps=1e-4)

        self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=lr)
        self.optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=lr)


        pretrain = torch.load('Checkpoints\\%s' % tn_path)
        pretrained_dict = {k[10:]: v for k, v in pretrain.items() if 'dis' not in k}

        self.tnet.load_state_dict(pretrained_dict)
        print('Teacher net load success')

        for k, v in self.tnet.named_parameters():
            v.requires_grad = False  # 固定参数

        # self.optimizer_G = RMSprop16(self.generator.parameters(), lr=lr, eps=1e-4)
        # self.optimizer_D = RMSprop16(self.discriminator.parameters(), lr=lr, eps=1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.decay_every, gamma=0.5)

        # self.loss = nn.MSELoss().cuda()
        self.baseloss = SSIM().cuda()
        self.testloss = SSIM().cuda()

        if self.config.load_model and self.config.model_name != '':
            self.load_state_dict(torch.load('Checkpoints\\Unet+wgan\\%s' % self.config.model_name))
            print('load success')

        # writer = SummaryWriter()

        gs_d, gs_g = 0, 0
        # cuda = True if torch.cuda.is_available() else False
        # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        for epoch in range(0, epoches):
            if epoch % self.config.save_interval == 0:
                torch.save(self.state_dict(), 'Checkpoints\\8-6-Unet+wgan\\net_params_%s_%s.pkl' % (
                    epoch, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))

            # scheduler.step()
            # Train
            count_train = 0

            for train_input, train_label in self.ds_train:
                self.generator.train()
                self.discriminator.train()
                self.optimizer_D.zero_grad()

                count_train += 1
                inputs = Variable(train_input.float()).cuda()
                labels = Variable(train_label.float()).cuda()

                fake_imgs = self.generator(inputs)[:-1]

                real_out = self.discriminator(labels)
                fake_out = self.discriminator(fake_imgs)
                loss_D = - torch.mean(real_out) + torch.mean(fake_out)
                # writer.add_scalar('loss_D', loss_D.item(), global_step=gs_d)
                gs_d += 1

                loss_D.backward()
                self.optimizer_D.step()

                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-self.config.clip_value, self.config.clip_value)

                self.optimizer_G.zero_grad()
                if gs_d % self.config.n_critic == 0:
                    fake_imgs = self.generator(inputs)

                    t_outputs = self.tnet(inputs)
                    loss_Distillation_lst = [
                        self.mseloss(torch.mean(t_outputs[i], dim=1), torch.mean(fake_imgs[i], dim=1))
                        for i in range(len(t_outputs) - 1)]
                    loss_Distillation = 0.0
                    for i in loss_Distillation_lst:
                        loss_Distillation += i

                    # Adversarial loss
                    fake_out = torch.mean(self.discriminator(fake_imgs))

                    train_ssim_loss = 1 - self.baseloss(fake_imgs, labels)
                    train_ssim_loss_01 = 1 - self.testloss(fake_imgs, labels)

                    # writer.add_scalar('train ssim', 1-train_ssim_loss.item(), global_step=gs_g)
                    # writer.add_scalars('train ssim',
                    #                    {'no norm': 1 - train_ssim_loss.item(), 'norm': 1 - train_ssim_loss_01.item()},
                    #                    global_step=gs_g)

                    loss_G = 0.001 * (-fake_out) + train_ssim_loss
                    # writer.add_scalar('loss_G', loss_G.item(), global_step=gs_g)

                    # print(loss_G.item())
                    gs_g += 1
                    loss_G.backward()
                    self.optimizer_G.step()

                if gs_d % 1000 == 1:
                    count_test = 0
                    loss_test = 0.
                    loss_test_nonorm = 0.0
                    batch = 0
                    with torch.no_grad():
                        for test_input, test_label in self.ds_test:

                            self.generator.eval()
                            count_test += 1
                            inputs = Variable(test_input.float()).cuda()
                            labels = Variable(test_label.float()).cuda()

                            outputs = self.generator(inputs)
                            loss_no_norm = self.baseloss(outputs, labels)
                            loss_norm = self.testloss(outputs, labels)
                            if batch % 100 == 0:
                                a = torch.squeeze(outputs[0, :, :, :]).cpu().detach().numpy()
                                b = torch.squeeze(test_label[0, :, :]).numpy()
                                outputs = np.concatenate((a, np.zeros((512, 5), dtype=np.float32), b), axis=1) * 255.
                                outputs = np.clip(outputs, 0, 255)

                                outputs = outputs.astype(np.uint8)

                                cv2.imwrite('Logs\\result\\8-6-Unet+wgan\\%d_%d_%d-result.png' % (epoch, gs_g, batch),
                                            outputs)
                            batch += 1
                            loss_test += loss_norm.item()
                            loss_test_nonorm += loss_no_norm.item()

                        # writer.add_scalar('test SSIM', loss_test / count_test, global_step=gs_g)
                        # writer.add_scalars('test ssim', {'no norm': loss_test_nonorm / count_test,
                        #                                  'norm': loss_test / count_test}, global_step=gs_g)
                        print('######################################  TEST  ######################################')
                        print('Test: epoch = %s, loss = %s' % (epoch, loss_test / count_test))

    def predict_gan(self, model_name, shape=[]):

        # xx = torch.load('Checkpoints\\%s' % model_name)
        self.load_state_dict(torch.load('Checkpoints\\%s' % model_name))
        print('load success')
        count = 0
        bs = self.config.bs
        self.baseloss = SSIM().cuda()

        self.mseloss = nn.MSELoss().cuda()
        loss_avg_stand = 0
        loss_avg_nostand = 0
        with torch.no_grad():
            for test_input, test_label in self.ds_test:
                self.generator.eval()
                inputs = Variable(test_input.float()).cuda()
                labels = Variable(test_label.float()).cuda()

                outputs = self.generator(inputs)

                loss = self.baseloss(outputs, labels)
                loss_avg_nostand += loss.item()

                loss_stand = self.baseloss(labels, outputs)
                loss_avg_stand += loss_stand.item()
                mse = self.mseloss(outputs, labels)
                count += 1

                print(loss.item(), loss_stand.item(), mse.item())
            print(loss_avg_nostand / count, loss_avg_stand / count)

            # a = torch.squeeze(outputs[0, :, :, :]).cpu().detach().numpy()
            # b = torch.squeeze(test_label[0, :, :]).numpy()
            #
            # outputs = np.concatenate((a, np.zeros((a.shape[0], 5), dtype=np.float32), b), axis=1) * 255.
            # outputs = np.clip(outputs, 0, 255)
            #
            # outputs = outputs.astype(np.uint8)
            # import matplotlib.pyplot as plt
            # plt.imshow(outputs, cmap='gray')
            # plt.show()

            # cv2.imwrite('Logs\\result\\Unet+wgan\\%d_%d_%d-result.png' % (epoch, gs_g, batch), outputs)


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if k != 'meta':
                    self.batch[k] = self.batch[k].to(device=self.opt.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch



class NANO_cnn(nn.Module):
    def __init__(self, ds_train, config, ds_test=None, ds_val=None):
        super(NANO_cnn, self).__init__()

        self.ds_train = ds_train
        self.ds_test = ds_test
        self.ds_val = ds_val
        self.config = config
        # self.generator = UNet(4, 1)
        self.generator = UNet_multi_scale(4, 1)

        self.generators = {'ours': UNet_multi_scale(4, 1), 'deepshading':DSUNet(6, 1)}
        # self.generator = DSUNet(6, 1)
        # input_size, input_dim, hidden_dim, kernel_size, num_layers
        # self.generator = ConvLSTM((512, 512), 4, 1, (3, 3), 8)
        # self.generator = RRDBNet(4, 1, 8, 8, gc=8)
        # self.discriminator = Discriminator()

    def forward(self, x):
        res = self.generator(x)
        return res

    def fit(self, epoches, lr=1e-4):

        self.generator.cuda()
        print('# generator parameters:', sum(param.numel() for param in self.generator.parameters()))
        # self.optimizer_G = RAdam(self.generator.parameters(), lr=lr)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr)
        # self.ssimloss = MY_SSIM().cuda()
        self.ms_ssimloss = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=1)
        self.L1loss = torch.nn.L1Loss().cuda()
        self.ssimloss = SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=1).cuda()

        if self.config.load_model and self.config.model_name != '':
            self.load_state_dict(torch.load('Checkpoints\\%s' % self.config.model_name))
            print('load success')

        writer = SummaryWriter()
        gs = 0

        for epoch in range(0, epoches):

            if epoch % self.config.save_interval == 0:
                torch.save(self.state_dict(), 'Checkpoints\\%s\\net_params_%s_%s.pkl' % (
                    self.config.save_name, epoch, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))

            # Train
            loss_train, loss_train_exp, acc_train, = 0., 0., 0.
            count_train = 0

            train_prefetcher = DataPrefetcher(self.ds_train)
            data, label = train_prefetcher.next()

            iteration = 0
            while data is not None:
                iteration += 1
                # 训练代码
                data, label = train_prefetcher.next()
                print()

            for train_input, train_label in self.ds_train:
                self.generator.train()
                self.optimizer_G.zero_grad()

                count_train += 1
                inputs = Variable(train_input.float()).cuda()
                labels = Variable(train_label.float()).cuda()

                outputs = self.forward(inputs)

                msssim_loss = 1-self.ms_ssimloss(labels, outputs)
                # if torch.isnan(msssim_loss):
                #     continue
                # print('loss', msssim_loss.item())
                ssim_loss = self.ssimloss(labels, outputs)
                l1loss = self.L1loss(labels, outputs)
                total_loss = 0.84*msssim_loss + (1-0.84)*1.5*l1loss

                # print(1-msssim_loss.item())
                writer.add_scalars('train SSIM', {'MS-SSMM':1-msssim_loss.item(), 'SSIM':ssim_loss.item()}, global_step=gs)
                writer.add_scalar('train L1',l1loss.item(), global_step=gs)
                # writer.add_scalar('train ssim loss', 1 - msssim_loss.item(), global_step=gs)
                # writer.add_scalar('train L1 loss', l1loss.item(), global_step=gs)
                # print(loss.item())
                gs += 1
                # loss_train += loss.item()

                total_loss.backward()
                self.optimizer_G.step()
                if gs % 20000 == 0:

                    count_test = 0
                    msssim_loss_test = 0.
                    ssim_loss_test = 0.
                    L1_loss_test = 0.

                    batch = 0
                    with torch.no_grad():
                        for test_input, test_label in self.ds_test:
                            self.generator.eval()
                            count_test += 1
                            inputs = Variable(test_input.float()).cuda()
                            labels = Variable(test_label.float()).cuda()

                            outputs = self.forward(inputs)
                            msssim_loss = self.ms_ssimloss(labels, outputs)
                            l1loss = self.L1loss(labels, outputs)
                            ssim_loss = self.ssimloss(labels, outputs)

                            msssim_loss_test += msssim_loss.item()
                            ssim_loss_test += ssim_loss.item()
                            L1_loss_test += l1loss.item()

                            if batch % 400 == 0:
                                a = torch.squeeze(outputs[0, :, :, :]).cpu().detach().numpy()
                                b = torch.squeeze(test_label[0, :, :]).numpy()
                                outputs = np.concatenate((a,
                                                          np.zeros((512, 5), dtype=np.float32),
                                                          b
                                                          ), axis=1) * 255.
                                outputs = np.clip(outputs, 0, 255)

                                outputs = outputs.astype(np.uint8)

                                cv2.imwrite('Logs\\result\\%s\\%d_%d_%d-result.png' % (self.config.save_name, epoch, gs, batch), outputs)
                            batch += 1

                            # loss_test += loss.item()

                        writer.add_scalars('Test SSIM', {'MS-SSIM':msssim_loss_test / count_test, 'SSIM':ssim_loss_test / count_test}, global_step=gs)
                        writer.add_scalar('Test L1', L1_loss_test / count_test, global_step=gs)
                        # writer.add_scalar('test ssim', ssim_loss_test / count_test, global_step=gs)
                        # writer.add_scalar('test l1', L1_loss_test / count_test, global_step=gs)
                        # print('Test: global step = %s, ssim = %s' % (gs, loss_test / count_test))



    def predict(self, model_name, model='ours'):

        print(model)
        if model=='ours':
            self.generator=self.generators['ours']
        elif model=='deepshading':
            self.generator = self.generators['deepshading']
        else:
            print('model not exist!')
            return
        self.generator.cuda()
        self.load_state_dict(torch.load('Checkpoints\\%s' % model_name))

        print('# generator parameters:', sum(param.numel() for param in self.generator.parameters()))
        print('load success')

        self.baseloss = SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=1).cuda()

        with torch.no_grad():

            count = 0
            total_loss = 0
            time_count = 0
            total_rmse_loss = 0

            for test_input, test_label in self.ds_test:
                self.generator.eval()
                inputs = Variable(test_input.float()).cuda()
                labels = Variable(test_label.float()).cuda()

                start_ = datetime.datetime.now()
                outputs = self.generator(inputs)
                end_ = datetime.datetime.now()
                times = (end_-start_).microseconds
                if count != 0:
                    time_count += times
                loss = self.baseloss(labels, outputs)
                rmse_loss = torch.nn.L1Loss()(labels, outputs).item()
                total_rmse_loss += rmse_loss
                # print(times)
                print(loss.item())
                total_loss += loss.item()

                print(loss.item())
                count += 1
                print(count)
                # if count == 100:
                #     break
                for i in range(outputs.shape[0]):
                    res = np.clip(outputs[i, 0, :, :].detach().cpu().numpy(), 0, 1.)


                    import pyexr
                    pyexr.write('Logs\\test2\\%s-%s.exr' % (count, model), np.concatenate([np.expand_dims(res, axis=-1), np.expand_dims(res, axis=-1), np.expand_dims(res, axis=-1)], axis=-1))
                    res = res*1.0
                    res = res*255.
                    res = np.clip(res, 0, 255)
                    res = res.astype(np.uint8)
                    cv2.imwrite('Logs\\test2\\%s-%s.png' % (count, model), res)
                    lb = torch.squeeze(test_label[0, :, :]).numpy()

                    gt = lb * 255.
                    gt = np.clip(gt, 0, 255)
                    gt = gt.astype(np.uint8)
                    cv2.imwrite('Logs\\test2\\%s-gt.png' % count, gt)


                    out = np.concatenate((res,
                                              np.zeros((512, 5), dtype=np.float32),
                                              lb
                                              ), axis=1) * 255.

                    cv2.imwrite('Logs\\test2\\%05d-result_%s.png' % (count, model), out)
                    # cv2.imwrite('Logs\\K8\\%05d-gt.png' % count, lb)
                    count += 1
            print('\n')
            print('loss = %f' % (total_loss/count))
            print('rmse_loss = %f' % (total_rmse_loss / count))
            # print('time = %f' % (time_count / count))

    def fit_wgan(self, epoches, lr):
        lr = 5e-5
        # self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=lr, eps=1e-4)
        # self.optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=lr, eps=1e-4)
        self.generator.cuda()
        self.discriminator.cuda()
        self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=lr)
        self.optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=lr)

        # self.optimizer_G = RMSprop16(self.generator.parameters(), lr=lr, eps=1e-4)
        # self.optimizer_D = RMSprop16(self.discriminator.parameters(), lr=lr, eps=1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.decay_every, gamma=0.5)

        # self.loss = nn.MSELoss().cuda()
        self.baseloss = SSIM().cuda()

        if self.config.load_model and self.config.model_name != '':
            self.load_state_dict(torch.load('Checkpoints\\Unet+wgan\\%s' % self.config.model_name))
            print('load success')

        # writer = SummaryWriter()

        gs_d, gs_g = 0, 0
        # cuda = True if torch.cuda.is_available() else False
        # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        for epoch in range(0, epoches):
            if epoch % self.config.save_interval == 0:
                torch.save(self.state_dict(), 'Checkpoints\\8-9-Unet+wgan\\net_params_%s_%s.pkl' % (
                    epoch, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))

            # scheduler.step()
            # Train
            count_train = 0

            for train_input, train_label in self.ds_train:
                self.generator.train()
                self.discriminator.train()
                self.optimizer_D.zero_grad()

                count_train += 1
                inputs = Variable(train_input.float()).cuda()
                labels = Variable(train_label.float()).cuda()

                fake_imgs = self.generator(inputs).detach()

                real_out = self.discriminator(labels)
                fake_out = self.discriminator(fake_imgs)
                loss_D = - torch.mean(real_out) + torch.mean(fake_out)
                # writer.add_scalar('loss_D', loss_D.item(), global_step=gs_d)
                gs_d += 1

                loss_D.backward()
                self.optimizer_D.step()

                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-self.config.clip_value, self.config.clip_value)

                self.optimizer_G.zero_grad()
                if gs_d % self.config.n_critic == 0:
                    fake_imgs = self.generator(inputs)
                    # Adversarial loss
                    fake_out = torch.mean(self.discriminator(fake_imgs))

                    train_ssim_loss = 1 - self.baseloss(fake_imgs, labels)

                    # writer.add_scalar('train ssim', 1-train_ssim_loss.item(), global_step=gs_g)
                    # writer.add_scalar('train ssim', 1-train_ssim_loss.item(), global_step=gs_g)

                    loss_G = 0.01*(-fake_out) + train_ssim_loss
                    # writer.add_scalar('loss_G', loss_G.item(), global_step=gs_g)

                    # print(loss_G.item())
                    gs_g += 1
                    loss_G.backward()
                    self.optimizer_G.step()

                if gs_d % 1000 == 1:
                    count_test = 0
                    loss_test = 0.
                    loss_test_nonorm = 0.0
                    batch = 0
                    with torch.no_grad():
                        for test_input, test_label, hbao in self.ds_test:

                            self.generator.eval()
                            count_test += 1
                            inputs = Variable(test_input.float()).cuda()
                            labels = Variable(test_label.float()).cuda()

                            outputs = self.generator(inputs)
                            loss_no_norm = self.baseloss(labels, outputs)
                            if batch % 100 == 0:
                                a = torch.squeeze(outputs[0, :, :, :]).cpu().detach().numpy()
                                b = torch.squeeze(test_label[0, :, :]).numpy()
                                c = hbao[0, 0, :, :].numpy()
                                outputs = np.concatenate((a,
                                                          np.zeros((512, 5), dtype=np.float32),
                                                          b,
                                                          np.zeros((512, 5), dtype=np.float32),
                                                          c), axis=1) * 255.
                                outputs = np.clip(outputs, 0, 255)

                                outputs = outputs.astype(np.uint8)

                                cv2.imwrite('Logs\\result\\8-9-Unet+wgan\\%d_%d_%d-result.png' % (epoch, gs_g, batch), outputs)
                            batch += 1
                            loss_test_nonorm += loss_no_norm.item()

                        # writer.add_scalar('test SSIM', loss_test / count_test, global_step=gs_g)
                        # writer.add_scalar('test ssim', loss_test_nonorm / count_test, global_step=gs_g)
                        # print('######################################  TEST  ######################################')
                        # print('Test: epoch = %s, loss = %s' % (epoch, loss_test / count_test))

    def fit_wgan_div(self, epoches, lr=0.0002):

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.decay_every, gamma=0.5)

        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        # self.mseloss = nn.MSELoss().cuda()
        self.baseloss = SSIM().cuda()

        if self.config.load_model and self.config.model_name != '':
            self.load_state_dict(torch.load('Checkpoints\\Unet+wgan-div\\%s' % self.config.model_name))
            print('load success')

        # writer = SummaryWriter()

        # gs_d, gs_g = 120800, 24200
        gs_d, gs_g = 0, 0
        # cuda = True if torch.cuda.is_available() else False
        # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        for epoch in range(0, epoches):

            if epoch % self.config.save_interval == 0:
                torch.save(self.state_dict(), 'Checkpoints\\Unet+wgan-div\\net_params_%s_%s.pkl' % (
                    epoch, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))

            loss_train, loss_train_exp, acc_train, = 0., 0., 0.
            count_train = 0
            k = 2
            p = 6
            for train_input, train_label in self.ds_train:
                self.generator.train()
                self.discriminator.train()
                self.optimizer_D.zero_grad()

                count_train += 1
                inputs = Variable(train_input.float()).cuda()
                labels = Variable(train_label.float(), requires_grad=True).cuda()

                # Sample noise as generator input
                # z = Variable(Tensor(np.random.normal(0, 1, (inputs.shape)))).cuda()
                fake_imgs = self.generator(inputs)

                real_out = self.discriminator(labels)
                fake_out = self.discriminator(fake_imgs)

                real_grad_out = Variable(Tensor(real_out.size(0)).fill_(1.0), requires_grad=False)
                real_grad = autograd.grad(
                    real_out, labels, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True)[0]
                real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

                fake_grad_out = Variable(Tensor(fake_imgs.size(0)).fill_(1.0), requires_grad=False)
                fake_grad = autograd.grad(
                    fake_out, fake_imgs, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True)[0]
                fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

                div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2

                loss_D = -torch.mean(real_out) + torch.mean(fake_out) + div_gp
                # loss_D = 1 - torch.mean(real_out) + torch.mean(fake_out)
                # writer.add_scalar('loss_D', loss_D.item(), global_step=gs_d)

                # outputs = self.forward(inputs)

                # loss = 1-self.loss(outputs, labels)
                # nn.MSELoss
                # writer.add_scalar('train loss', loss.item(), global_step=gs)
                gs_d += 1
                # loss_train += loss.item()

                loss_D.backward()
                self.optimizer_D.step()

                # Clip weights of discriminator
                # for p in self.discriminator.parameters():
                #     p.data.clamp_(-self.config.clip_value, self.config.clip_value)
                self.optimizer_G.zero_grad()
                if gs_d % self.config.n_critic == 0:

                    fake_imgs = self.generator(inputs)
                    # Adversarial loss
                    fake_out = torch.mean(self.discriminator(fake_imgs))

                    train_ssim_loss = 1 - self.baseloss(fake_imgs, labels)
                    # l1_loss = torch.nn.L1Loss()(fake_imgs, labels)

                    # writer.add_scalar('train_ssim_loss', 1-train_ssim_loss.item(), global_step=gs_g)
                    loss_G = 0.001 * (-fake_out) + train_ssim_loss
                    # writer.add_scalar('loss_G', loss_G.item(), global_step=gs_g)
                    gs_g += 1
                    loss_G.backward()
                    self.optimizer_G.step()

                if gs_d % 1000 == 0:
                    count_test = 0
                    loss_test = 0.
                    loss_ssim_test = 0.
                    batch = 0
                    with torch.no_grad():
                        for test_input, test_label in self.ds_test:

                            self.generator.eval()
                            count_test += 1
                            inputs = Variable(test_input.float()).cuda()
                            labels = Variable(test_label.float()).cuda()

                            outputs = self.generator(inputs)
                            ssim_loss = self.baseloss(outputs, labels)
                            # loss = torch.nn.L1Loss()(outputs, labels)
                            if batch % 100 == 0:
                                a = torch.squeeze(outputs[0, :, :, :]).cpu().detach().numpy()
                                b = torch.squeeze(test_label[0, :, :]).numpy()
                                outputs = np.concatenate((a, np.zeros((512, 5), dtype=np.float32), b),
                                                         axis=1) * 255.
                                outputs = np.clip(outputs, 0, 255)

                                outputs = outputs.astype(np.uint8)

                                cv2.imwrite('Logs\\result\\Unet+wgan-div\\%d_%d_%d-result.png' % (epoch, gs_g, batch),
                                            outputs)
                            batch += 1
                            # loss_test += ssim_loss.item()
                            loss_ssim_test += ssim_loss.item()

                        # writer.add_scalar('test l1_loss', loss_test / count_test, global_step=gs_g)
                        # writer.add_scalar('test ssim_loss', loss_ssim_test / count_test, global_step=gs_g)

                        print(
                            '######################################  TEST  ######################################')
                        print('Test: epoch = %s, loss = %s' % (epoch, loss_test / count_test))


class NANO_LSTM(nn.Module):
    def __init__(self, ds_train, config, ds_test=None, ds_val=None):
        super(NANO_LSTM, self).__init__()

        self.ds_train = ds_train
        self.ds_test = ds_test
        self.ds_val = ds_val
        self.config = config
        # self.generator = UNet(4, 1)
        # input_size, input_dim, hidden_dim, kernel_size, num_layers
        self.generator = ConvLSTM((512, 512), 4, 1, (3, 3), 8)
        # self.generator = RRDBNet(4, 1, 8, 8, gc=8)
        # self.discriminator = Discriminator()

    def forward(self, x):
        res = self.generator(x)
        return res

    def fit(self, epoches, lr=1e-4):

        self.generator.cuda()
        print('# generator parameters:', sum(param.numel() for param in self.generator.parameters()))
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.ssimloss = SSIM().cuda()

        if self.config.load_model and self.config.model_name != '':
            self.load_state_dict(torch.load('Checkpoints\\%s' % self.config.model_name))
            print('load success')

        writer = SummaryWriter()
        gs = 0
        for epoch in range(0, epoches):

            if epoch % self.config.save_interval == 0:
                torch.save(self.state_dict(), 'Checkpoints\\%s\\net_params_%s_%s.pkl' % (
                    self.config.save_name, epoch, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))

            # Train
            loss_train, loss_train_exp, acc_train, = 0., 0., 0.
            count_train = 0

            for train_input, train_label in self.ds_train:
                self.generator.train()
                self.optimizer_G.zero_grad()

                count_train += 1
                inputs = Variable(train_input.float()).cuda()
                labels = Variable(train_label.float()).cuda()

                outputs = torch.squeeze(self.forward(inputs)[0][0])[:, -1:, :, :]

                loss = 1-self.ssimloss(labels[:, -1:, :, :], outputs)
                writer.add_scalar('train ssim', 1-loss.item(), global_step=gs)
                # print(loss.item())
                gs += 1
                loss_train += 1-loss.item()

                loss.backward()
                self.optimizer_G.step()
                if gs % 100 == 0:
                    print('Train SSIM: global step = %s, ssim = %s' % (gs, loss_train / count_train))
                # print('train', gs)
                if gs % 1000 == 0:
                    # print(gs)
                    count_test = 0
                    loss_test = 0.
                    batch = 0
                    with torch.no_grad():
                        # print('test')
                        for test_input, test_label in self.ds_test:
                            self.generator.eval()
                            count_test += 1
                            inputs = Variable(test_input.float()).cuda()
                            labels = Variable(test_label.float()).cuda()

                            # outputs = self.forward(inputs)
                            outputs = torch.squeeze(self.forward(inputs)[0][0])[:, -1:, :, :]
                            loss = self.ssimloss(labels[:, -1:, :, :], outputs)

                            # print('')
                            if batch % 4 == 0:
                                a = torch.squeeze(outputs[0, -1, :, :]).cpu().detach().numpy()
                                # a = [np.squeeze(i) for i in np.split(a, 4)]
                                # a = np.concatenate(a, axis=1)


                                b = torch.squeeze(test_label[0, -1, :, :]).numpy()
                                # b = [np.squeeze(i) for i in np.split(b, 4)]
                                # b = np.concatenate(b, axis=1)
                                # plt.imshow(b[ :, :])
                                # plt.show()
                                outputs = np.concatenate((a,

                                                          b
                                                          ), axis=1) * 255.
                                outputs = np.clip(outputs, 0, 255)

                                # plt.imshow(outputs[ :, :])
                                # plt.show()

                                outputs = outputs.astype(np.uint8)

                                cv2.imwrite('Logs\\result\\%s\\%d_%d_%d-result.png' % (self.config.save_name, epoch, gs, batch), outputs)
                            batch += 1

                            loss_test += loss.item()

                        writer.add_scalar('test ssim', loss_test / count_test, global_step=gs)
                        print('Test: global step = %s, ssim = %s' % (gs, loss_test / count_test))


    def predict(self, model_name):

        self.generator.cuda()
        self.load_state_dict(torch.load('Checkpoints\\%s' % model_name))

        print('# generator parameters:', sum(param.numel() for param in self.generator.parameters()))
        print('load success')

        self.baseloss = SSIM().cuda()

        with torch.no_grad():

            count = 0
            for test_input, test_label in self.ds_test:
                self.generator.eval()
                inputs = Variable(test_input.float()).cuda()
                # labels = Variable(test_label.float()).cuda()
                outputs = self.generator(inputs)

                for i in range(outputs.shape[0]):
                    res = np.clip(outputs[i, 0, :, :].detach().cpu().numpy()*255., 0, 255)
                    cv2.imwrite('Logs\\K4\\%05d-result_2.png' % count, res)
                    # cv2.imwrite('Logs\\K8\\%05d-gt.png' % count, lb)
                    count += 1


    def fit_wgan(self, epoches, lr):
        lr = 5e-5
        # self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=lr, eps=1e-4)
        # self.optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=lr, eps=1e-4)
        self.generator.cuda()
        self.discriminator.cuda()
        self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=lr)
        self.optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=lr)

        # self.optimizer_G = RMSprop16(self.generator.parameters(), lr=lr, eps=1e-4)
        # self.optimizer_D = RMSprop16(self.discriminator.parameters(), lr=lr, eps=1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.decay_every, gamma=0.5)

        # self.loss = nn.MSELoss().cuda()
        self.baseloss = SSIM().cuda()

        if self.config.load_model and self.config.model_name != '':
            self.load_state_dict(torch.load('Checkpoints\\Unet+wgan\\%s' % self.config.model_name))
            print('load success')

        # writer = SummaryWriter()

        gs_d, gs_g = 0, 0
        # cuda = True if torch.cuda.is_available() else False
        # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        for epoch in range(0, epoches):
            if epoch % self.config.save_interval == 0:
                torch.save(self.state_dict(), 'Checkpoints\\8-9-Unet+wgan\\net_params_%s_%s.pkl' % (
                    epoch, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))

            # scheduler.step()
            # Train
            count_train = 0

            for train_input, train_label in self.ds_train:
                self.generator.train()
                self.discriminator.train()
                self.optimizer_D.zero_grad()

                count_train += 1
                inputs = Variable(train_input.float()).cuda()
                labels = Variable(train_label.float()).cuda()

                fake_imgs = self.generator(inputs).detach()

                real_out = self.discriminator(labels)
                fake_out = self.discriminator(fake_imgs)
                loss_D = - torch.mean(real_out) + torch.mean(fake_out)
                # writer.add_scalar('loss_D', loss_D.item(), global_step=gs_d)
                gs_d += 1

                loss_D.backward()
                self.optimizer_D.step()

                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-self.config.clip_value, self.config.clip_value)

                self.optimizer_G.zero_grad()
                if gs_d % self.config.n_critic == 0:
                    fake_imgs = self.generator(inputs)
                    # Adversarial loss
                    fake_out = torch.mean(self.discriminator(fake_imgs))

                    train_ssim_loss = 1 - self.baseloss(fake_imgs, labels)

                    # writer.add_scalar('train ssim', 1-train_ssim_loss.item(), global_step=gs_g)
                    # writer.add_scalar('train ssim', 1-train_ssim_loss.item(), global_step=gs_g)

                    loss_G = 0.01*(-fake_out) + train_ssim_loss
                    # writer.add_scalar('loss_G', loss_G.item(), global_step=gs_g)

                    # print(loss_G.item())
                    gs_g += 1
                    loss_G.backward()
                    self.optimizer_G.step()

                if gs_d % 1000 == 1:
                    count_test = 0
                    loss_test = 0.
                    loss_test_nonorm = 0.0
                    batch = 0
                    with torch.no_grad():
                        for test_input, test_label, hbao in self.ds_test:

                            self.generator.eval()
                            count_test += 1
                            inputs = Variable(test_input.float()).cuda()
                            labels = Variable(test_label.float()).cuda()

                            outputs = self.generator(inputs)
                            loss_no_norm = self.baseloss(labels, outputs)
                            if batch % 100 == 0:
                                a = torch.squeeze(outputs[0, :, :, :]).cpu().detach().numpy()
                                b = torch.squeeze(test_label[0, :, :]).numpy()
                                c = hbao[0, 0, :, :].numpy()
                                outputs = np.concatenate((a,
                                                          np.zeros((512, 5), dtype=np.float32),
                                                          b,
                                                          np.zeros((512, 5), dtype=np.float32),
                                                          c), axis=1) * 255.
                                outputs = np.clip(outputs, 0, 255)

                                outputs = outputs.astype(np.uint8)

                                cv2.imwrite('Logs\\result\\8-9-Unet+wgan\\%d_%d_%d-result.png' % (epoch, gs_g, batch), outputs)
                            batch += 1
                            loss_test_nonorm += loss_no_norm.item()

                        # writer.add_scalar('test SSIM', loss_test / count_test, global_step=gs_g)
                        # writer.add_scalar('test ssim', loss_test_nonorm / count_test, global_step=gs_g)
                        # print('######################################  TEST  ######################################')
                        # print('Test: epoch = %s, loss = %s' % (epoch, loss_test / count_test))

    def fit_wgan_div(self, epoches, lr=0.0002):

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.decay_every, gamma=0.5)

        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        # self.mseloss = nn.MSELoss().cuda()
        self.baseloss = SSIM().cuda()

        if self.config.load_model and self.config.model_name != '':
            self.load_state_dict(torch.load('Checkpoints\\Unet+wgan-div\\%s' % self.config.model_name))
            print('load success')

        # writer = SummaryWriter()

        # gs_d, gs_g = 120800, 24200
        gs_d, gs_g = 0, 0
        # cuda = True if torch.cuda.is_available() else False
        # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        for epoch in range(0, epoches):

            if epoch % self.config.save_interval == 0:
                torch.save(self.state_dict(), 'Checkpoints\\Unet+wgan-div\\net_params_%s_%s.pkl' % (
                    epoch, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))

            loss_train, loss_train_exp, acc_train, = 0., 0., 0.
            count_train = 0
            k = 2
            p = 6
            for train_input, train_label in self.ds_train:
                self.generator.train()
                self.discriminator.train()
                self.optimizer_D.zero_grad()

                count_train += 1
                inputs = Variable(train_input.float()).cuda()
                labels = Variable(train_label.float(), requires_grad=True).cuda()

                # Sample noise as generator input
                # z = Variable(Tensor(np.random.normal(0, 1, (inputs.shape)))).cuda()
                fake_imgs = self.generator(inputs)

                real_out = self.discriminator(labels)
                fake_out = self.discriminator(fake_imgs)

                real_grad_out = Variable(Tensor(real_out.size(0)).fill_(1.0), requires_grad=False)
                real_grad = autograd.grad(
                    real_out, labels, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True)[0]
                real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

                fake_grad_out = Variable(Tensor(fake_imgs.size(0)).fill_(1.0), requires_grad=False)
                fake_grad = autograd.grad(
                    fake_out, fake_imgs, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True)[0]
                fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

                div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2

                loss_D = -torch.mean(real_out) + torch.mean(fake_out) + div_gp
                # loss_D = 1 - torch.mean(real_out) + torch.mean(fake_out)
                # writer.add_scalar('loss_D', loss_D.item(), global_step=gs_d)

                # outputs = self.forward(inputs)

                # loss = 1-self.loss(outputs, labels)
                # nn.MSELoss
                # writer.add_scalar('train loss', loss.item(), global_step=gs)
                gs_d += 1
                # loss_train += loss.item()

                loss_D.backward()
                self.optimizer_D.step()

                # Clip weights of discriminator
                # for p in self.discriminator.parameters():
                #     p.data.clamp_(-self.config.clip_value, self.config.clip_value)
                self.optimizer_G.zero_grad()
                if gs_d % self.config.n_critic == 0:

                    fake_imgs = self.generator(inputs)
                    # Adversarial loss
                    fake_out = torch.mean(self.discriminator(fake_imgs))

                    train_ssim_loss = 1 - self.baseloss(fake_imgs, labels)
                    # l1_loss = torch.nn.L1Loss()(fake_imgs, labels)

                    # writer.add_scalar('train_ssim_loss', 1-train_ssim_loss.item(), global_step=gs_g)
                    loss_G = 0.001 * (-fake_out) + train_ssim_loss
                    # writer.add_scalar('loss_G', loss_G.item(), global_step=gs_g)
                    gs_g += 1
                    loss_G.backward()
                    self.optimizer_G.step()

                if gs_d % 1000 == 0:
                    count_test = 0
                    loss_test = 0.
                    loss_ssim_test = 0.
                    batch = 0
                    with torch.no_grad():
                        for test_input, test_label in self.ds_test:

                            self.generator.eval()
                            count_test += 1
                            inputs = Variable(test_input.float()).cuda()
                            labels = Variable(test_label.float()).cuda()

                            outputs = self.generator(inputs)
                            ssim_loss = self.baseloss(outputs, labels)
                            # loss = torch.nn.L1Loss()(outputs, labels)
                            if batch % 100 == 0:
                                a = torch.squeeze(outputs[0, :, :, :]).cpu().detach().numpy()
                                b = torch.squeeze(test_label[0, :, :]).numpy()
                                outputs = np.concatenate((a, np.zeros((512, 5), dtype=np.float32), b),
                                                         axis=1) * 255.
                                outputs = np.clip(outputs, 0, 255)

                                outputs = outputs.astype(np.uint8)

                                cv2.imwrite('Logs\\result\\Unet+wgan-div\\%d_%d_%d-result.png' % (epoch, gs_g, batch),
                                            outputs)
                            batch += 1
                            # loss_test += ssim_loss.item()
                            loss_ssim_test += ssim_loss.item()

                        # writer.add_scalar('test l1_loss', loss_test / count_test, global_step=gs_g)
                        # writer.add_scalar('test ssim_loss', loss_ssim_test / count_test, global_step=gs_g)

                        print(
                            '######################################  TEST  ######################################')
                        print('Test: epoch = %s, loss = %s' % (epoch, loss_test / count_test))

