import numpy as np
import scipy.misc as misc
import h5py
from Models.NANO import NANO
from config import config
import torch
""" Load Network """
train_loader = None

mlp = NANO(train_loader, config).cuda()
mlp.load_state_dict(torch.load('Checkpoints\\net_params_90_2019-07-12_13-07.pkl'))
params = [i for i in mlp.named_parameters()]
print('Load success')

# network = np.load('network.npz')

W0, b0 = params[0][1].cpu().detach().numpy(), params[1][1].cpu().detach().numpy()
# print()
W1, b1 = params[4][1].cpu().detach().numpy(), params[5][1].cpu().detach().numpy()
W2, b2 = params[8][1].cpu().detach().numpy(), params[9][1].cpu().detach().numpy()
W3, b3 = params[12][1].cpu().detach().numpy(), params[13][1].cpu().detach().numpy()

alpha0 = params[2][1].cpu().detach().numpy()
beta0 = params[3][1].cpu().detach().numpy()

alpha1 = params[6][1].cpu().detach().numpy()
beta1 = params[7][1].cpu().detach().numpy()

alpha2 = params[10][1].cpu().detach().numpy()
beta2 = params[11][1].cpu().detach().numpy()

alpha3 = params[14][1].cpu().detach().numpy()
beta3 = params[15][1].cpu().detach().numpy()


# beta0 = np.ones((4,), dtype=np.float32)
# beta1 = np.ones((4,), dtype=np.float32)
# beta2 = np.ones((4,), dtype=np.float32)
# beta3 = np.ones((4,), dtype=np.float32)
#
# alpha0 = params[2][1].cpu().detach().numpy()
# alpha0 = np.concatenate((alpha0, alpha0, alpha0, alpha0))
#
# alpha1 = params[5][1].cpu().detach().numpy()
# alpha1 = np.concatenate((alpha1, alpha1, alpha1, alpha1))
#
# alpha2 = params[8][1].cpu().detach().numpy()
# alpha2 = np.concatenate((alpha2, alpha2, alpha2, alpha2))
#
# alpha3 = params[11][1].cpu().detach().numpy()
# alpha3 = np.concatenate((alpha3, alpha3, alpha3, alpha3))

f = h5py.File("Datasets/AO/database.hdf5", "r")
Ymean, Ystd = np.array(f['Ymean']), np.array(f['Ystd'])
Xmean, Xstd = np.array(f['Xmean']), np.array(f['Xstd'])
f.close()

print('')

""" Export Weights Image """

for i in range(W0.shape[1]):
    xx = W0.T[:, i]
    part = np.transpose(W0[:, i].reshape(31, 31, 4), (1, 0, 2))

    Fa, Fb = part.max(0).max(0) - part.min(0).min(0), part.min(0).min(0)
    print('static const float4 F%ia = float4(% f, % f, % f, % f);' % (i, Fa[0], Fa[1], Fa[2], Fa[3]))
    print('static const float4 F%ib = float4(% f, % f, % f, % f);' % (i, Fb[0], Fb[1], Fb[2], Fb[3]))
    part = (part - Fb) / Fa
    part = np.concatenate([part, np.zeros((1, part.shape[1], part.shape[2]))], axis=0)
    part = np.concatenate([part, np.zeros((part.shape[0], 1, part.shape[2]))], axis=1)
    misc.imsave('nnao_f%i.tga' % i, part)


print('')

""" Export Constants """

print('static const float4  Xmean = float4(% f, % f, % f, % f);' % (Xmean[0], Xmean[1], Xmean[2], Xmean[3]))
print('static const float4  Xstd  = float4(% f, % f, % f, % f);' % ( Xstd[0],  Xstd[1],  Xstd[2],  Xstd[3]))
print('static const float Ymean = % f;' % Ymean)
print('static const float Ystd  = % f;' % Ystd)
print('')

print('static const float4x4 W1 = float4x4(\n % f, % f, % f, % f,\n % f, % f, % f, % f,\n % f, % f, % f, % f,\n % f, % f, % f, % f);' % tuple(list(W1.ravel())))
print('')
print('static const float4x4 W2 = float4x4(\n % f, % f, % f, % f,\n % f, % f, % f, % f,\n % f, % f, % f, % f,\n % f, % f, % f, % f);' % tuple(list(W2.ravel())))
print('')
W3 = W3.T
print('static const float4 W3 = float4(% f, % f, % f, % f);' % (W3.T[0,0], W3.T[1,0], W3.T[2,0], W3.T[3,0]))
print('')

print('static const float4  b0 = float4(% f, % f, % f, % f);' % (b0[0], b0[1], b0[2], b0[3]))
print('static const float4  b1 = float4(% f, % f, % f, % f);' % (b1[0], b1[1], b1[2], b1[3]))
print('static const float4  b2 = float4(% f, % f, % f, % f);' % (b2[0], b2[1], b2[2], b2[3]))
print('static const float b3 = % f;' % b3[0])
print('')

print('static const float4  alpha0 = float4(% f, % f, % f, % f);' % (alpha0[0], alpha0[1], alpha0[2], alpha0[3]))
print('static const float4  alpha1 = float4(% f, % f, % f, % f);' % (alpha1[0], alpha1[1], alpha1[2], alpha1[3]))
print('static const float4  alpha2 = float4(% f, % f, % f, % f);' % (alpha2[0], alpha2[1], alpha2[2], alpha2[3]))
print('static const float alpha3 = % f;' % alpha3[0])
print('')

print('static const float4  beta0 = float4(% f, % f, % f, % f);' % (beta0[0], beta0[1], beta0[2], beta0[3]))
print('static const float4  beta1 = float4(% f, % f, % f, % f);' % (beta1[0], beta1[1], beta1[2], beta1[3]))
print('static const float4  beta2 = float4(% f, % f, % f, % f);' % (beta2[0], beta2[1], beta2[2], beta2[3]))
print('static const float beta3 = % f;' % beta3[0])
print('')