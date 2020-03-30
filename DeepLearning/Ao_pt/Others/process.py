import sys
import numpy as np

rng = np.random.RandomState(23455)

clip_near = 0.1
clip_far = 1000.0

cam_inv_proj = np.array([
    [-1.398332, -0.000000, 0.000000, -0.000000],
    [-0.000000, 0.786562, -0.000000, 0.000000],
    [0.000000, -0.000000, 0.000000, -1.000000],
    [-0.000000, 0.000000, -4.999500, 5.000499]])

cam_proj = np.array([
    [-0.715138, 0.000000, -0.000000, 0.000000],
    [0.000000, 1.271356, 0.000000, 0.000000],
    [0.000000, 0.000000, -1.000200, -0.200020],
    [0.000000, 0.000000, -1.000000, 0.000000]])


def perspective_depth(d, n, f):
    return -((2 * n) / d - f - n) / (f - n)


def camera_space(D):
    D = perspective_depth(1 - D, clip_near, clip_far) * 2.0 - 1.0
    U = np.empty((D.shape[0], D.shape[1], 2))
    U[:, :, 0] = (((np.arange(D.shape[0]) + 0.5) / D.shape[0]) * 2.0 - 1.0)[:, np.newaxis]
    U[:, :, 1] = (((np.arange(D.shape[1]) + 0.5) / D.shape[1]) * 2.0 - 1.0)[np.newaxis, :]
    P = np.concatenate([U[:, :, 0:1], U[:, :, 1:2], D, np.ones(D.shape)], axis=-1)
    P = cam_inv_proj.dot(P.reshape(-1, 4).T).T.reshape(P.shape)
    return P[:, :, :3] / P[:, :, 3:4]


nsamples = 1024
fw = 31
radius = 1.0
hw = int(((fw - 1) / 2))
coords = np.array([[x, y, 0] for x in range(-hw, hw + 1) for y in range(-hw, hw + 1)], dtype=np.float) / hw

print()
databases = [
    ('mesa_props', 100)
]

total_images = sum([nimages for (name, nimages) in databases])

X = np.empty((nsamples * total_images, fw, fw, 4), dtype=np.float32)
Y = np.empty((nsamples * total_images, 1), dtype=np.float32)

ii = 0

res = []
for name, nimages in databases:

    print('Processing Database "%s\n"' % name)

    for i in range(1, nimages + 1):

        sys.stdout.write('\rProcessing Image %i of %i\n' % (i, nimages))

        S = np.fromfile('Datasets/data/raw/' + name + '/AO/' + name + '.' + str(i) + '.bin', dtype=np.float32).reshape(720,
                                                                                                                1280, 1)
        N = np.fromfile('Datasets/data/raw/' + name + '/normalCam/' + name + '.' + str(i) + '.bin', dtype=np.float32).reshape(
            720, 1280, 4)
        # out = np.concatenate((N, S), axis=-1)
        #
        # res.append(out[:360, :640, :])
        # res.append(out[360:, :640, :])
        # res.append(out[:360, 640:, :])
        # res.append(out[360:, 640:, :])
        # print()

# res = np.array(res)
# print('start')
# np.save('Datasets/dataset_full.npy', res)

        # plt.subplot(131), plt.imshow(N[:, :, :3])
        # plt.subplot(132), plt.imshow(N[:, :, 3], cmap='gray')
        # plt.subplot(133), plt.imshow(S[:, :, 0], cmap='gray')
        # plt.show()

#         print(np.max(N[:, :, 3]))
        N, D = N[:, :, :3], N[:, :, 3:]



        S, N, D = np.swapaxes(S, 0, 1), np.swapaxes(N, 0, 1), np.swapaxes(D, 0, 1)
        W = camera_space(D)  # depth 转换成相机空间坐标 (1280, 720, 3)

        for _ in range(nsamples):
            x, y = rng.randint(S.shape[0]), rng.randint(S.shape[1])

            nexts = W[x, y] + radius * coords  # (961, 3)
            nexts = np.concatenate([nexts, np.ones((len(coords), 1))], axis=-1)  # coords(961, 3) nexts (961, 4)
            nexts = cam_proj.dot(nexts.T).T  # nexts (961, 4)
            nexts = (nexts[:, :2] / nexts[:, 3:]) * 0.5 + 0.5

            indices = (
                np.clip(nexts[:, 0] * W.shape[0], 0, W.shape[0] - 1).astype(np.int),
                np.clip(nexts[:, 1] * W.shape[1], 0, W.shape[1] - 1).astype(np.int))

            length = np.sqrt(np.sum((W[indices] - W[x, y]) ** 2, axis=-1))[..., np.newaxis]  # (961, 1)
            # // 距离中心点的距离
            dists = 1.0 - np.clip(length / radius, 0, 1)  # (961, 1)  ==> X

            X[ii] = np.concatenate([
                (N[indices] - N[x, y]) * dists,
                (W[indices] - W[x, y])[:, 2:] * dists,
            ], axis=-1).reshape(fw, fw, 4)

            # if np.sum(X[0, :, :, :3]) > 0.01:
            #     plt.subplot(131), plt.imshow(X[0, :, :, :3])
            #     plt.subplot(132), plt.imshow(X[0, :, :, 3], cmap='gray')
            #     plt.show()
            # xxx = X[0, :, :, :3]
            # print(np.sum(X[0, :, :, :3]))
            Y[ii] = S[x:x + 1, y:y + 1]

            ii += 1
#
#     print('')
#
# print(X.shape, Y.shape, ii)
#
# Ymean = 0.0
# Ystd = Y.reshape(-1, Y.shape[-1])[::100].std(axis=0, dtype=np.float64).astype(np.float32)
# print(Ymean, Ystd)
#
# Xmean = X.reshape(-1, X.shape[-1])[::100].mean(axis=0, dtype=np.float64).astype(np.float32)
# Xstd = X.reshape(-1, X.shape[-1])[::100].std(axis=0, dtype=np.float64).astype(np.float32)
# print(Xmean, Xstd)

# f = h5py.File('database.hdf5', 'w')
# f.create_dataset('X', data=X)
# f.create_dataset('Y', data=Y)
# f.create_dataset('Xmean', data=Xmean)
# f.create_dataset('Xstd', data=Xstd)
# f.create_dataset('Ymean', data=Ymean)
# f.create_dataset('Ystd', data=Ystd)
# f.close()
