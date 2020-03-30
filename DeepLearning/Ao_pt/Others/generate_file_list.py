import os
import pyexr
import matplotlib.pyplot as plt


base_dir = 'Datastes\\Unity_sceenshot'
# D:\Projects\Pycharm\Ao_pt\Datastes\Unity_sceenshot
lst = [os.path.join(base_dir, i) for i in os.listdir(base_dir)]

# ao_lst = [os.path.join(i, 'GroundTruth') for i in lst]
# normal_lst = [os.path.join(i, 'Normals') for i in lst]

ao_lst = [[os.path.join(os.path.join(i, 'GroundTruth'), j) for j in os.listdir(os.path.join(i, 'GroundTruth'))] for i in lst]
normal_lst = [[os.path.join(os.path.join(i, 'Normals'), j) for j in os.listdir(os.path.join(i, 'Normals'))] for i in lst]
position_lst = [[os.path.join(os.path.join(i, 'Position'), j) for j in os.listdir(os.path.join(i, 'Position'))] for i in lst]


f = open('C:\\Users\\39796\\PycharmProjects\\neterase\Datasets\\train_lst.txt', mode='w')
count = 0
for i in range(len(ao_lst)):
    for j in range(len(ao_lst[i])):

        # print(j)
        # print(ao_lst[i][j])
        ao_fn = ao_lst[i][j]
        normal_fn = normal_lst[i][j]
        position_fn = position_lst[i][j]

        our_str = '%s,%s,%s\n' % (ao_fn, normal_fn, position_fn)
        f.write(our_str)
        # print(our_str)
        # print()
        # ao = pyexr.open(ao_fn).get()
        # normal = pyexr.open(ao_fn).get()
        # position = pyexr.open(ao_fn).get()
        #
        # plt.subplot(131)
        # plt.imshow(ao)
        # plt.subplot(132)
        # plt.imshow(normal)
        # plt.subplot(133)
        # plt.imshow(position[:, :, 2])
        # plt.show()
        # print()
f.close()
# print(position_lst)
