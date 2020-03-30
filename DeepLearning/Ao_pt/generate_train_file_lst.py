import os
from sklearn.model_selection import train_test_split
import pyexr
import matplotlib.pyplot as plt
import numpy as np

def generate_file_list(data_dir):
    lst = [[os.path.join(os.path.join(data_dir, i), j) for j in os.listdir(os.path.join(data_dir, i))] for i in os.listdir(data_dir)]

    file_lst = []
    # remove_lst = []
    # lst = lst[6:]
    # print()
    for i in range(len(lst)):

        ao_lst = [i for i in lst[i] if 'normal' not in i and 'Z D' not in i]
        normal_lst = [i for i in lst[i] if 'normal' in i]
        position_lst = [i for i in lst[i] if 'Z D' in i]
        print(len(ao_lst), len(normal_lst), len(position_lst))
        # 2400 1800 3500 3600

        for j in range(len(ao_lst)):
            ao_fn = ao_lst[j]
            normal_fn = normal_lst[j]
            position_fn = position_lst[j]

            # depth = pyexr.open(position_fn).get()
            # if 'scene09' not in ao_fn:
            #     continue
            # ao = pyexr.open(ao_fn).get()

            our_str = '%s,%s,%s\n' % (ao_fn, normal_fn, position_fn)

            # if np.mean(ao[:, :, 0]) <=1e-5:
            #     print(our_str)

            # if np.min(depth) < 1e-5:
            #     plt.subplot(121)
            #     plt.imshow(depth)
            #     plt.subplot(122)
            #     plt.imshow(ao)
            #     plt.show()
            #     print(our_str)
            file_lst.append(our_str)
# generate_file_list('D:\\Projects\Datasets\\3dsmax\scene')

# print()
        # for j in range(len(ao_lst)):
        #     our_str = ''
        #     for k in range(3, -1, -1):
        #         index = j - k if j // 100 * 100 == (j - k) // 100 * 100 else j // 100 * 100
        #         ao_fn = ao_lst[index]
        #         normal_fn = normal_lst[index]
        #         position_fn = position_lst[index]
        #         our_str += '%s,%s,%s\n' % (ao_fn, normal_fn, position_fn)
        #     file_lst.append(our_str)

    # train, test = train_test_split(file_lst, test_size=0.2, random_state=42)
    train, test = file_lst[:-7176], file_lst[-7176:]
    f = open('train_lst.txt', mode='w')
    for i in train:
        f.write(i)
    f.close()
    f = open('test_lst.txt', mode='w')
    for i in test:
        f.write(i)
    f.close()


def generate_deepshading_file_list(data_dir):

    train_dir, test_dir = os.path.join(data_dir, 'Train'), os.path.join(data_dir, 'Test')

    train_lst = [[os.path.join(os.path.join(train_dir, i), j) for j in os.listdir(os.path.join(train_dir, i))] for i in
                 os.listdir(train_dir)]

    test_lst = [[os.path.join(os.path.join(test_dir, i), j) for j in os.listdir(os.path.join(test_dir, i))] for i in
                 os.listdir(test_dir)]

    file_lst = []
    for i in train_lst:

        ao_lst = [os.path.join(i[0], j) for j in os.listdir(i[0])]
        normal_lst = [os.path.join(i[1], j) for j in os.listdir(i[1])]
        position_lst = [os.path.join(i[2], j) for j in os.listdir(i[2])]

        for j in range(len(ao_lst)):
            ao_fn = ao_lst[j]
            normal_fn = normal_lst[j]
            position_fn = position_lst[j]
            our_str = '%s,%s,%s\n' % (ao_fn, normal_fn, position_fn)
            file_lst.append(our_str)

    test_file_lst = []
    for i in test_lst:
        ao_lst = [os.path.join(i[0], j) for j in os.listdir(i[0])]
        normal_lst = [os.path.join(i[1], j) for j in os.listdir(i[1])]
        position_lst = [os.path.join(i[2], j) for j in os.listdir(i[2])]

        for j in range(len(ao_lst)):
            ao_fn = ao_lst[j]
            normal_fn = normal_lst[j]
            position_fn = position_lst[j]
            our_str = '%s,%s,%s\n' % (ao_fn, normal_fn, position_fn)
            test_file_lst.append(our_str)
    # print()
    #
    # train, test = train_test_split(file_lst, test_size=0.2, random_state=42)
    f = open('train_lst.txt', mode='w')
    for i in file_lst:
        f.write(i)
    f.close()
    f = open('test_lst.txt', mode='w')
    for i in test_file_lst:
        f.write(i)
    f.close()


def generate_nnao_file_list(data_dir):

    train_dir, test_dir = os.path.join(data_dir, 'Train'), os.path.join(data_dir, 'Test')

    train_lst = [[os.path.join(os.path.join(train_dir, i), j) for j in os.listdir(os.path.join(train_dir, i))] for i in
                 os.listdir(train_dir)]

    test_lst = [[os.path.join(os.path.join(test_dir, i), j) for j in os.listdir(os.path.join(test_dir, i))] for i in
                 os.listdir(test_dir)]

    file_lst = []
    for i in train_lst:

        ao_lst = [os.path.join(i[0], j) for j in os.listdir(i[0])]
        normal_lst = [os.path.join(i[1], j) for j in os.listdir(i[1])]
        position_lst = [os.path.join(i[2], j) for j in os.listdir(i[2])]

        for j in range(len(ao_lst)):
            ao_fn = ao_lst[j]
            normal_fn = normal_lst[j]
            position_fn = position_lst[j]
            our_str = '%s,%s,%s\n' % (ao_fn, normal_fn, position_fn)
            file_lst.append(our_str)

    test_file_lst = []
    for i in test_lst:
        ao_lst = [os.path.join(i[0], j) for j in os.listdir(i[0])]
        normal_lst = [os.path.join(i[2], j) for j in os.listdir(i[2])]
        position_lst = [os.path.join(i[3], j) for j in os.listdir(i[3])]

        for j in range(len(ao_lst)):
            ao_fn = ao_lst[j]
            normal_fn = normal_lst[j]
            position_fn = position_lst[j]
            our_str = '%s,%s,%s\n' % (ao_fn, normal_fn, position_fn)
            test_file_lst.append(our_str)
    # print()
    #
    # train, test = train_test_split(file_lst, test_size=0.2, random_state=42)
    f = open('train_lst.txt', mode='w')
    for i in file_lst:
        f.write(i)
    f.close()
    f = open('test_lst.txt', mode='w')
    for i in test_file_lst:
        f.write(i)
    f.close()


def generate_lstm_file_list(data_dir):
    lst = [[os.path.join(os.path.join(data_dir, i), j) for j in os.listdir(os.path.join(data_dir, i))] for i in os.listdir(data_dir)]

    file_lst = []
    for i in range(len(lst)):

        ao_lst = [i for i in lst[i] if 'normal' not in i and 'Z D' not in i]
        normal_lst = [i for i in lst[i] if 'normal' in i]
        position_lst = [i for i in lst[i] if 'Z D' in i]
        print(len(ao_lst), len(normal_lst), len(position_lst))
        # 2400 1800 3500 3600

        # if len(ao_lst) == 1800:
        for j in range(len(ao_lst)):
            our_str = ''
            for k in range(3, -1, -1):
                index = j-k if  j//100*100 == (j-k)//100*100 else j//100*100
                ao_fn = ao_lst[index]
                normal_fn = normal_lst[index]
                position_fn = position_lst[index]
                our_str += '%s,%s,%s,' % (ao_fn, normal_fn, position_fn)
            our_str = our_str[:-1]
            our_str += '\n'
            file_lst.append(our_str)

        # elif len(ao_lst) == 2400:
        #     for j in range(len(ao_lst)):
        #         ao_fn = ao_lst[j]
        #         normal_fn = normal_lst[j]
        #         position_fn = position_lst[j]
        #         our_str = '%s,%s,%s\n' % (ao_fn, normal_fn, position_fn)
        #         file_lst.append(our_str)
        # elif len(ao_lst) == 4800:
        #     for j in range(len(ao_lst)):
        #         ao_fn = ao_lst[j]
        #         normal_fn = normal_lst[j]
        #         position_fn = position_lst[j]
        #         our_str = '%s,%s,%s\n' % (ao_fn, normal_fn, position_fn)
        #         file_lst.append(our_str)
        # else:
        #     for j in range(len(ao_lst)):
        #         ao_fn = ao_lst[j]
        #         normal_fn = normal_lst[j]
        #         position_fn = position_lst[j]
        #         our_str = '%s,%s,%s\n' % (ao_fn, normal_fn, position_fn)
        #         file_lst.append(our_str)

    train, test = train_test_split(file_lst, test_size=1000, random_state=42)
    f = open('train_lst.txt', mode='w')
    for i in train:
        f.write(i)
    f.close()
    f = open('test_lst.txt', mode='w')
    for i in test:
        f.write(i)
    f.close()

