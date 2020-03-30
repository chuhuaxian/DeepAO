import os

f = open('D:\\Projects\\Python\\PycharmProjects\\Ao_pt\\rm_lst.txt')

rm_lst = f.readlines()

for i in rm_lst:
    line = i.strip().split(',')
    for j in line:
        # print(j)
        if os.path.exists(j):
            print('true')
            os.remove(j)