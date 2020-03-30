import os, shutil
import pyexr
import matplotlib.pyplot


base_dir = 'C:\\Users\\39796\PycharmProjects\\neterase\\Datasets\\train\s3\scene_03'

save_dir = 'C:\\Users\\39796\PycharmProjects\\neterase\Datasets\\train\scene_03'

file_list = os.listdir(base_dir)

ao_lst = [i for i in file_list if '1_View' in i or '1_view' in i ]
normal_lst = [i for i in file_list if 'normal' in i]
position_lst = [i for i in file_list if 'position' in i]
count = 0

for i in range(len(ao_lst)):
    ao_fn = os.path.join(base_dir, ao_lst[i])
    normal_fn = os.path.join(base_dir, normal_lst[i])
    position_fn = os.path.join(base_dir, position_lst[i])

    save_ao_fn = os.path.join(save_dir+'\\GroundTruth', '%04d.exr' % i)
    save_normal_fn = os.path.join(save_dir+'\\Normals', '%04d.exr' % i)
    save_position_fn = os.path.join(save_dir+'\\Position', '%04d.exr' % i)

    shutil.move(ao_fn, save_ao_fn)
    shutil.move(normal_fn, save_normal_fn)
    shutil.move(position_fn, save_position_fn)

    # print()