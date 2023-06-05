import os
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import shutil

data_root = 'data/'
save_path = 'annotation/'
from sklearn.model_selection import train_test_split


def FileList_Generation():
    # Update your data path below
    data_name_list = [
        # e.g., (path_to_dataset, label)
        # ('AIDraw_1122/Fake/Diffusiondb_1k', 1),  # Fake Dataset
        # ('AIDraw_1122/Real/Ctrip_real', 0),  # Real Dataset
    ]
    with open(r'/data1/snaqlov/codes/LASTED/leaf_directories.txt', 'r') as f:
    # with open(r'D:\su.q_BLJ\out\leaf_directories.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('.'):
                line = line[2:]

            if len(data_name_list) < 3 or ((len(line.split('/')) >= 3 and (line.split('/')[2] == 'afgq' or line.split('/')[2] == 'celebahq' or line.split('/')[2] == 'coco' or line.split('/')[2] == 'ffhq' or line.split('/')[2] == 'imagenet' or line.split('/')[2] == 'landscape' or line.split('/')[2] == 'lsun' or line.split('/')[2] == 'metfaces'))):
                data_name_list.append((line, 0))
            else:
                data_name_list.append((line, 1))



    img_list = []
    for data_name, label in data_name_list:
        if label == 0:
            continue
        path = '%s/' % data_name
        flist = sorted(os.listdir(data_root + path))
        for file in tqdm(flist):
            img_list.append((path + file, label))
    img_list2 = []
    for data_name, label in data_name_list:
        if label == 1:
            continue
        path = '%s/' % data_name
        flist = sorted(os.listdir(data_root + path))
        for file in tqdm(flist):
            img_list2.append((path + file, label))
    img_list = img_list + img_list2

    print('#Images: %d' % len(img_list))

    # Split the data into train and test sets with a 9:1 ratio
    temp_list, test_list = train_test_split(img_list, test_size=0.1, random_state=42)
    train_list, val_list = train_test_split(temp_list, test_size=0.005, random_state=42)
    train_list, val_list, test_list = list(train_list), list(val_list), list(test_list)

    # 打印数据集划分的信息
    print(f'Total: {len(img_list)}, Train: {len(train_list)}, Val: {len(val_list)}, Test: {len(test_list)}')

    # Write the train and test sets to separate text files
    with open(save_path + 'datasetv1_train_num%d.txt' % (len(train_list)), 'w') as train_file:
        for item in train_list:
            train_file.write('%s %s\n' % (item[0], item[1]))

    with open(save_path + 'datasetv1_val_num%d.txt' % (len(val_list)), 'w') as val_file:
        for item in val_list:
            val_file.write('%s %s\n' % (item[0], item[1]))


    with open(save_path + 'datasetv1_test_num%d.txt' % (len(test_list)), 'w') as test_file:
        for item in test_list:
            test_file.write('%s %s\n' % (item[0], item[1]))


if __name__ == '__main__':
    # generate file list for training/testing
    # E.g., Train_VISION.txt contains [[image_path_1, image_label_1], [image_path_2, image_label_2], ...]
    FileList_Generation()


