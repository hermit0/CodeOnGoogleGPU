# -*- coding:utf-8 -*-
'''
将train中的视频划分为训练集和验证集
输入train.txt:包含训练集中所有视频名称的文本
输出：train_set.txt-新的训练集视频列表文件
    val.txt-新的验证集视频列表文件
'''

import numpy as np

def split_train_val(video_list,ratio):
    #设置随机数种子，保证每次生成的结果都是一样的
    np.random.seed(40)
    #permutation随机生成0-len(video_list)随机序列
    shuffled_indices = np.random.permutation(len(video_list))
    val_set_size = int(len(video_list) * ratio)
    val_indices = shuffled_indices[:val_set_size]
    train_indices = shuffled_indices[val_set_size:]
    #保存train_set.txt
    with open('train_set.txt','w') as train_fd:
        for index in train_indices:
            video_name = video_list[index]
            train_fd.write(video_name)
            train_fd.write('\n')
    #保存val.txt
    with open('val.txt','w') as val_fd:
        for index in val_indices:
            video_name = video_list[index]
            val_fd.write(video_name)
            val_fd.write('\n')

def split(video_list_path,ratio):
    video_list=[]
    with open(video_list_path) as fd:
        for line in fd:
            video_name = line.strip()
            video_list.append(video_name)
    split_train_val(video_list,ratio)
    
if __name__ == '__main__':
    video_list_path = input('Enter the path of video list file: ')
    ratio = input('Enter the ratio of validation set: ')
    ratio = float(ratio)
    split(video_list_path,ratio)