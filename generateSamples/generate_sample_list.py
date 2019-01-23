# -*- coding:utf-8 -*-
'''
根据给定的视频列表，标记文件，以及candidates 文件生成用于切变检测的样本列表
生成的样本列表的格式： 视频名 center_frame label(0-非切变 1-切变)  当label=1时，center_frame为前一个镜头的边界
'''
import numpy as np
from numpy import random
import os
import json
import pdb
import argparse

#生成训练用样本
#训练样本集中需要考虑正负样本的均衡，因此只根据candidates选择部分的负样本
def generate_train_samples(video_list_path, gt_path, candidates_path,output_file):
    video_list=[]
    gts = json.load(open(gt_path))
        
    with open(video_list_path,'r') as fd:
        for line in fd:
            video_name = line.strip()
            if video_name != '':
                video_list.append(video_name)
    all_samples = []
    for video in video_list:
        candidates_file = video + '_candidates'
        out_samples = []
        if video in gts:
            cuts = [(begin,end) for begin,end in gts[video]['transitions'] if end-begin==1]
            graduals = [(begin,end) for begin,end in gts[video]['transitions'] if end-begin>1]
            negative_candidates = []
            non_count = 0
            gra_count = 0
            with open(os.path.join(candidates_path,candidates_file)) as fd:
                cut_index = 0
                gradual_index = 0
                for line in fd:
                    frame_no = int(line.strip())
                    label = 0#non transition
                                 
                    
                    #找到第一个开始帧大于等于candidate的cut
                    while cut_index < len(cuts) and cuts[cut_index][0] < frame_no:
                        cut_index += 1
                    if cut_index < len(cuts) and frame_no == cuts[cut_index][0]:
                        label = 1 #切变
                
                    #找到第一个结束帧大于等于candidate的gradual
                    while gradual_index < len(graduals) and graduals[gradual_index][1] < frame_no:
                        gradual_index +=1
                    if gradual_index < len(graduals) and frame_no >= graduals[gradual_index][0]:
                        label = 2 #渐变
                    if label != 1:
                        negative_candidates.append((frame_no,label))
                    if label == 0:
                        non_count += 1
                    if label == 2:
                        gra_count += 1
                    
            weights=[]
            base_prob = 1.0   #基本概率
            if non_count != 0 and gra_count != 0:
                base_prob = 0.5
            for _,label in negative_candidates:
                if label == 0:
                    #pdb.set_trace()
                    prob = base_prob / non_count
                    weights.append(prob)
                if label == 2:
                    prob = base_prob / gra_count
                    weights.append(prob)

            #获得正样本
            for begin,_ in cuts:
                sample = (video, begin, 1)
                out_samples.append(sample)
            #negative_num = min(2 * len(cuts),len(negative_candidates))
            negative_num = int(min(0.5 * len(cuts),len(negative_candidates)))#正负样本比例为2:1
            #pdb.set_trace()
            if negative_num == 0:
                negative_num = min(len(negative_candidates),10)
            if negative_num > 0:    
                for index in random.choice(len(negative_candidates),negative_num,replace=False,p=weights):
                    frame_no,_=negative_candidates[index]
                    sample = (video,frame_no,0)
                    out_samples.append(sample)
            random.shuffle(out_samples)
            if len(out_samples) > 0:
                all_samples.extend(out_samples)
    #保存所有的样本
    save_to_file(all_samples,output_file)
'''
保存样本集
输入：列表形式的样本集, 输出文件名
'''
def save_to_file(samples, file_path):
    fd = open(file_path,'w')
    for video_name,begin,label in samples:
        line = '{} {} {}\n'.format(video_name,begin,label)
        fd.write(line)
    fd.close()

'''
生成验证用样本集
验证样本集需要根据candidates文件中的候选帧，生成所有需要检测的样本
'''
def generate_test_samples(video_list_path,gt_path,candidates_path,output_file):
    video_list = []
    gts = json.load(open(gt_path))
    with open(video_list_path,'r') as fd:
        for line in fd:
            video_name = line.strip()
            if video_name != '':
                video_list.append(video_name)
    all_samples = []
    count = 0
    
    for video in video_list:
        candidates_file = video + '_candidates'
        if video in gts:
            cuts = [(begin,end) for begin,end in gts[video]['transitions'] if end-begin==1]
            #graduals = [(begin,end) for begin,end in gts[video]['transitions'] if end-begin>1]
            cut_index = 0
            with open(os.path.join(candidates_path,candidates_file)) as fd:
                for line in fd:
                    count += 1
                    frame_no = int(line.strip())
                    label = 0#non cut transition
                    #找到第一个开始帧大于等于candidate的cut
                    while cut_index < len(cuts) and cuts[cut_index][0] < frame_no:
                        cut_index += 1
                    if cut_index < len(cuts) and frame_no == cuts[cut_index][0]:
                        label = 1 #切变
                    sample = (video,frame_no,label)
                    all_samples.append(sample)
    print('total candidates : %d' % count)
    save_to_file(all_samples,output_file)

'''
从原始的验证样本集中进行采样，生成实际用于验证学习效果的样本集
因为原始的验证样本集过大，因此对其进行随机采样，以获得一个合适大小的样本集
正样本保留，负样本和正样本的比例控制在4：1
'''
def sample_from_val(sample_list_file):
    fd = open(sample_list_file)
    raw_samples=[]
    positive_count = 0
    final_samples=[]
    for line in fd:
        line = line.strip()
        line = line.split(' ')
        video_name = line[0]
        frame_no = int(line[1])
        label = int(line[2])
        raw_samples.append((video_name,frame_no,label))
        if label == 1:
            positive_count += 1
            final_samples.append((video_name,frame_no,label))
    fd.close()
    negative_num = 4 * positive_count
    prob = 1.0 / (len(raw_samples) - positive_count)
    weights = []
    for _,_,label in raw_samples:
        if label == 1:
            weights.append(0)
        else:
            weights.append(prob)

    for index in np.random.choice(len(raw_samples),negative_num,replace=False,p=weights):
        final_samples.append(raw_samples[index])
    final_samples.sort()
    print('the size of new validation set is :%d\n' % len(final_samples))
    save_to_file(final_samples,sample_list_file)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_list_path', required=True, type=str, help='the path of video list file')
    parser.add_argument('--gt_path', required=True, type=str, help='the path of ground truth json file')
    parser.add_argument('--candidates_dir', required=True, type=str, help='Root directory path of candidates data')
    parser.add_argument('--output_file', required=True, type=str, help='the path of output file')
    parser.add_argument('--test',action='store_true')
    args = parser.parse_args()
    video_list_path = args.video_list_path
    gt_path = args.gt_path
    candidates_dir = args.candidates_dir
    output_file = args.output_file
    if args.test:
        generate_test_samples(video_list_path,gt_path,candidates_dir,output_file)
    else:
        generate_train_samples(video_list_path,gt_path,candidates_dir,output_file)