# -*- coding:utf-8 -*-
'''
提取视频的resNet特征，然后计算相邻帧之间的距离
'''
import cv2
from keras.applications.resnet50 import ResNet50
#from keras_squeezenet import SqueezeNet
from keras.models import Model
import time
import pdb
import numpy as np
from scipy.spatial.distance import cosine
import os

'''
提取多个视频的特征，并计算图像帧之间的距离(1-余弦相似度)序列
video_list:包含有要处理的视频文件路径的文件
layer_name:要提取的特征对应的层名
all_rates: 对视频进行采样的采样率序列
output_dir: 输出文件的存放路径
'''
def process(video_list,layer_name,all_rates,output_dir):
    with open(video_list) as fd:
        base_model = ResNet50(weights='imagenet')
        #base_model = SqueezeNet()
        model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)
        for video_file in fd:
            #pdb.set_trace()
            video_file = video_file.strip()
            video_name = video_file.split('/')[-1]
            print("Start processing video %s\n" % video_name)
            print(time.localtime(time.time()))#打印当前时间
            processVideo(video_file,layer_name,all_rates,output_dir,model)
            print(time.localtime(time.time()))#打印当前时间
'''
提取单个视频的特征，并计算图像帧之间的距离（1-余弦相似度）序列
输入参数：
video_file: 视频文件的路径
layer_name: 要提取的输出所在的层名
all_rates: 对视频进行采样的采样率序列
output_dir: 输出文件的存放路径
model:用于提取特征的网络
输出：包含距离序列的一系列文件，文件命名方式：视频名_采样率

'''
def processVideo(video_file,layer_name,all_rates,output_dir,model):
    cap = cv2.VideoCapture(video_file)
    num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    batch_size = 10
    frame_no = 0
    
    clip = []
    all_distances=[None for i in range(len(all_rates))]#用于存放不同采样率上的距离序列
    to_compare=[]#to_compare[i]表示在采样率i上要计算的帧的序号，初始均从0开始
    for i in range(len(all_rates)):
        to_compare.append(0)
    last_compare =[None for i in range(len(all_rates))]#last_compare[i]表示采样率i在前次滑动窗中的最后一帧的特征
    window_begin = 0
    window_end = 0
    while True:
        ret,frame = cap.read()
        #pdb.set_trace()
        if ret:
            img = cv2.resize(frame,(224,224))
            frame_no += 1
            window_end += 1
            clip.append(img)    
            if len(clip) == batch_size or frame_no == num_of_frames:
                #pdb.set_trace()
                inputs = np.array(clip).astype(np.float32)
                inputs[:, :, :, 0] -= 103.939
                inputs[:, :, :, 1] -= 116.779
                inputs[:, :, :, 2] -= 123.68
                print("extract features for frame %d - frame %d\n"%(window_begin,window_end-1))
                features = model.predict(inputs)#提取特征
                #pdb.set_trace()
                #计算不同采样率上的距离
                for rate_index in range(len(all_rates)):
                    frame1 = None
                    frame2 = None
                    while to_compare[rate_index] + all_rates[rate_index] < window_end:
                        frame1_no = to_compare[rate_index]
                        if frame1 is None:
                            #该滑动窗口上的第一次比较
                            if frame1_no >= window_begin:
                                if layer_name == 'avg_pool':
                                    frame1 = features[frame1_no - window_begin,:]
                                else:
                                  frame1 = features[frame1_no - window_begin,:,:,:]
                            else:
                                frame1 = last_compare[rate_index]
                        frame2_no = frame1_no + all_rates[rate_index]
                        if layer_name == 'avg_pool':
                            frame2 = features[frame2_no - window_begin,:]
                        else:
                            frame2 = features[frame2_no - window_begin,:,:,:]
                        frame1 = np.asarray(frame1).flatten()
                        frame2 = np.asarray(frame2).flatten()
                        value = cosine(frame1,frame2)
                        if all_distances[rate_index] == None :
                            all_distances[rate_index] = [(frame1_no,value)]
                        else:
                            all_distances[rate_index].append((frame1_no,value))
                        to_compare[rate_index] += all_rates[rate_index]
                        frame1 = frame2
                    if to_compare[rate_index] >= window_begin:
                        if layer_name == 'avg_pool':
                            last_compare[rate_index] = features[to_compare[rate_index] - window_begin,:]
                        else:
                            last_compare[rate_index] = features[to_compare[rate_index] - window_begin,:,:,:]
                window_begin = window_end
                window_end = window_begin
                clip.clear()
        else:
            break
    #保存distance文件
    video_name = video_file.split('/')[-1]
    for rate_index in range(len(all_rates)):
        file_name = video_name + "_" + str(all_rates[rate_index])
        file_path = os.path.join(output_dir, layer_name)
        if os.path.exists(file_path) is not True:
            os.mkdir(file_path)
        file_path = os.path.join(file_path,file_name)
        #pdb.set_trace()
        with open(file_path,'w') as fd:
            for frame_no,value in all_distances[rate_index]:
                fd.write("%10d,%f\n" %(frame_no,value))
if __name__ == '__main__':
    videos_list_file = input("Please enter the video_list that contains the videos to be processed: ")
    layer_name = input("Enter the layer name: ")
    all_rates_str = input("Enter the sample rates,seperated by ,: ")
    all_rates = []
    for rate in  all_rates_str.split(','):
      all_rates.append(int(rate))
    
    output_dir = input("Enter the output directory which distance files are saved to: ")
    process(videos_list_file,layer_name,all_rates,output_dir)