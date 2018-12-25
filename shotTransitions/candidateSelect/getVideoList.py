# -*- coding: UTF-8 -*-
'''根据当前目录下的文件名获取video list
'''
import os

def get_video_list(dir_name):
    video_list = []
    for file in os.listdir(dir_name):
        if os.path.isfile(os.path.join(dir_name,file)):
            pos = file.find('.mp4')
            if pos != -1:
                video_name = file[:pos+4]
                if video_name not in video_list:
                    video_list.append(video_name)
    return video_list

def output_video_list(video_list):
    fd = open('video_list','w')
    for video_name in video_list:
        fd.write('%s\n'%video_name)
    fd.close()

if __name__ == '__main__':
    dir_name = input('enter the directory path that is used to extract video list: ')
    
    output_video_list(get_video_list(dir_name))
