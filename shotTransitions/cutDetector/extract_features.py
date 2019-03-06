#-*- coding:utf-8 -*-
#提取特征的同时并预测结果
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import json

from utils import AverageMeter

global inter_feature

def get_features(m,input,output):
    global inter_feature
    inter_feature = output.cpu().clone()
    return None
def extract(data_loader, model, opt):
    print('extracting feature')

    model.eval()
    #for name,module in model.named_modules():
    #    print(name)
    #extracted_layer = model.avgpool #要提取输出的层的完整名称
    extracted_layer = model.fc
    extracted_layer.register_forward_hook(get_features)
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    output_buffer = []  #用于存放预测结果
    test_results = {}
    previous_video_name = ''
    output_features = []#存放的每一项的格式为(gt_label,特征张量) 
    global inter_feature
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        with torch.no_grad():
            inputs = Variable(inputs)
            if not opt.no_cuda:
                inputs = inputs.cuda()
            outputs = model(inputs)
            outputs = F.softmax(outputs,dim=1)
            _, pred = outputs.topk(1, 1, True)    
            pred = torch.squeeze(pred)
            features = torch.chunk(inter_feature,outputs.size(0),dim=0)
            #print(len(features))
            #print(outputs.size(0))
        for j in range(outputs.size(0)):
            video_name = os.path.basename(targets['video_path'][j])
            frame_no = targets['frame_no'][j].item()
            annotation = targets['label'][j]
            if annotation != -1:
                if annotation == pred[j].item():
                    accuracies.update(1,1)
                else:
                    accuracies.update(0,1)
            if video_name != previous_video_name:
                if len(output_buffer) > 0:
                    test_results[previous_video_name] = output_buffer
                    output_buffer = []
            output_buffer.append((frame_no,pred[j].item()))
            previous_video_name = video_name
            
            #生成用于svm的特征 正样本的标签为+1,负样本的标签为-1
            gt_label = 3 #一个大于所有标签的数字，表示标签位置
            if annotation == 0:
                gt_label = -1
            if annotation == 1:
                gt_label = 1
            #print(features[j].size())
            output_features.append((gt_label,features[j]))
            
        if (i % 100) == 0:
            with open(os.path.join(opt.result_path, 'predict.json'), 'w') as f:
                json.dump(test_results, f)
            #保存特征文件，保存成svm所需要的格式
            #格式 0 1:1.679500e+01 2:4.695801e+01 3:9.103831e-02 4:4.400916e+01
            #with open(os.path.join(opt.result_path,opt.output_feature_path), 'w') as f:
            #    for label,feature in output_features:
            #        line = '{:+d}'.format(label)
            #        feature_size = feature.size()
            #        feature_dims = 1
            #        for k in range(len(feature_size)):
            #            feature_dims *= feature_size[k]
                    #print(feature_dims)
            #        feature = feature.view(feature_dims)
            #        index = 1
            #        for value in feature.tolist():
            #            line += ' {index:d}:{value:.8f}'.format(index=index,value=value)
            #            index += 1
            #        f.write(line)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('[{}/{}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time))
    if previous_video_name != '':
        if len(output_buffer) > 0:
            test_results[previous_video_name] = output_buffer
            output_buffer = []
    print('Average accuracy is %.3f', accuracies.avg)
    with open(os.path.join(opt.result_path, 'predict.json'), 'w') as f:
        json.dump(test_results, f)
    with open(os.path.join(opt.result_path,opt.output_feature_path), 'w') as f:
                for label,feature in output_features:
                    line = '{:+d}'.format(label)
                    feature_size = feature.size()
                    feature_dims = 1
                    for k in range(len(feature_size)):
                        feature_dims *= feature_size[k]
                    feature = feature.view(feature_dims)
                    index = 1
                    for value in feature.tolist():
                        line += ' {index:d}:{value:.8f}'.format(index=index,value=value)
                        index += 1
                    line += '\n'
                    f.write(line)
