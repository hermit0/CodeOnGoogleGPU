import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import json

from utils import AverageMeter

def test(data_loader, model, opt):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    output_buffer = []
    test_results = {}
    previous_video_name = ''
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
                    if video_name not in test_results:
                        output_buffer = []
                    else:
                        output_buffer = test_results[video_name]
            output_buffer.append((frame_no,pred[j].item()))
            previous_video_name = video_name
            
        if (i % 100) == 0:
            with open(os.path.join(opt.result_path, 'predict.json'), 'w') as f:
                json.dump(test_results, f)

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
    print('Average accuracy is .3f', accuracies.avg)
    with open(os.path.join(opt.result_path, 'predict.json'), 'w') as f:
        json.dump(test_results, f)
