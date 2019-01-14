import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import json


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

        inputs = Variable(inputs, volatile=True)
        outputs = model(inputs)
        outputs = F.softmax(outputs)
        _, pred = outputs.topk(1, 1, True)    

        for j in range(outputs.size(0)):
            video_name = os.path.basename(targets[j]['video_path'])
            frame_no = targets[j]['frame_no']
            annotation = targets[j]['label']
            if annotation != -1:
                if annotation == pred[j].data:
                    accuracies.update(1,1)
                else:
                    accuracies.update(0,1)
            if video_name != previous_video_name:
                if len(output_buffer) > 0:
                    test_results[previous_video_name] = output_buffer
                    output_buffer = []
            output_buffer.append((frame_no,pred[j].data.cpu()))
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