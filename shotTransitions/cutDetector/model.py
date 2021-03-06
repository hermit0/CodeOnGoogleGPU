# -*- coding:utf-8 -*-
import torch
from torch import nn

from models import x_channel_resnet,resnet,resnext,i6f_resnet

def generate_model(opt):
    assert opt.model in [
        'xcresnet','resnet','resnext','i6f_resnet'
    ]
    
    if opt.model == 'xcresnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152]
        from models.x_channel_resnet import get_fine_tuning_parameters
        
        if opt.model_depth == 10:
            model = x_channel_resnet.xcresnet10(
                num_classes=opt.n_classes,
                image_nums=opt.sample_duration)
        elif opt.model_depth == 18:
            model = x_channel_resnet.xcresnet18(
                num_classes=opt.n_classes,
                image_nums=opt.sample_duration)
        elif opt.model_depth == 34:
            model = x_channel_resnet.xcresnet34(
                num_classes=opt.n_classes,
                image_nums=opt.sample_duration)
        elif opt.model_depth == 50:
            model = x_channel_resnet.xcresnet50(
                num_classes=opt.n_classes,
                image_nums=opt.sample_duration)
        elif opt.model_depth == 101:
            model = x_channel_resnet.xcresnet101(
                num_classes=opt.n_classes,
                image_nums=opt.sample_duration)
        elif opt.model_depth == 152:
            model = x_channel_resnet.xcresnet152(
                num_classes=opt.n_classes,
                image_nums=opt.sample_duration)
    elif opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        from models.resnet import get_fine_tuning_parameters

        if opt.model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]

        from models.resnext import get_fine_tuning_parameters

        if opt.model_depth == 50:
            model = resnext.resnext50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnext.resnext101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnext.resnext152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'i6f_resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        #from models.resnet import get_fine_tuning_parameters

        if opt.model_depth == 10:
            model = i6f_resnet.i6f_resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 18:
            model = i6f_resnet.i6f_resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = i6f_resnet.i6f_resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = i6f_resnet.i6f_resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = i6f_resnet.i6f_resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = i6f_resnet.i6f_resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = i6f_resnet.i6f_resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    
    if not opt.no_cuda:
        model = model.cuda()
        #model = nn.DataParallel(model,device_ids=None)
        
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']
            
            model.load_state_dict(pretrain['state_dict'])
            
            model.fc = nn.Linear(model.fc.in_features,
                                            opt.n_finetune_classes)
            model.fc = model.fc.cuda()
            
            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            model = nn.DataParallel(model,device_ids=None)
            return model,parameters
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']
            
            model.load_state_dict(pretrain['state_dict'])
            
            modele.fc = nn.Linear(model.module.fc.in_features,
                                            opt.n_finetune_classes)
            
            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model,parameters
    model = nn.DataParallel(model,device_ids=None)
    return model, model.parameters()
                
