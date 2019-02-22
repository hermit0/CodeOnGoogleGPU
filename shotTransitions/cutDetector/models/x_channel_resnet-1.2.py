import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo



__all__ = ['xcResNet', 'xcresnet18', 'xcresnet34', 'xcresnet50', 'xcresnet101',
           'xcresnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
'''
将输入的张量(N,C,T,H,W)转换为(N,C*T,H,W)
'''
class imgConcat(nn.Module):
    def __init__(self):
        super(imgConcat, self).__init__()
    
    def forward(self, x):
        #print('test')
        #print(x.size())
        if len(x.size()) == 5:
            batch_size = x.size()[0]
            img_h = x.size()[-2]
            img_w = x.size()[-1]
            x = x.view(batch_size,-1,img_h,img_w)
            #print(x.size())
        return x
'''
将输入的张量(N,C,T,H,W)通过时间上的卷积转换为(N,C*T,H,W)
'''
class imgConcat_new(nn.Module):
    def __init__(self,image_nums):
        super(imgConcat_new, self).__init__()
        self.conv = nn.Conv3d(3,3*image_nums,kernel_size=(image_nums,1,1),stride=1,padding=0,bias=False)
        self.bn = nn.BatchNorm2d(3*image_nums)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        if len(x.size()) == 5:
            x = self.conv(x)
            batch_size = x.size()[0]
            img_h = x.size()[-2]
            img_w = x.size()[-1]
            channel_size = x.size()[1]
            depth = x.size()[2]
            x = x.view(batch_size,-1,img_h,img_w)
            x = self.bn(x)
            x = self.relu(x)
        return x

'''
在ResNet的基础上修改，增加image concatenation层，将多张图像视为多通道图像
'''
class xcResNet(nn.Module):

    def __init__(self, block, layers, image_nums,num_classes=1000, zero_init_residual=False):
        super(xcResNet, self).__init__()
        self.inplanes = 64
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        #self.concat = imgConcat()
        self.concat = imgConcat_new(image_nums)
        self.conv1 = nn.Conv2d(3*image_nums,64,kernel_size=7,stride=2,padding=3,bias=False) 
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.concat(x)
        if isinstance(next(self.conv1.parameters()),torch.cuda.FloatTensor):
            x = x.cuda()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    def load_weights(self,url):
        pretrain_state_dict = model_zoo.load_url(url)
        current_param = self.state_dict()
        pretrained={k:v for k,v in pretrain_state_dict.items() if k in current_param and k.split('.')[0]!='fc' and k.split('.')[0]!='conv1'}
        current_param.update(pretrained)
        print(pretrained.keys())
        self.load_state_dict(current_param)
        print('Finished load pretained weights!')

def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters
def xcresnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = xcResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_weights(model_urls['resnet18'])
    return model


def xcresnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = xcResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_weights(model_urls['resnet34'])
    return model


def xcresnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = xcResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_weights(model_urls['resnet50'])
    return model


def xcresnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = xcResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_weights(model_urls['resnet101'])
    return model


def xcresnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = xcResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_weights(model_urls['resnet152'])
    return model
