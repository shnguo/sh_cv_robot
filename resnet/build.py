import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from resnet.net import BasicBlock, Bottleneck, ResNet

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def resnet18(pretrained=False, progress=True, num_classes=1000):
    model = ResNet(18, 1000)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet18'], model_dir='./weights',
                                              progress=progress)
        model.load_state_dict(state_dict)

    if num_classes!=1000:
        model.fc = nn.Linear(512 * 1, num_classes)
    return model

def resnet34(pretrained=False, progress=True, num_classes=1000):
    model = ResNet(34, 1000)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet34'], model_dir='./weights',
                                              progress=progress)
        model.load_state_dict(state_dict)

    if num_classes!=1000:
        model.fc = nn.Linear(512 * 1, num_classes)
    return model

def resnet50(pretrained=False, progress=True, num_classes=1000):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50'], model_dir='./weights',
                                              progress=progress)
        model.load_state_dict(state_dict)

    if num_classes!=1000:
        model.fc = nn.Linear(512 * 4, num_classes)
    return model

def resnet101(pretrained=False, progress=True, num_classes=1000):
    model = ResNet(101, 1000)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet101'], model_dir='./weights',
                                              progress=progress)
        model.load_state_dict(state_dict)

    if num_classes!=1000:
        model.fc = nn.Linear(512 * 4, num_classes)
    return model

def resnet152(pretrained=False, progress=True, num_classes=1000):
    model = ResNet(152, 1000)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet152'], model_dir='./weights',
                                              progress=progress)
        model.load_state_dict(state_dict)

    if num_classes!=1000:
        model.fc = nn.Linear(512 * 4, num_classes)
    return model