import copy
import torch
from torch import nn
# from convs.cifar_resnet import resnet32
# from convs.resnet import resnet18, resnet34, resnet50
# from convs.linears import SimpleContinualLinear
from vits import vit_base_patch16_224_in21k, vit_base_patch16_224_mocov3, vit_base_lora_patch16_224_in21k, vit_base_lora_patch16_224_mocov3
import torch.nn.functional as F

def get_convnet(cfg, pretrained=False):
    name = cfg['convnet_type']
    name = name.lower()
    if name == 'resnet32':
        return resnet32()
    elif name == 'resnet18':
        return resnet18(pretrained=pretrained)
    elif name == 'resnet18_cifar':
        return resnet18(pretrained=pretrained, cifar=True)
    elif name == 'resnet18_cifar_cos':
        return resnet18(pretrained=pretrained, cifar=True, no_last_relu=True)
    elif name == 'resnet34':
        return resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        return resnet50(pretrained=pretrained)
    elif name == 'vit-b-p16':
        return vit_base_patch16_224_in21k(pretrained=True)
    elif name == 'vit-b-p16-mocov3':
        return vit_base_patch16_224_mocov3(pretrained=True)
    elif name == 'vit-b-p16-lora':
        return vit_base_lora_patch16_224_in21k(pretrained=True, lora_rank=cfg['lora_rank'])
    elif name == 'vit-b-p16-lora-mocov3':
        return vit_base_lora_patch16_224_mocov3(pretrained=True, lora_rank=cfg['lora_rank'])
    else:
        raise NotImplementedError('Unknown type {}'.format(name))


class MyVitNet(nn.Module):

    def __init__(self, nclasses):
        super(MyVitNet, self).__init__()

        self.convnet = vit_base_patch16_224_in21k(pretrained=True)
        self.fc = nn.Linear(self.encoder.hidden_dim, nclasses)

    def features(self, x):
        '''在全连接层之前提取特征'''
        out = self.encoder(x)
        return out

    def logits(self, x):
        '''通过最后的全连接层获得 logits'''
        x = self.linear(x)
        return x

    def forward(self, x):
        out = self.features(x)
        out = self.logits(out)  # 获取最终的 logits 输出
        return out


