import copy
import torch
from torch import nn
# from convs.cifar_resnet import resnet32
# from convs.resnet import resnet18, resnet34, resnet50
# from convs.linears import SimpleContinualLinear
from .vits import vit_base_patch16_224_in21k, vit_base_patch16_224_mocov3, vit_base_lora_patch16_224_in21k, \
    vit_base_lora_patch16_224_mocov3
import torch.nn.functional as F


class MyVit(nn.Module):

    def __init__(self, nclasses):
        super(MyVit, self).__init__()

        self.encoder = vit_base_patch16_224_in21k(pretrained=True)
        self.fc = nn.Linear(self.encoder.embed_dim, nclasses)

    def features(self, x):
        out = self.encoder(x)['features']
        return out

    def logits(self, x):
        x = self.fc(x)
        return x

    def forward(self, x):
        out = self.features(x)
        out = self.logits(out)
        return out


def MyViT16(nclasses):
    return MyVit(nclasses)
