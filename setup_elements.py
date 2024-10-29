import torch
from models.resnet import Reduced_ResNet18, ImageNet_ResNet18, ResNet18_pretrained, ResNet18, ResNet50, SupConResNet, \
    distLinear
from models.foundation_model import MyViT16
from torchvision import transforms
import torch.nn as nn
from loss.FocalLoss import FocalLoss
from loss.RevisedFocalLoss import RevisedFocalLoss
from loss.SCRLoss import SupConLoss
import kornia
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale, \
    RandomVerticalFlip, RandomRotation

input_size_match = {
    'cifar100': (3, 32, 32),
    # 'cifar100': (3, 224, 224), # ViT-B/16
    'cifar10': (3, 32, 32),
    'mini_imagenet': (3, 84, 84),
    'imagenet100': (3, 224, 224),
}

n_classes = {
    'cifar100': 100,
    'cifar10': 10,
    'mini_imagenet': 100,
    'imagenet100': 100,
}

feature_size_match = {
    'cifar100': 160,  # Reduced_ResNet-18
    # 'cifar100': 512, # ResNet-18
    # 'cifar100': 768, # ViT-B/16
    'cifar10': 160,
    'mini_imagenet': 640,
    'imagenet100': 160,
}

transforms_match = {
    'cifar100': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'cifar10': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'mini_imagenet': transforms.Compose([
        transforms.ToTensor()
    ]),
    'imagenet100': transforms.Compose([
        transforms.ToTensor()
    ]),
}


def setup_architecture(params):
    nclass = n_classes[params.data]
    if params.data == 'cifar100':
        if params.vit:
            model = MyViT16(nclass)
            if params.bf:
                model.BatchFormer = nn.TransformerEncoderLayer(768, 4, 512, params.drop_rate)
                if params.dist:
                    model.linear = distLinear(768, nclass, params.temperature)
        else:
            model = Reduced_ResNet18(nclass)
            if params.bf:
                model.BatchFormer = nn.TransformerEncoderLayer(160, 4, 160, params.drop_rate)
                if params.dist:
                    model.linear = distLinear(160, nclass, params.temperature)
        return model
    elif params.data == 'cifar10':
        model = Reduced_ResNet18(nclass)
        if params.bf:
            model.BatchFormer = nn.TransformerEncoderLayer(160, 4, 160, params.drop_rate)
            if params.dist:
                model.linear = distLinear(160, nclass, params.temperature)
        return model
    elif params.data == 'mini_imagenet':
        model = Reduced_ResNet18(nclass)
        if params.bf:
            model.BatchFormer = nn.TransformerEncoderLayer(640, 4, 640, params.drop_rate)
            if params.dist:
                model.linear = distLinear(640, nclass, params.temperature)
            else:
                model.linear = nn.Linear(640, nclass, bias=True)
        else:
            model.linear = nn.Linear(640, nclass, bias=True)
        return model
    elif params.data == 'imagenet100':
        model = ImageNet_ResNet18(nclass)
        if params.bf:
            model.BatchFormer = nn.TransformerEncoderLayer(160, 4, 160, params.drop_rate)
            if params.dist:
                model.linear = distLinear(160, nclass, params.temperature)
            else:
                model.linear = nn.Linear(160, nclass, bias=True)
        else:
            model.linear = nn.Linear(160, nclass, bias=True)
        return model
    else:
        Exception('wrong dataset name')


def setup_opt(optimizer, model, lr, wd):
    if optimizer == 'SGD':
        optim = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                weight_decay=wd)
    elif optimizer == 'Adam':
        optim = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=wd)
    else:
        raise Exception('wrong optimizer name')
    return optim


def setup_crit(params):
    if params.agent == "scr" or params.loss == "scl":
        criterion = SupConLoss(params.temperature)
    else:
        if params.loss == "focal":
            criterion = FocalLoss(params.focal_alpha, params.focal_gamma)
        elif params.loss == "ce":
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        elif params.loss == "rfocal":
            criterion = RevisedFocalLoss(params.rfl_alpha, params.rfl_sigma, params.rfl_miu)
        else:
            raise NotImplementedError(
                'loss not supported: {}'.format(params.loss))
    return criterion


def setup_augment(params):
    aug_transform = nn.Sequential(
        RandomResizedCrop(size=(input_size_match[params.data][1], input_size_match[params.data][2]), scale=(0.75, 1.)),
        RandomHorizontalFlip(),
    )
    return aug_transform
