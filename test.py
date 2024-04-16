import logging

import torch
from torchvision.transforms import transforms

from tools.classfication import test
import torchvision.datasets
from torchvision.transforms.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from models.resnet import ResNet18
import torch.nn.functional as F

if __name__ == '__main__':
    net = ResNet18(num_classes=10)
    for name, param in net.named_parameters():
        print(name)


