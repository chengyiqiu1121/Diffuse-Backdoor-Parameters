import glob

import torch
from models.resnet import ResNet18

p = '/home/chengyiqiu/code/diffusion/Diffuse-Backdoor-Parameters/tmp/tmp1713177373.713903.pth'
t =torch.load(p)
print()