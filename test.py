import torch
from torchvision.transforms import transforms

from tools.classfication import test
import torchvision.datasets
from torchvision.transforms.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from models.resnet import ResNet18
import torch.nn.functional as F

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_data = torchvision.datasets.CIFAR10(
        root='data/cifar10', train=False, download=False, transform=transform
    )
    test_loader = DataLoader(dataset=test_data, batch_size=2048, shuffle=True, num_workers=8)
    net = ResNet18(num_classes=10)
    net = net.to('cuda:0')
    res = test(net=net, criterion=F.cross_entropy, testloader=test_loader, device='cuda:0')
    print(res)
    resnet = ResNet18(num_classes=10)
    ld = torch.load('tmp/whole_model_resnet18_cifar10.pth')
    resnet.load_state_dict(ld['state_dict'])
    resnet.to('cuda:0')
    net = resnet
    res = test(net=net, criterion=F.cross_entropy, testloader=test_loader, device='cuda:0')
    print(res)
