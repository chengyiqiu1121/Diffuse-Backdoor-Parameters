import random
import time

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from tg_bot import send2bot

collect_layer = ['linear.weight', 'linear.bias']


def fix_partial_model(train_list, net):
    print(train_list)
    for name, weights in net.named_parameters():
        if name not in train_list:
            weights.requires_grad = False
        else:
            weights.requires_grad = True


def state_part(train_list, net):
    part_param = {}
    for name, weights in net.named_parameters():
        if name in train_list:
            part_param[name] = weights.detach().cpu()
    return part_param


def pdata_dic2tensor(pdata):
    res = []
    for i in pdata:
        for k, v in i.items():
            res.append(v.reshape(-1))
    return res


def train_one_epoch(net, criterion, optimizer, trainloader, current_epoch, device, lr_schedule):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    if lr_schedule is not None:
        lr_schedule.step()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(net, criterion, testloader, device):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return 100. * correct / total


def train(net, criterion, optimizer, trainloader, testloader, epoch, device, train_layer=None, lr_schedule=None):
    if train_layer is None:
        train_layer = 'all'
    best_acc = 0
    parameter_data = []
    acc_list = []
    if train_layer == 'all':
        for i in tqdm(range(epoch)):
            train_one_epoch(net, criterion, optimizer, trainloader, i, device, lr_schedule)
            current_acc = test(net, criterion, testloader, device)
            acc_list.append(current_acc)
            best_acc = max(current_acc, best_acc)
            if lr_schedule is not None:
                lr_schedule.step()
        res_dict = {
            'acc_list': acc_list,
            'state_dict': net.state_dict(),
        }
        torch.save(res_dict, '../tmp/whole_model_resnet18_cifar10.pth')
    else:
        fix_partial_model(train_layer, net)
        for i in tqdm(range(epoch)):
            train_one_epoch(net, criterion, optimizer, trainloader, i, device, lr_schedule)
            current_acc = test(net, criterion, testloader, device)
            acc_list.append(current_acc)
            best_acc = max(current_acc, best_acc)
            # parameter_data.append(state_part(train_layer, net))
            parameter_data.append(state_part(collect_layer, net))
            if lr_schedule is not None:
                lr_schedule.step()
        t1 = pdata_dic2tensor(parameter_data)
        res_pdata = []
        index = 0
        for i in range(int(len(t1) / len(collect_layer))):
            res_pdata.append(torch.cat(t1[index: (index + len(collect_layer))], dim=0))
            index = index + len(collect_layer) - 1
        res_pdata = torch.stack(res_pdata)
        res_dict = {
            'best_acc': best_acc,
            'acc_list': acc_list,
            'pdata': res_pdata,
        }
        torch.save(res_dict, f'../tmp/tmp{time.time().__str__()}.pth')
    return best_acc


if __name__ == '__main__':
    import sys

    sys.path.append('..')
    from models.resnet import ResNet18

    net = ResNet18(num_classes=10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_data = datasets.CIFAR10(
        root='../data/cifar10',
        train=True,
        download=True,
        transform=transform
    )
    test_data = datasets.CIFAR10(
        root='../data/cifar10',
        train=False,
        download=True,
        transform=transform
    )
    batch = 64
    num_workers = 2
    train_loader = DataLoader(train_data, batch, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch, shuffle=False, num_workers=num_workers)
    device = 'cuda:0'
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
    net = net.to(device)
    train_layer_1 = ['layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias',
                     'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias',
                     'linear.weight', 'linear.bias']
    train_layer_2 = ['layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias',
                     'layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'linear.weight', 'linear.bias']
    train_layer_3 = ['layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias',
                     'linear.weight', 'linear.bias']
    train_layer_4 = ['layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'linear.weight', 'linear.bias']
    train_layer_5 = ['linear.weight', 'linear.bias']

    lr_schedule = MultiStepLR(milestones=[30, 60, 90, 100], gamma=0.2, optimizer=optimizer)
    train(net=net, criterion=loss_fn, optimizer=optimizer, epoch=100, trainloader=train_loader, device=device,
          testloader=test_loader, lr_schedule=lr_schedule)
    send2bot('train whole model done', 'train whole')
    train(net=net, criterion=loss_fn, optimizer=optimizer, epoch=50, trainloader=train_loader, device=device,
          testloader=test_loader, train_layer=train_layer_1)
    send2bot(msg='done', title='train layer 1')
    train(net=net, criterion=loss_fn, optimizer=optimizer, epoch=50, trainloader=train_loader, device=device,
          testloader=test_loader, train_layer=train_layer_2)
    send2bot(msg='done', title='train layer 2')
    train(net=net, criterion=loss_fn, optimizer=optimizer, epoch=50, trainloader=train_loader, device=device,
          testloader=test_loader, train_layer=train_layer_3)
    send2bot(msg='done', title='train layer 3')
    train(net=net, criterion=loss_fn, optimizer=optimizer, epoch=50, trainloader=train_loader, device=device,
          testloader=test_loader, train_layer=train_layer_4)
    send2bot(msg='done', title='train layer 4')
    train(net=net, criterion=loss_fn, optimizer=optimizer, epoch=50, trainloader=train_loader, device=device,
          testloader=test_loader, train_layer=train_layer_5)
    send2bot(msg='done', title='train layer 5')
