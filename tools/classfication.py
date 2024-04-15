import logging
import random
import time

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
# from tools.tg_bot import send2bot
from tg_bot import send2bot
import torch.nn.init as init

collect_layer = ['linear.weight', 'linear.bias']
batch_list = [64, 128, 256, 512, 1024]
default_batch = batch_list[0]
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def fix_partial_model(train_list, net):
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


def train_one_epoch(net, criterion, optimizer, trainloader, current_epoch, device, lr_schedule=None):
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


def test(net, criterion, testloader, device):
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


def rand_init_layer(net, init_layer):
    target_layer_params = net.named_parameters()
    target_params = []
    for name, param in target_layer_params:
        if name in init_layer:
            target_params.append(param)
    for param in target_params:
        if len(param.shape) > 1:
            init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        else:
            init.uniform_(param, a=-0.01, b=0.01)


def train(net, criterion, optimizer, trainloader, testloader, epoch, device, lr_schedule=None):
    best_acc = 0
    acc_list = []
    print('-' * 50)
    print('start train')
    print('-' * 50)
    for i in range(epoch):
        train_one_epoch(net, criterion, optimizer, trainloader, i, device, lr_schedule)
        current_acc = test(net, criterion, testloader, device)
        logger.info(f'epoch{i}: current acc {current_acc: .2f}')
        acc_list.append(current_acc)
        best_acc = max(current_acc, best_acc)
        if best_acc >= 88.88:
            logger.debug('acc is already 88.88, exit')
            break
        if lr_schedule is not None:
            lr_schedule.step()
    res_dict = {
        'acc_list': acc_list,
        'state_dict': net.state_dict(),
    }
    torch.save(res_dict, '../tmp/whole_model_resnet18_cifar10.pth')


def get_data_loader(name, batch, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_data, test_data = None, None
    if name == 'CIFAR10':
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
    return (DataLoader(train_data, batch, shuffle=True, num_workers=num_workers),
            DataLoader(test_data, batch, shuffle=True, num_workers=num_workers))


def train_for_pdata(net, criterion, optimizer, trainloader, testloader, device, init_layer=None, mode=0):
    print('*' * 50)
    print('prepare pdata')
    print('*' * 50)
    if init_layer is None:
        init_layer = []
    best_acc = 0
    n = 10
    model_num = 200
    batch = default_batch
    while True:
        if mode == 0:
            acc_list = []
            parameter_data = []
            for i in range(2 * n):
                train_one_epoch(net, criterion, optimizer, trainloader, i, device, lr_schedule)
                current_acc = test(net, criterion, testloader, device)
                logger.info(f'epoch{i}: current acc {current_acc:.2f}')
                acc_list.append(current_acc)
                best_acc = max(current_acc, best_acc)
                parameter_data.append(state_part(collect_layer, net))
                model_num -= 1
                if model_num == 0:
                    # jump out for loop
                    break
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
            if model_num == 0:
                # jump out while loop
                break
            # prepare new model
            best_acc = 0
            mode = 1
            n = 0
        elif mode == 1:
            new_batch = random.choice(batch_list)
            old_batch = batch
            while True:
                if new_batch == old_batch:
                    new_batch = random.choice(batch_list)
                else:
                    break
            batch = new_batch
            logger.debug(f'new batch: {batch}')
            trainloader, testloader = get_data_loader(name='CIFAR10', batch=batch,
                                                      num_workers=num_workers)
            rand_init_layer(net, init_layer)
            init_acc = test(net, criterion, testloader, device)
            logger.debug(f'after random init acc: {init_acc: .2f}')
            for i in range(10):
                train_one_epoch(net, criterion, optimizer, trainloader, i, device)
                current_acc = test(net, criterion, testloader, device)
                logger.debug(f'epoch{i}: retrain acc {current_acc: .2f}')
                best_acc = max(best_acc, current_acc)
                n += 1
                if current_acc > 88:
                    break
            mode = 0


if __name__ == '__main__':
    import sys

    sys.path.append('..')
    from models.resnet import ResNet18

    net = ResNet18(num_classes=10)
    batch = 128
    num_workers = 8
    train_loader, test_loader = get_data_loader('CIFAR10', batch, num_workers)
    device = 'cuda:0'
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
    net = net.to(device)
    batch_list = [32, 64, 128, 256, 512, 1024]

    lr_schedule = MultiStepLR(milestones=[30, 60, 90, 100], gamma=0.2, optimizer=optimizer)
    train(net=net, criterion=loss_fn, optimizer=optimizer, epoch=100, trainloader=train_loader, device=device,
          testloader=test_loader, lr_schedule=lr_schedule)
    send2bot('train whole model done', 'train whole')
    init_layer = ['conv1.weight', 'bn1.weight', 'bn1.bias']
    # update optimizer, set lr=1e-4 to fine-tune
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)
    train_for_pdata(net=net, criterion=loss_fn, optimizer=optimizer, trainloader=train_loader, testloader=test_loader,
                    device=device, init_layer=init_layer, mode=0)
    send2bot('train pdata done', 'train pdata')
