import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR


def fix_partial_model(train_list, net):
    print(train_list)
    for name, weights in net.named_parameters():
        if name not in train_list:
            weights.requires_grad = False


def state_part(train_list, net):
    part_param = {}
    for name, weights in net.named_parameters():
        if name in train_list:
            part_param[name] = weights.detach().cpu()
    return part_param


def pdata_dic2tensor(pdata):
    res = []
    for i in pdata:
        w = i['linear.weight'].reshape(-1)
        b = i['linear.bias']
        wb = torch.cat([w, b], dim=0)
        res.append(wb)
    return torch.stack(res)


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

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        return 100. * correct / total


def train(net, criterion, optimizer, trainloader, testloader, epoch, device, train_layer=None, lr_schedule=None):
    if train_layer is None:
        train_layer = 'all'
    best_acc = 0
    parameter_data = []
    acc_list = []
    if train_layer == 'all':
        for i in tqdm(range(epoch), desc=f'training: {train_layer}'):
            train_one_epoch(net, criterion, optimizer, trainloader, i, device, lr_schedule)
            current_acc = test(net, criterion, testloader, device)
            acc_list.append(current_acc)
            best_acc = max(current_acc, best_acc)
        res_dict = {
            'acc_list': acc_list,
            'state_dict': net.state_dict(),
        }
        torch.save(res_dict, '../tmp/whole_model_resnet18_cifar10.pth')
    else:
        fix_partial_model(train_layer, net)
        for i in tqdm(range(epoch)):
            train_one_epoch(net, criterion, optimizer, trainloader, i, device)
            current_acc = test(net, criterion, testloader, device)
            acc_list.append(current_acc)
            best_acc = max(current_acc, best_acc)
            parameter_data.append(state_part(train_layer, net))
        res_dict = {
            'best_acc': best_acc,
            'acc_list': acc_list,
            'pdata': pdata_dic2tensor(parameter_data),
        }
        torch.save(res_dict, '../tmp/pdata_resnet18_cifar10.pth')
    return best_acc





if __name__ == '__main__':
    import sys

    sys.path.append('..')
    from models.resnet import ResNet18

    net = ResNet18(num_classes=10)
    train_data = datasets.CIFAR10(
        root='../data/cifar10',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    test_data = datasets.CIFAR10(
        root='../data/cifar10',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    batch = 2048
    num_workers = 8
    train_loader = DataLoader(train_data, batch, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch, shuffle=False, num_workers=num_workers)
    device = 'cuda:0'
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
    net = net.to(device)
    train_layer = ['linear.weight', 'linear.bias']
    lr_schedule = MultiStepLR(milestones=[30, 60, 90, 100], gamma=0.2, optimizer=optimizer)
    train(net=net, criterion=loss_fn, optimizer=optimizer, epoch=100, trainloader=train_loader, device=device,
          testloader=test_loader, lr_schedule=lr_schedule)
    train(net=net, criterion=loss_fn, optimizer=optimizer, epoch=200, trainloader=train_loader, device=device,
          testloader=test_loader, train_layer=train_layer, lr_schedule=lr_schedule)
