import torch
from models.resnet import ResNet18
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm


def train_one_epoch(net, criterion, optimizer, trainloader, current_epoch, device):
    print('\nEpoch: %d' % current_epoch)
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

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(net, criterion, testloader, device):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        print('\nstart testing')
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
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


def train(net, criterion, optimizer, trainloader, testloader, epoch, device):
    best_scc = 0
    for i in tqdm(range(epoch)):
        train_one_epoch(net, criterion, optimizer, trainloader, i, device)
        current_acc = test(net, criterion, testloader, device)
        best_scc = max(current_acc, best_scc)
    return best_scc


if __name__ == '__main__':
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
    train_loader = DataLoader(train_data, batch, shuffle=True)
    test_loader = DataLoader(test_data, batch, shuffle=False)
    device = 'cuda:1'
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
    net = net.to(device)
    train(net=net, criterion=loss_fn, optimizer=optimizer, epoch=10, trainloader=train_loader, device=device,
          testloader=test_loader)
    acc = test(net=net, criterion=loss_fn, testloader=test_loader, device=device)
    print(acc)
