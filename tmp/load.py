import torch

t1 = torch.load('whole_model_resnet18_cifar10.pth')
t2 = torch.load('pdata_resnet18_cifar10.pth')
# res = []
# for i in t:
#     w = i['linear.weight'].reshape(-1)
#     b = i['linear.bias']
#     wb = torch.cat([w, b], dim=0)
#     res.append(wb)
# res = torch.stack(res)
print()
