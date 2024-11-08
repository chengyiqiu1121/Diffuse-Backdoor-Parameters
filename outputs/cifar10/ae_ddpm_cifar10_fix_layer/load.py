import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from core.module.modules.encoder import medium
from core.module.modules.unet import AE_CNN_bottleneck
import torch.nn.functional as F
from tqdm import tqdm


# from core.utils.utils import partial_reverse_tomodel
def partial_reverse_tomodel(flattened, model, train_layer):
    layer_idx = 0
    for name, pa in model.named_parameters():
        if name in train_layer:
            pa_shape = pa.shape
            pa_length = pa.view(-1).shape[0]
            pa.data = flattened[layer_idx:layer_idx + pa_length].reshape(pa_shape)
            pa.data.to(flattened.device)
            layer_idx += pa_length
    return model


ae_ddpm_path = 'ae_ddpm59999.pth'
ld = torch.load(ae_ddpm_path)
ae_model = medium(
    in_dim=7178,
    input_noise_factor=1e-3,
    latent_noise_factor=0.5
)
ae_model.load_state_dict(ld['ae_model'])
ae_cnn = AE_CNN_bottleneck(
    in_channel=1,
    in_dim=44
)
ae_cnn.load_state_dict(ld['model'])
noice = torch.randn(200, 4, 11)
time = (torch.rand(noice.shape[0]) * 1000).type(torch.int64).to(noice.device)
latent = ae_cnn(noice, time, cond=None)
ae_params = ae_model.decode(latent)
ae_params = ae_params.cpu()
del ae_model, ae_cnn
print(f'ae param shape: {ae_params.shape}')
# ----------------prepare resnet -----------------
from models.resnet import ResNet18

res_path = '/home/chengyiqiu/code/backdoors/stable_backdoor_purification/record_cifar10/badnet/pratio_0.1-target_0-archi_resnet18-dataset_cifar10-sratio_0.02-initlr_0.1/attack_result.pt'
t = torch.load(res_path)
resnet = ResNet18(num_classes=10)
resnet.load_state_dict(t['model'])
resnet.to('cuda:0')
train_layer = ['layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn2.bias', 'layer4.1.bn2.weight', 'linear.weight',
               'linear.bias']
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
])
test_dataset = datasets.CIFAR10(
    root='../../../data/cifar10', train=False, download=True,
    transform=transform,
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)
device = 'cuda:0'
resnet = resnet.to(device)

# -------------- do eval --------------------
acc_list, test_loss_list = [], []
for i, param in enumerate(tqdm(ae_params)):
    param = param.to(device)  # (1, 2048)
    target_num = 0
    for name, module in resnet.named_parameters():
        if name in train_layer:
            target_num += torch.numel(module)
    params_num = torch.squeeze(param).shape[0]  # + 30720
    assert (target_num == params_num)
    param = torch.squeeze(param)
    model = resnet
    model = partial_reverse_tomodel(param, model, train_layer).to(param.device)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    output_list = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            target = target.to(torch.int64)
            # output = F.softmax(output, dim=1)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            total += data.shape[0]
            pred = torch.max(output, 1)[1]
            output_list += pred.cpu().numpy().tolist()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total
    acc = 100. * correct / total
    del model
    # return acc, test_loss, output_list
    acc_list.append(acc)
    test_loss_list.append(test_loss)
    # print(f'acc: {acc: .2f}, test loss: {test_loss: .2f}')
# Sort the list
sorted_acc = sorted(acc_list)

# Calculate the average
average = sum(acc_list) / len(acc_list)

# Get the maximum value
max_value = max(acc_list)

# Get the minimum value
min_value = min(acc_list)

# Calculate the median
middle = len(acc_list) // 2
if len(acc_list) % 2 == 0:  # If the list length is even
    median = (sorted_acc[middle - 1] + sorted_acc[middle]) / 2
else:
    median = sorted_acc[middle]

# Print the results
print(f"Sorted list of accuracies: {sorted_acc}")
print(f"Average accuracy: {average:.2f}")
print(f"Max accuracy: {max_value:.2f}")
print(f"Min accuracy: {min_value:.2f}")
print(f"Median accuracy: {median:.2f}")
