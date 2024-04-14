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

ae_ddpm_path = '../outputs/cifar10/ae_ddpm_cifar10_pth/ae_ddpm59999.pth'
ld = torch.load(ae_ddpm_path)
ae_model = medium(
    in_dim=5130,
    input_noise_factor=1e-3,
    latent_noise_factor=0.5
)
ae_cnn = AE_CNN_bottleneck(
    in_channel=1,
    in_dim=32
)
ae_model.load_state_dict(ld['ae_model'])
ae_cnn.load_state_dict(ld['model'])
noice = torch.randn(300, 4, 8)
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
# resnet.load_state_dict(torch.load(res_path)['model'])
state_dict = torch.load(res_path)['model']
# train_layer = ['layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn2.bias', 'layer4.1.bn2.weight']
train_layer = ['linear.weight', 'linear.bias']
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
test_dataset = datasets.CIFAR10(
    root='/home/chengyiqiu/code/diffusion/Diffuse-Backdoor-Parameters/data/cifar10', train=True, download=True,
    transform=transform,
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)
device = 'cuda:0'
resnet = ResNet18()
resnet.load_state_dict(state_dict)
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
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

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
import tools.tg_bot as bot
bot.send2bot('done', 'test pdiff')