import torch
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path='../../configs', config_name='base')
def reload(config: DictConfig):
    ld = torch.load('data.pt')
    target_path = '/home/chengyiqiu/code/diffusion/Diffuse-Backdoor-Parameters/tmp/pdata_resnet18_cifar10.pth'
    target_ld = torch.load(target_path)
    ld['pdata'] = target_ld['pdata']
    ld['cfg'] = config.task
    ld['train_layer'] = ld['cfg']['train_layer']
    torch.save(ld, 'data.pt')
    print()

if __name__ == '__main__':
    reload()
