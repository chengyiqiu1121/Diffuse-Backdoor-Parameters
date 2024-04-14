import hydra
import torch
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path='../configs', config_name='base')
def reload_pdata(config: DictConfig):
    source_path = '../param_data/cifar10/data.pt'
    old = torch.load(source_path)
    target_path = '../tmp/pdata.pth'
    new = torch.load(target_path)
    old['pdata'] = new
    old['cfg'] = config.task
    old['train_layer'] = config.task['train_layer']
    torch.save(old, source_path)


if __name__ == '__main__':
    reload_pdata()