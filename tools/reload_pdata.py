import glob

import hydra
import torch
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path='../configs', config_name='base')
def reload_pdata(config: DictConfig):
    source_path = '../param_data/cifar10/data.pt'
    old = torch.load(source_path)
    target_path = './pdata.pth'
    new = torch.load(target_path)
    old['pdata'] = new
    old['cfg'] = config.task
    old['train_layer'] = config.task['train_layer']
    torch.save(old, source_path)


def cat_pdata():
    folder_path = '../tmp/'
    files = glob.glob(f'{folder_path}/tmp*')
    pdata_list = []
    for file in files:
        t = torch.load(file)
        # if t['pdata'].shape[0] == 20:
        pdata_list.append(t['pdata'])
    res = torch.cat(pdata_list, dim=0)
    # res = res.view((res.shape[0] * res.shape[1], -1))
    torch.save(res, 'pdata.pth')


if __name__ == '__main__':
    cat_pdata()
    reload_pdata()
