import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from tools.ae_ddpm import AE_DDPM


@hydra.main(version_base=None, config_path='configs', config_name='base')
def train_pdiff(config: DictConfig):
    system = AE_DDPM(config=config)
    datamodule = system.get_task().get_param_data()
    pdata_path = '/home/chengyiqiu/code/diffusion/Diffuse-Backdoor-Parameters/tmp/pdata_resnet18_cifar10.pth'
    datamodule.pdata = torch.load(pdata_path)['pdata']
    # running
    trainer: Trainer = hydra.utils.instantiate(config.system.train.trainer)
    trainer.fit(system, datamodule=datamodule, ckpt_path=config.load_system_checkpoint)
    trainer.test(system, datamodule=datamodule)


if __name__ == '__main__':
    train_pdiff()
