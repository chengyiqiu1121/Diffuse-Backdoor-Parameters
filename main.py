import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from tools.tg_bot import send2bot
from tools.ae_ddpm import AE_DDPM


@hydra.main(version_base=None, config_path='configs', config_name='base')
def train_pdiff(config: DictConfig):
    system = AE_DDPM(config=config)
    datamodule = system.get_task().get_param_data()
    # running
    trainer: Trainer = hydra.utils.instantiate(config.system.train.trainer)
    trainer.fit(system, datamodule=datamodule, ckpt_path=config.load_system_checkpoint)
    trainer.test(system, datamodule=datamodule)


if __name__ == '__main__':
    try:
        train_pdiff()
        send2bot('done', 'train p diffusion')
    except Exception as e:
        send2bot(msg=str(e), title='error occur')

