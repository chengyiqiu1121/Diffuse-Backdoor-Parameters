import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig


def extract(input, t, shape):
    input = input.to(t.device)
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out


class AE_DDPM():
    def __init__(self, config):
        self.ae_model = hydra.utils.instantiate(config.system.ae_model)
        input_dim = config.system.ae_model.in_dim
        input_noise = torch.randn((1, input_dim))
        latent_dim = self.ae_model.encode(input_noise).shape
        config.system.model.arch.model.in_dim = latent_dim[-1] * latent_dim[-2]
        self.diff_model = hydra.utils.instantiate(config.system.model.arch.model)
        self.config = config
        self.split_epoch = self.config.system.train.split_epoch
        self.max_epoch = self.config.system.train.max_epoch
        self.loss_func = nn.MSELoss()
        self.n_timestep = self.config.system.beta_schedule.n_timestep
        self.ae_optimizer = hydra.utils.instantiate(self.config.system.train.ae_optimizer,
                                                    params=self.ae_model.parameters())
        self.diff_optimizer = hydra.utils.instantiate(self.config.system.train.optimizer,
                                                      params=self.diff_model.parameters())

    def ae_forward(self, batch):
        output = self.ae_model(batch)
        loss = self.loss_func(batch, output)
        return loss

    def pre_process(self, batch):
        latent = self.ae_model.encode(batch)
        self.latent_shape = latent.shape[-2:]
        return latent

    def post_process(self, latent):
        latent = latent.reshape(-1, *self.latent_shape)
        return self.ae_model.decode(latent)


@hydra.main(version_base=None, config_path='../configs', config_name='base')
def train(config: DictConfig):
    task = AE_DDPM(config=config)
    print(task.ae_model)


if __name__ == '__main__':
    train()
