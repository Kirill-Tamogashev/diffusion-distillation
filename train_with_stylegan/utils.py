from diffusers import UNet2DModel
import numpy as np

import torch


class DiffusionScheduler:
    def __init__(
        self, 
        b_min:      float,
        b_max:      float,
        device,
        n_steps:    int = 1000, 
        continious: bool = False,
    ):
        self._countinious = continious
        self._device = device
        def beta_fn(t):
            return 0.5 * t**2 * (b_max - b_min) + b_min * t
            
        if continious:
            self._beta_fn = beta_fn
        else:
            betas = np.linspace(b_min, b_max, n_steps, dtype=np.float64)
            alphas = 1 - betas
            alphas_comprod = np.cumprod(alphas)
            self._shifts = torch.from_numpy(np.sqrt(alphas_comprod))
            self._sigmas = torch.from_numpy(np.sqrt(1 - alphas_comprod))
            
    def get_schedule(self, t):
        if self._countinious:
            alpha = torch.exp(-0.5 * self._beta_fn(t))
            sigma = torch.sqrt(1 - torch.exp(- self._beta_fn(t)))
        else:
            alpha = self._shifts[t]
            sigma = self._sigmas[t]
        alpha = alpha.reshape(-1, 1, 1, 1)
        sigma = sigma.reshape(-1, 1, 1, 1)
        return alpha.to(self._device), sigma.to(self._device)

def configure_unet_model_from_pretrained(model_config):
    return UNet2DModel.from_pretrained(model_config)
