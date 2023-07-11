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
            scale = 1000 / n_steps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            alphas = 1 - np.linspace(
                beta_start, beta_end, n_steps, dtype=np.float64
            )
            alphas_comprod = np.cumprod(alphas)
            self._shifts = torch.tensor(np.sqrt(alphas_comprod))
            self._sigmas = torch.tensor(np.sqrt(1 - alphas_comprod))
            
    def get_schedule(self, t):
        if self._countinious:
            alpha = torch.exp(-0.5 * self._beta_fn(t))
            sigma = torch.sqrt(1 - torch.exp(- self._beta_fn(t)))
        else:
            alpha = self._shifts[t]
            sigma = self._sigmas[t]
        return alpha.to(self._device), sigma.to(self._device)
               

def get_shifts_and_sigmas(n_steps: int):
    scale = 1000 / n_steps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    alphas = 1 - np.linspace(
        beta_start, beta_end, n_steps, dtype=np.float64
    )
    alphas_comprod = np.cumprod(alphas)
    shifts = np.sqrt(alphas_comprod)
    sigmas = np.sqrt(1 - alphas_comprod)
    return shifts, sigmas


def freeze_weights(model: torch.nn.Module):
    for _, params in model.named_parameters:
        params.requires_grad = False

def unfreeze_weights(model: torch.nn.Module):
    for _, params in model.named_parameters:
        params.requires_grad = True


def configure_unet_model_from_pretrained(model_config):
    return UNet2DModel.from_pretrained(model_config)

# def condigure_unet_model(unet_params: dict):
#     return UNet2DModel(
#         sample_size=...,
#         in_channels=...,
#         out_channels=...,
#         layers_per_block=...,
#         attention_head_dim=...,
#         block_out_channels=[...],
#         down_block_types=[
#             ...
#         ],
#         up_block_types=[
#             ...
#         ],
        
        
#     )