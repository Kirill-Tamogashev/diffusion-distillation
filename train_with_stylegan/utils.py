from diffusers import UNet2DModel

import torch

import copy


class EMA:
    def __init__(
        self,
        model:  torch.nn.Module,
        beta:   float = 0.9999, 
    ):
        self._beta = beta

        self._model = model
        self._ema_model = copy.deepcopy(self._model)
        self._ema_detach_grads()
        self._ema_model.eval()

    def _ema_detach_grads(self):
        for param in self._ema_model.parameters():
            param.detach_()
        
    @property
    def ema(self):
        return self._ema_model
    
    def __call__(self, inputs):
        return self._ema_model(*inputs)
    
    def to(self, device):
        self._ema_model.to(device)
        
    def _update_params(self):
        for param_ema, param_model in zip(self._ema_model.parameters(), self._model.parameters()):
            param_ema.mul_(self._beta).add_(param_model.data, alpha=1 - self._beta)
        
    def _update_buffers(self):
        for buffer_ema, buffer_model in zip(self._ema_model.buffers(), self._model.model()):
            buffer_ema.data = buffer_model.date
        
    def update(self):
        self._update_params()
        self._update_buffers()

    
class DiffusionScheduler:
    def __init__(
        self, 
        b_min:          float,
        b_max:          float,
        device,
        n_timesteps:    int = 1000,
        continuous:     bool = False,
        t_eps:          float = 1e-5,
    ):
        self._continuous = continuous
        self._device = device

        self.timesteps = torch.arange(0, n_timesteps, dtype=torch.int64)

        if continuous:
            def beta_fn(t: torch.Tensor) -> torch.Tensor:
                return 0.5 * t**2 * (b_max - b_min) + b_min * t

            #  [0, 1] -> [t_eps, 1] to avoid collapsing in 0
            timesteps = t_eps + (1.0 - t_eps) * self.timesteps / n_timesteps
            self.alphas_cumprod = torch.exp(- beta_fn(timesteps))
        else:
            betas = torch.linspace(b_min, b_max, n_timesteps, dtype=torch.float32)
            self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    def alpha(self, t):
        #  Use mask to handle cases where t < 0, only for sampling
        mask = (t >= 0).int()
        alpha_positive_t = torch.sqrt(self.alphas_cumprod)[t * mask] * mask
        alpha_negative_t = torch.ones_like(t, device=t.device) * (1 - mask)
        return alpha_positive_t + alpha_negative_t

    def sigma(self, t):
        #  Use mask to handle cases where t < 0, only for sampling
        mask = (t >= 0).int()
        sigma_positive_t = torch.sqrt(1 - self.alphas_cumprod)[t * mask] * mask
        sigma_negative_t = torch.zeros_like(t, device=t.device) * (1 - mask)
        return sigma_positive_t + sigma_negative_t
            
    def get_schedule(self, t):
        alpha = self.alpha(t).reshape(-1, 1, 1, 1)
        sigma = self.sigma(t).reshape(-1, 1, 1, 1)
        return alpha.to(self._device), sigma.to(self._device)


def configure_unet_model_from_pretrained(model_config):
    return UNet2DModel.from_pretrained(model_config)
