import torch
from torch import nn
import lpips


class LPIPS(nn.Module):
    def __init__(self, net='alex'):
        super(LPIPS, self).__init__()
        self.metric = lpips.LPIPS(net=net)

        for m in self.metric.modules():
            names = [name for name, _ in m.named_parameters()]
            for name in names:
                if hasattr(m, name):
                    data = getattr(m, name).data
                    delattr(m, name)
                    m.register_buffer(name, data, persistent=False)

    @torch.no_grad()
    def __call__(self, inputs, targets):
        return self.metric(inputs, targets, normalize=True)
    
    def calc_loss(self, inputs, targets):
        return self.metric(inputs, targets, normalize=True)

    def train(self, mode: bool = True):
        return self
    

class DiscriminatorLoss:
    def __init__(self, use_wgan: bool = False) -> None:
        self._use_wgan = use_wgan
        
        
    def _wgan_loss(predicted, real):
        wgan_objective = real.mean() - predicted.mean()
        return - wgan_objective
