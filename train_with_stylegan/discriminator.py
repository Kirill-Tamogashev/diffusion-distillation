from functools import partial

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(
        self, 
        image_resolution:   int,
        n_classes:          int,
        in_features:        int = 64, 
        img_features:       int = 3,
        lowest_resolution:  int = 4,
        time_features:      int = 64,
        time_countinious:   bool = False,
        n_timesteps:        int = 1000
    ) -> None:
        super().__init__()
        self._in_features = in_features

        self._from_rgb = Conv2dLayer(
            img_features, in_features, kernel_size=1
            )
        conv_map = []
        features = in_features
        while image_resolution != lowest_resolution:
            conv_map.append(
                DBlock(
                    in_features=features, 
                    out_features=features * 2, 
                    kernel_size=3,
                    time_features=time_features,
                    time_countinious=time_countinious,
                    n_timesteps=n_timesteps
                )
            )
            features *= 2
            image_resolution //= 2
        
        self._conv_map = nn.ModuleList(conv_map)
        self._linear_map = DLinearMap(
            in_features=features, n_classes=n_classes, kernel_size=1,
            resolution=image_resolution
        )
    
    def forward(self, x, t):
        x = F.gelu(self._from_rgb(x))
        for layer in self._conv_map:
            x = layer(x, t)
        return self._linear_map(x)


class DBlock(nn.Module):
    def __init__(
        self, 
        in_features:        int, 
        out_features:       int,
        kernel_size:        int,
        stride:             int = 1,
        padding:            int = 1,
        time_features:      int = 64,
        time_countinious:   bool = False,
        n_timesteps:        int = 1000,
    ) -> None:
        super().__init__()
        conv_kwags = dict(stride=stride, padding=padding)
        self._conv1 = Conv2dLayer(
            in_features, out_features, kernel_size, 
            conv_kwags=conv_kwags
        )
        self._conv2 = Conv2dLayer(
            out_features, out_features, kernel_size, 
            downsample=True, conv_kwags=conv_kwags
        )
        
        self._norm1 = SPADE(
            input_features=out_features, time_features=time_features, 
            time_countinious=time_countinious, n_timesteps=n_timesteps
            )
        self._norm2 = SPADE(
            input_features=out_features, time_features=time_features,
            time_countinious=time_countinious, n_timesteps=n_timesteps
            )
        
        self._skip = Conv2dLayer(
            in_features, out_features, kernel_size=1, 
            downsample=False, use_bias=False
        )
        self._scale = 1 / math.sqrt(2)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        
        skip = F.interpolate(
            x, (x.shape[2] // 2, x.shape[3] // 2), 
            mode='bilinear', align_corners=False
        )
        skip = self._skip(skip)
        
        x = self._conv1(x)
        x = self._norm1(x, t)
        x = F.gelu(x)
        
        x = self._conv2(x)
        x = self._norm2(x, t)
        x = F.gelu(x)
        return (x + skip) * self._scale


class SPADE(nn.Module):

    def __init__(
        self, 
        input_features:     int,
        time_features:      int,
        eps:                float = 1e-5, 
        norm_layer_type:    str = "bn_2d",
        time_countinious:   bool = False,
        n_timesteps:        int = 1000,
    ):
        super(SPADE, self).__init__()
        self._time_countinious = time_countinious
        self._time_features = time_features
        
        if time_countinious:
            self._time_embedding = partial(time_map, n_features=time_features)
        else:
            self._time_embedding = nn.Parameter(
                torch.empty(n_timesteps, time_features), 
                requires_grad=True
            )
            nn.init.xavier_normal_(self._time_embedding, 0.02)
        
        # Init biases
        self.conv_weight = nn.Conv2d(time_features, input_features, 
                                     kernel_size=1, stride=1, padding=0)
        self.conv_bias = nn.Conv2d(time_features, input_features, 
                                   kernel_size=1, stride=1, padding=0)

        layer_kwargs = dict(num_features=input_features, eps=eps, affine=False)
        self.norm_layer = _get_norm_layer(norm_layer_type, layer_kwargs)
        
    def _embedd_time(self, t: torch.Tensor) -> torch.Tensor:
        if self._time_countinious:
            embedding = self._time_embedding(t)
        else:
            embedding = self._time_embedding[t]
        return embedding.view(t.size(0), self._time_features, 1, 1)

    def forward(self, inputs: torch.Tensor, t: torch.Tensor):
        t_embedding = self._embedd_time(t)
        outputs = self.norm_layer(inputs)
        gamma = self.conv_weight(t_embedding)
        beta = self.conv_bias(t_embedding)
        return outputs * (gamma + 1.0) + beta
 
def _get_norm_layer(layer_type: str, layer_kwargs: dict):
    # match layer_type:
    if layer_type == "bn_2d":
        return nn.BatchNorm2d(**layer_kwargs)
    elif layer_type == "ln_2d":
        return nn.InstanceNorm2d(**layer_kwargs)
    else:
        raise ValueError(f"Unknown layer_type: {layer_type}")

   
class Conv2dLayer(nn.Module):
    """Conv2d layer with wieghts equalization."""
    def __init__(
        self, 
        in_features:    int, 
        out_features:   int,
        kernel_size:    int,
        use_bias:       bool = True,
        downsample:     bool = False,
        conv_kwags:     dict = {},
        down_kwargs:    dict = dict(mode='bilinear', align_corners=False)
    ) -> None:
        super().__init__()
        
        self._downsample = downsample
        self._down_kwargs = down_kwargs
        self._conv_kwags = conv_kwags
        
        shape = (out_features, in_features, kernel_size, kernel_size)
        self._conv_weight = self._init_conv_weights(shape)
        
        self._bias = nn.Parameter(
                torch.zeros(out_features), requires_grad=True
            ) if use_bias else None

    def _init_conv_weights(self, shape: tuple):
        std = 1 # math.sqrt(2 / (shape[0] + shape[1])) # Xiavier Init
        weight = torch.randn(shape, requires_grad=True) * std
        c = 1 / math.sqrt(np.prod(shape[1:]))
        return nn.Parameter(weight * c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.device, self._conv_weight.device, self._bias.device)
        x = F.conv2d(
            x, weight=self._conv_weight, bias=self._bias, **self._conv_kwags
        )
        if self._downsample:
            x = F.interpolate(
                x, (x.shape[2] // 2, x.shape[3] // 2), 
                **self._down_kwargs
            )
        return x


class FCLayer(nn.Module):
    def __init__(
        self, 
        in_features:    int, 
        out_features:   int, 
        use_bias:       bool = True
    ) -> None:
        super().__init__()
        shape = (out_features, in_features)
        self._weights = self.__init_fc_weights(shape)
        self._bias = None
        if use_bias:
            self._bias = nn.Parameter(
                torch.zeros(out_features), requires_grad=True
            )

    def __init_fc_weights(self, shape: tuple):
        std = 1 # math.sqrt(2 / (shape[0] + shape[1])) # Xiavier Init
        weights = torch.randn(shape, requires_grad=True) * std
        c = 1 / math.sqrt(np.prod(shape[1]))
        return nn.Parameter(weights * c)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, weight=self._weights, bias=self._bias)


class StdFeatureLayer(nn.Module):
    def __init__(self, group_size: int = None):
        super().__init__()
        self._group_size = group_size

    def forward(self, x, eps=1e-8):
        bs, _, height, width = x.size()
        group_size = self._group_size \
            if self._group_size is not None and bs % self._group_size == 0 \
            else bs
        
        std = torch.sqrt(x.view(group_size, -1).var(dim=0) + eps)
        std = std.mean().view(1, 1, 1, 1)
        std = std.expand(bs, -1, height, width)
        return torch.cat([x, std], dim=1)


class DLinearMap(nn.Module):
    def __init__(
        self, 
        in_features:    int, 
        n_classes:      int,
        kernel_size:    int,
        resolution:     int,
    ) -> None:
        super().__init__()
        self._std_feat_layer = StdFeatureLayer()
        self._conv = Conv2dLayer(
            in_features=in_features + 1, out_features=in_features, 
            kernel_size=kernel_size
        )
        self._fc1 = FCLayer(in_features * (resolution ** 2), in_features)
        self._fc2 = FCLayer(in_features, n_classes)
        
    def forward(self, x):
        x = self._std_feat_layer(x)
        x = self._conv(x)
        x = self._fc1(F.gelu(x.view(x.size(0), -1)))
        return self._fc2(F.gelu(x))


def time_map(t, n_features, max_period=10_000):
    log_max_period = math.log(max_period)
    half = n_features // 2
    freqs = torch.exp(
        -log_max_period * torch.arange(0, half, dtype=torch.float32) / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if n_features % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], 
            dim=-1
        )
    return embedding
