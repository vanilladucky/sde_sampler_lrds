from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import nn


class Model(nn.Module):
    """
    Base class for different models.
    """

    def __init__(self, dim: int, dim_out=None):
        super().__init__()

        # Dims
        self.dim = dim
        self.dim_in = dim + 1
        self.dim_out = dim_out or dim

    @staticmethod
    def init_linear(
        layer: nn.Linear,
        bias_init: Callable | None = None,
        weight_init: Callable | None = None
    ):
        if weight_init:
            weight_init(layer.weight)
        if bias_init:
            if ((hasattr(bias_init, '__code__') and 'weight' in bias_init.__code__.co_varnames)) \
                    or ((hasattr(bias_init, 'func') and 'weight' in bias_init.func.__code__.co_varnames)):
                bias_init(layer.bias, weight=layer.weight)
            else:
                bias_init(layer.bias)

    def flatten(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0 or t.shape[0] == 1:
            t = t.expand(x.shape[0], 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        assert x.shape[-1] == self.dim
        assert t.shape == (x.shape[0], 1)
        return torch.cat([t, x], dim=1)


class AngleEncoding(nn.Module):

    def __init__(self, dim=-1):
        super(AngleEncoding, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.concatenate([torch.sin(x), torch.cos(x)], dim=self.dim)


class TimeEmbed(Model):
    def __init__(
        self,
        dim_out: int,
        activation: Callable,
        num_layers: int = 2,
        channels: int = 64,
        last_bias_init: Callable | None = None,
        last_weight_init: Callable | None = None,
    ):
        super().__init__(dim=1, dim_out=dim_out)
        self.channels = channels
        self.activation = activation
        self.register_buffer(
            "timestep_coeff",
            torch.linspace(start=0.1, end=100, steps=self.channels).unsqueeze(0),
            persistent=False,
        )
        self.timestep_phase = nn.Parameter(torch.randn(1, self.channels))
        self.hidden_layer = nn.ModuleList([nn.Linear(2 * self.channels, self.channels)])
        self.hidden_layer += [
            nn.Linear(self.channels, self.channels) for _ in range(num_layers - 2)
        ]
        self.out_layer = nn.Linear(self.channels, self.dim_out)
        Model.init_linear(
            self.out_layer, bias_init=last_bias_init, weight_init=last_weight_init
        )

    def forward(self, t: torch.Tensor, *args) -> torch.Tensor:
        assert t.ndim in [0, 1, 2]
        if t.ndim == 2:
            assert t.shape[1] == 1
        t = t.view(-1, 1).float()
        sin_embed_t = torch.sin((self.timestep_coeff * t) + self.timestep_phase)
        cos_embed_t = torch.cos((self.timestep_coeff * t) + self.timestep_phase)
        assert cos_embed_t.shape == (t.shape[0], self.channels)
        embed_t = torch.cat([sin_embed_t, cos_embed_t], dim=1)
        for layer in self.hidden_layer:
            embed_t = self.activation(layer(embed_t))
        return self.out_layer(embed_t)


class FourierMLP(Model):
    def __init__(
        self,
        dim: int,
        activation: Callable,
        num_layers: int = 4,
        channels: int = 64,
        last_bias_init: Callable | None = None,
        last_weight_init: Callable | None = None,
        use_angle_encoding: bool = False,
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.channels = channels
        self.activation = activation
        if use_angle_encoding:
            self.input_embed = nn.Sequential(
                AngleEncoding(),
                nn.Linear(2 * self.dim, self.channels)
            )
        else:
            self.input_embed = nn.Linear(self.dim, self.channels)
        self.timestep_embed = TimeEmbed(
            dim_out=self.channels,
            activation=self.activation,
            num_layers=2,
            channels=self.channels,
        )
        self.hidden_layer = nn.ModuleList(
            [nn.Linear(self.channels, self.channels) for _ in range(num_layers - 2)]
        )
        self.out_layer = nn.Linear(self.channels, self.dim_out)
        Model.init_linear(
            self.out_layer, bias_init=last_bias_init, weight_init=last_weight_init
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1).expand(x.shape[0], 1).float()
        embed_t = self.timestep_embed(t)
        embed_x = self.input_embed(x)
        assert embed_t.shape == embed_x.shape
        embed = embed_x + embed_t
        for layer in self.hidden_layer:
            embed = layer(self.activation(embed))
        return self.out_layer(self.activation(embed))


class DenseNet(Model):
    def __init__(
        self,
        dim: int,
        arch: list[int],
        activation: Callable,
        last_bias_init: Callable | None = None,
        last_weight_init: Callable | None = None,
        use_angle_encoding: bool = False,
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        if use_angle_encoding:
            first_layer = [AngleEncoding()]
            self.nn_dims = [2 * self.dim_in] + arch
        else:
            first_layer = []
            self.nn_dims = [self.dim_in] + arch
        self.hidden_layer = nn.ModuleList(
            first_layer + [
                nn.Linear(sum(self.nn_dims[: i + 1]), self.nn_dims[i + 1])
                for i in range(len(self.nn_dims) - 1)
            ]
        )
        self.out_layer = nn.Linear(sum(self.nn_dims), self.dim_out)
        Model.init_linear(
            self.out_layer, bias_init=last_bias_init, weight_init=last_weight_init
        )
        self.activation = activation

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        tensor = self.flatten(t, x)
        for layer in self.hidden_layer:
            tensor = torch.cat([tensor, self.activation(layer(tensor))], dim=1)
        return self.out_layer(tensor)
