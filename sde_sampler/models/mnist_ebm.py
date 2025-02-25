# EBM architecture used for MNIST

import torch
from .mlp import TimeEmbed


class MNISTNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n_c, n_f = 1, 14
        self.act = torch.nn.SiLU()
        self.time_mlp_1 = TimeEmbed(
            dim_out=n_f,
            activation=self.act,
            num_layers=2,
            channels=2 * n_f,
        )
        self.conv_1 = torch.nn.Conv2d(n_c, n_f, 3, 1, 1)
        self.time_mlp_2 = TimeEmbed(
            dim_out=2 * n_f,
            activation=self.act,
            num_layers=2,
            channels=4 * n_f,
        )
        self.conv_2 = torch.nn.Conv2d(n_f, n_f * 2, 4, 2, 1)
        self.time_mlp_3 = TimeEmbed(
            dim_out=4 * n_f,
            activation=self.act,
            num_layers=2,
            channels=8 * n_f,
        )
        self.conv_3 = torch.nn.Conv2d(n_f*2, n_f*4, 4, 2, 1)
        self.conv_4 = torch.nn.Conv2d(n_f*4, n_f*8, 4, 2, 1)

    def forward(self, t, x):
        x = x.view((-1, 1, 14, 14))
        x = self.conv_1(x)
        x = self.act(x + self.time_mlp_1(t).unsqueeze(-1).unsqueeze(-1))
        x = self.conv_2(x)
        x = self.act(x + self.time_mlp_2(t).unsqueeze(-1).unsqueeze(-1))
        x = self.conv_3(x)
        x = self.act(x + self.time_mlp_3(t).unsqueeze(-1).unsqueeze(-1))
        x = self.conv_4(x)
        return x.squeeze()
