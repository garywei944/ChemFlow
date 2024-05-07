from abc import ABC
import torch
from torch import nn, Tensor

import math


log1e4 = math.log(10_000)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        assert (
            dim % 2 == 0
        ), "SinusoidalPositionEmbeddings only returns even number of dimensions"

        super(SinusoidalPositionEmbeddings, self).__init__()
        self.half_dim = dim // 2
        scale = log1e4 / (self.half_dim - 1)
        self.embedding = nn.Parameter(torch.arange(self.half_dim).float() * -scale)

    def forward(self, t: Tensor):
        x = torch.outer(t, self.embedding)
        x = torch.cat((x.sin(), x.cos()), dim=-1)
        return x


class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int = 1, h: int = 512):
        super(MLP, self).__init__()

        self.x_embedding = nn.Sequential(
            nn.Linear(n_in, h),
            nn.ReLU(),
        )  # (b, n_in) -> (b, h)

        self.t_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(n_in),
            nn.Linear(n_in, h),
            nn.ReLU(),
            nn.Linear(h, h),
        )  # (b,) -> (b, h)

        dims = [h, h, n_out]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)  # (b, h) -> (b, n_out)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Parametrized potential energy function u(x, t)
        :param x: (b, n_in)
        :param t: (b,)
        :return: (b, n_out) the potential energy u(x, t)
        """
        x = self.x_embedding(x)
        t = self.t_embedding(t)

        return self.mlp(x + t)


class PDE(nn.Module, ABC):
    generator: nn.Module

    k: int  # number of different semantically disentangled latent trajectories
    time_steps: int  # number of time steps in each trajectory

    def forward(self, idx: int, z: Tensor, t: int):
        """
        Return the loss, potential energy u(z, t), z_{t+1}, and z_{t+2}
        :param idx: index of the MLP
        :param z: latent vector
        :param t: time
        :return:
        """

    def inference(self, idx: int, z: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """
        Return the potential energy u(z, t) and the potential flow u_z(z, t)
        :param idx: index of the MLP
        :param z: latent vector
        :param t: time
        :return: u(z, t), u_z(z, t)
        """
