from abc import ABC
from torch import nn, Tensor

from src.moflow import MoFlow
from src.vae import VAE
from src.predictor import Predictor


class Generator(nn.Module, ABC):
    latent_size: int
    reverse_size: int

    def forward(self, z: Tensor) -> Tensor: ...


class MoFlowGenerator(Generator):
    def __init__(self, model: MoFlow):
        super().__init__()

        self.model = model

        self.latent_size = model.b_size + model.a_size
        self.reverse_size = (
            model.a_n_node * model.a_n_type
            + model.b_n_type * model.a_n_node * model.a_n_node
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.model.differentiable_reverse(z)


class VAEGenerator(Generator):
    def __init__(self, model: VAE):
        super().__init__()

        self.model = model
        self.latent_size = model.latent_dim
        self.reverse_size = model.max_len * model.vocab_size

    def forward(self, z: Tensor) -> Tensor:
        return self.model.decode(z)


class PropGenerator(Generator):
    def __init__(self, vae: VAE, model: Predictor):
        super().__init__()

        self.vae = vae
        self.model = model
        self.latent_size = vae.latent_dim
        self.reverse_size = 1

    def forward(self, z: Tensor) -> Tensor:
        return self.model(self.vae.decode(z).exp())
