import torch
from torch import Tensor
import numpy as np
import random

from cd2root import cd2root

cd2root()

from src.utils.scores import *
from src.vae import load_vae
from src.pinn.pde import load_wavepde
from src.pinn import PropGenerator, VAEGenerator
from src.predictor import Predictor

MODES = [
    "random",
    "random_1d",
    "fp",
    "limo",
    "chemspace",
    "wave_sup",
    "wave_unsup",
    "hj_sup",
    "hj_unsup",
]

WAVEPDE_IDX_MAP = {
    "plogp": 1,
    "sa": 1,
    "qed": 1,
    "drd2": 9,
    "jnk3": 4,
    "gsk3b": 0,
    "uplogp": 1,
    "1err": 2,
    "2iik": 4,
}
HJPDE_IDX_MAP = {
    "plogp": 0,
    "sa": 0,
    "qed": 9,
    "drd2": 2,
    "jnk3": 3,
    "gsk3b": 8,
    "uplogp": 0,
    "1err": 6,
    "2iik": 3,
}


class Traversal:
    """
    Uniformed class to perform 1 step of traversal in latent space
    """

    method: str
    prop: str
    data_name: str
    step_size: float
    relative: bool
    minimize: bool
    device: torch.device

    def __init__(
        self,
        method: str,
        prop: str,
        data_name: str = "zmc",
        step_size: float = 0.1,
        relative: bool = True,
        minimize: bool = False,
        k_idx: int | None = None,  # the index of unsupervised pde to use
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.method = method
        self.prop = prop
        self.data_name = data_name
        self.step_size = step_size
        self.relative = relative
        self.minimize = minimize
        self.device = device

        assert self.method in MODES, f"mode must be one of {MODES}"

        self.dm, self.vae = load_vae(
            file_path=f"data/processed/{self.data_name}.smi",
            model_path=f"checkpoints/vae/{self.data_name}/checkpoint.pt",
            latent_dim=1024,
            embedding_dim=128,
            device=self.device,
        )

        # Generate u_z for random
        if self.method == "random":
            self.u_z = torch.randn(self.vae.latent_dim, device=self.device)
            return
        elif self.method == "random_1d":
            self.u_z = torch.zeros(self.vae.latent_dim, device=self.device)
            self.u_z[random.randint(0, self.vae.latent_dim - 1)] = (
                1 if random.random() < 0.5 else -1
            )
            return
        elif self.method == "chemspace":
            # load boundary for chemspace
            boundary = np.load(
                f"src/chemspace/boundaries_{self.data_name}/boundary_{self.prop}.npy"
            )  # (1, latent_dim)
            self.u_z = torch.tensor(boundary, device=self.device).squeeze()

            return

        if self.method in {"limo", "fp", "wave_sup", "hj_sup"}:
            self.predictor = Predictor(self.dm.max_len * self.dm.vocab_size)
            self.predictor.load_state_dict(
                torch.load(
                    f"checkpoints/prop_predictor/{self.prop}/checkpoint.pt",
                    map_location=self.device,
                )
            )
            for p in self.predictor.parameters():
                p.requires_grad = False

        # LIMO and FP don't need to load the generator
        if self.method in {"limo", "fp"}:
            self.generator = PropGenerator(self.vae, self.predictor).to(self.device)
            return

        # All the other methods are pde related
        pde_name = self.method.split("_")[0]
        if self.method in {"wave_sup", "hj_sup"}:
            self.generator = PropGenerator(self.vae, self.predictor).to(self.device)
            self.pde = load_wavepde(
                checkpoint=f"checkpoints/{pde_name}pde_prop/{self.data_name}/{self.prop}/checkpoint.pt",
                generator=self.generator,
                k=1,
                device=self.device,
            )
            self.idx = 0
        else:
            self.generator = VAEGenerator(self.vae).to(self.device)
            self.pde = load_wavepde(
                checkpoint=f"checkpoints/{pde_name}pde/{self.data_name}/checkpoint.pt",
                generator=self.generator,
                k=10,
                device=self.device,
            )
            if k_idx is not None:
                self.idx = k_idx
            elif pde_name == "wave":
                self.idx = WAVEPDE_IDX_MAP[self.prop]
            elif pde_name == "hj":
                self.idx = HJPDE_IDX_MAP[self.prop]
            else:
                raise ValueError(f"Unknown pde {pde_name}")

        for p in self.pde.parameters():
            p.requires_grad = False

        self.k = self.pde.k
        self.half_range = self.pde.half_range

    def step(self, z: Tensor, t: int = 0, optimizer=None) -> Tensor:
        """
        Perform 1 step of traversal in latent space, return u_z

        When t=0, return 0 tensor
        """
        if t == 0:
            return torch.zeros_like(z)

        if self.method in ["random", "random_1d"]:
            u_z = self.u_z
            u_z = normalize(u_z, self.step_size, self.relative)
        elif self.method == "chemspace":
            u_z = self.u_z
            u_z = normalize(u_z, self.step_size, self.relative)
            if self.minimize:
                u_z = -u_z
        elif self.method == "limo":
            if optimizer is not None:
                return self.limo_optimizer_step(optimizer, z)
            z = z.detach().requires_grad_(True)
            u_z = torch.autograd.grad(self.generator(z).sum(), z)[0]
            u_z = normalize(u_z, self.step_size, self.relative)
            if self.minimize:
                u_z = -u_z
        elif self.method == "fp":
            z = z.detach().requires_grad_(True)
            u_z = torch.autograd.grad(self.generator(z).sum(), z)[0]
            u_z = (
                u_z * self.step_size
                + torch.randn_like(u_z) * np.sqrt(2 * self.step_size) * 0.1
            )
            if self.minimize:
                u_z = -u_z
        else:
            _, u_z = self.pde.inference(self.idx, z, t % self.half_range)
            u_z = normalize(u_z, self.step_size, self.relative)

        return u_z

    def limo_optimizer_step(self, optimizer, z):
        optimizer.zero_grad()
        u = -self.generator(z).sum()
        if self.minimize:
            u = -u
        u.backward()
        optimizer.step()
        return z.grad
