import torch
from torch import nn, Tensor
from lightning.pytorch import LightningModule
from torch.autograd import grad
from torch.autograd.functional import jvp

from attrs import define, asdict

import random

from src.pinn import AuxClassifier, Generator
from src.pinn.pde import PDE, MLP


@define
class WavePDEResult:
    loss: Tensor
    energy: Tensor
    latent1: Tensor
    latent2: Tensor
    loss_ic: Tensor
    loss_pde: Tensor
    loss_jvp: Tensor
    loss_cls: Tensor = None


class WavePDE(PDE):
    def __init__(
        self,
        k: int,
        generator: Generator = None,
        time_steps: int = 20,
        n_in: int = 1024,
        pde_function: str = "wave",
        normalize: float | None = None,
        minimize_jvp: bool = False,
    ):
        assert time_steps > 0
        assert pde_function in {"wave", "hj"}

        super().__init__()

        self.k = k
        self.time_steps = time_steps
        self.half_range = time_steps // 2
        self.pde_function = pde_function
        self.normalize = normalize
        self.minimize_jvp = minimize_jvp

        self.generator = generator

        self.mlp = nn.ModuleList([MLP(n_in=n_in, n_out=1) for _ in range(k)])
        self.c = nn.Parameter(torch.ones(k))

    def forward(self, idx: int, z: Tensor, t: int) -> WavePDEResult:
        assert t < self.half_range, f"t={t} must be less than {self.half_range}"

        # mlp = self.mlp[idx]
        c = self.c[idx]

        z = z.clone().requires_grad_()

        loss_ic, loss_pde, loss_jvp = 0, 0, 0
        energy, latent1, latent2 = None, None, None

        for i in range(self.half_range + 2):
            # Gary: use half_range + 2 to be the tick for latent 2
            if i >= self.half_range and latent2 is not None:
                break

            _t = torch.full((1,), i, dtype=z.dtype, device=z.device, requires_grad=True)

            u = self.mlp[idx](z, _t)  # (b, 1)
            u_z = grad(u.sum(), z, create_graph=True)[0]  # (b, n_in)

            # Initial condition loss
            if i == 0:
                loss_ic = u_z.square().mean()

            # PDE loss
            if i < self.half_range:
                u_t = grad(u.sum(), _t, create_graph=True)[0]  # (1,)
                if self.pde_function == "wave":
                    u_tt = grad(u_t.sum(), _t, create_graph=True)[0]  # (1,)
                    u_zz = grad(u_z.sum(), z, create_graph=True)[0]  # (b, n_in)
                    loss_pde += (u_tt - (c**2) * u_zz).square().mean()
                elif self.pde_function == "hj":
                    loss_pde += (u_t + 0.5 * u_z.square().mean()).square().mean()
                else:
                    raise NotImplementedError

            # JVP loss
            if i == t + 1:
                energy = u
                latent1 = z
                _, jvp_value = jvp(
                    self.generator, z, v=u_z, create_graph=True
                )  # (b, g_out_dim)
                if jvp_value.shape[1] == 1:
                    # supervised, jvp_value size: (b, 1)
                    loss_jvp = (jvp_value.sign() * jvp_value.square()).mean()
                    # Fix that for SA, we want to minimize the property
                    if self.minimize_jvp:
                        loss_jvp = -loss_jvp
                else:
                    loss_jvp = jvp_value.square().mean()

            elif i == t + 2:
                latent2 = z

            if self.normalize is not None:
                z = z + u_z / u_z.norm(dim=1, keepdim=True) * self.normalize
            else:
                z = z + u_z

        loss = loss_ic + loss_pde / self.half_range - loss_jvp

        return WavePDEResult(
            loss=loss,
            energy=energy,
            latent1=latent1,
            latent2=latent2,
            loss_ic=loss_ic,
            loss_pde=loss_pde,
            loss_jvp=loss_jvp,
        )

    def inference(self, idx: int, z: Tensor, t: Tensor | int) -> tuple[Tensor, Tensor]:
        z = z.clone().requires_grad_()

        if isinstance(t, int):
            t = torch.full((1,), t, dtype=z.dtype, device=z.device)

        u = self.mlp[idx](z, t)  # (b, 1)
        u_z = grad(u.sum(), z, create_graph=True)[0]  # (b, n_in)

        return u, u_z


class WavePDEModel(LightningModule):
    def __init__(
        self,
        generator: Generator = None,
        k: int = 10,
        time_steps: int = 20,
        n_in: int = 1024,
        pde_function: str = "wave",
        normalize: float | None = None,
        learning_rate: float = 1e-3,
        minimize_jvp: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=("generator",))

        self.k = k
        self.pde = WavePDE(
            k, generator, time_steps, n_in, pde_function, normalize, minimize_jvp
        )
        self.learning_rate = learning_rate
        self.generator = generator

        # Only train the aux classifier if there are multiple trajectories
        if k > 1:
            self.aux_cls = AuxClassifier(generator.reverse_size, k)
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, idx: int, z: Tensor, t: int, positive: bool = True):
        result = self.pde(idx, z, t)

        if self.k > 1:
            # Randomly choosing positive or negative traversal
            if positive:
                mol_shifted = self.generator(result.latent1)
                mol_shifted2 = self.generator(result.latent2)
            else:
                mol_shifted = self.generator(2 * z - result.latent2)
                mol_shifted2 = self.generator(2 * z - result.latent1)

            pred = self.aux_cls(torch.cat([mol_shifted, mol_shifted2], dim=1))
            result.loss_cls = self.loss_fn(
                pred, torch.full((z.shape[0],), idx, device=self.device)
            )
            result.loss += result.loss_cls

        return result

    def step(
        self,
        batch: tuple[Tensor],
        batch_idx: int,
        deterministic: bool = False,
        stage: str = None,
    ):
        (z,) = batch
        if deterministic:
            idx = batch_idx % self.pde.k
            t = batch_idx % self.pde.half_range
            positive = batch_idx % 2 == 0
        else:
            idx = random.randint(0, self.pde.k - 1)
            t = random.randint(0, self.pde.half_range - 1)
            positive = random.random() < 0.5

        results = self(idx, z, t, positive=positive)

        log = {
            f"{stage}/loss": results.loss,
            f"{stage}/loss_ic": results.loss_ic,
            f"{stage}/loss_pde": results.loss_pde,
            f"{stage}/loss_jvp": results.loss_jvp,
        }
        self.log_dict(log, on_epoch=True)

        return results.loss

    def training_step(self, batch: tuple[Tensor], batch_idx: int):
        return self.step(batch, batch_idx, stage="train")

    def validation_step(self, batch: tuple[Tensor], batch_idx: int):
        return self.step(batch, batch_idx, deterministic=True, stage="val")

    def on_validation_epoch_start(self) -> None:
        torch.set_grad_enabled(True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=2, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
        # return torch.optim.SGD(self.parameters(), lr=self.learning_rate)


def load_wavepde(
    checkpoint: str = "checkpoints/wavepde/zinc250k/checkpoint.pt",
    generator: Generator = None,
    k: int = 10,
    time_steps: int = 20,
    n_in: int = 1024,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> WavePDE:
    model = WavePDE(k, generator, time_steps, n_in).to(device)
    model.load_state_dict(torch.load(checkpoint))

    for param in model.parameters():
        param.requires_grad = False

    return model


if __name__ == "__main__":
    # wavepde = WavePDE(1)

    from cd2root import cd2root

    cd2root()

    from src.vae.vae import VAE
    from src.vae.datamodule import MolDataModule
    from src.pinn.generator import VAEGenerator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = MolDataModule()
    dm.prepare_data()
    dm.setup()

    obj = torch.load("checkpoints/vae/zinc250k/checkpoint.pt")
    vae = VAE(dm.max_len, dm.vocab_size).to(device)
    vae.load_state_dict(torch.load("checkpoints/vae/zinc250k/checkpoint.pt"))
    vae.eval()

    generator = VAEGenerator(vae).to(device)

    wavepde = WavePDE(1, generator).to(device)

    z = torch.randn(10, 1024, device=device)

    print(wavepde(0, z, 7))
    print(wavepde.inference(0, z, 7))
