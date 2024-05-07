import torch
from torch import nn, Tensor
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser

from pathlib import Path
import random

from cd2root import cd2root

cd2root()

# from src.vae.utils import *
from src.vae import VAE
from src.pinn.generator import VAEGenerator
from src.vae.datamodule import MolDataModule
from src.pinn.pde import WavePDEModel


class VAE_PDE_MODEL(L.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        pde_lambda: float = 0.01,
        # VAE parameters
        max_len: int = None,
        vocab_size: int = None,
        latent_dim: int = 1024,
        embedding_dim: int = 128,
        hidden_sizes: list[int] = None,
        p: float = 0.1,
        padding_idx: int = 0,
        # PDE parameters
        k: int = 10,
        time_steps: int = 20,
        n_in: int = 1024,
        pde_function: str = "wave",
        normalize: float | None = None,
        minimize_jvp: bool = False,
    ):
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.pde_lambda = pde_lambda

        super().__init__()
        self.vae = VAE(
            max_len=max_len,
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            embedding_dim=embedding_dim,
            learning_rate=learning_rate,
            hidden_sizes=hidden_sizes,
            p=p,
            padding_idx=padding_idx,
        )
        self.generator = VAEGenerator(self.vae)
        self.pde = WavePDEModel(
            generator=self.generator,
            k=k,
            time_steps=time_steps,
            n_in=n_in,
            pde_function=pde_function,
            normalize=normalize,
            minimize_jvp=minimize_jvp,
        )

    def forward(self, idx: int, x: Tensor, t: int, positive: bool = True):
        vae_loss = self.vae.training_step(x, 0)
        z = self.vae.encode(x)
        pde_loss = self.pde(idx, z, t, positive)
        return vae_loss + pde_loss * self.pde_lambda

    def step(
        self,
        batch: Tensor,
        batch_idx: int,
        deterministic: bool = False,
        stage: str = None,
    ):
        out, z, mu, log_var = self.vae(batch)
        vae_loss, nll, kld = self.vae.loss_function(
            out.reshape((-1, self.vae.vocab_size)),
            batch.flatten(),
            mu,
            log_var,
            len(batch),
            self.vae.p,
        )
        if deterministic:
            idx = batch_idx % self.pde.k
            t = batch_idx % self.pde.pde.half_range
            positive = batch_idx % 2 == 0
        else:
            idx = random.randint(0, self.pde.k - 1)
            t = random.randint(0, self.pde.pde.half_range - 1)
            positive = random.random() < 0.5

        results = self.pde(idx, z, t, positive=positive)

        # Get rid of JVP loss
        pde_loss = (
                       results.loss_ic
                       + results.loss_pde / self.pde.pde.half_range
                       # + results.loss_cls
                   ) * self.pde_lambda

        loss = vae_loss + pde_loss

        log = {
            f"{stage}/vae_loss": vae_loss,
            f"{stage}/nll": nll,
            f"{stage}/kld": kld,
            f"{stage}/pde_loss": pde_loss,
            f"{stage}/loss_ic": results.loss_ic,
            f"{stage}/loss_pde": results.loss_pde,
            # f"{stage}/loss_cls": results.loss_cls,
            # f"{stage}/loss_jvp": results.loss_jvp,
            f"{stage}/loss": loss,
        }
        if stage == "train":
            self.log("loss", loss, sync_dist=True, prog_bar=True)
            self.log("vae_loss", vae_loss, sync_dist=True, prog_bar=True)
            self.log("pde_loss", pde_loss, sync_dist=True, prog_bar=True)
        elif stage == "val":
            self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        self.log_dict(log, on_epoch=True)

        return loss

    def training_step(self, batch: Tensor, batch_idx: int):
        return self.step(batch, batch_idx, stage="train")

    def validation_step(self, batch: Tensor, batch_idx: int):
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


def parse_args():
    parser = LightningArgumentParser()

    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-e", "--epochs", type=int, default=30)
    parser.add_argument("-o", "--output", type=str, default="checkpoints/vae_pde/zmc")

    parser.add_lightning_class_args(VAE_PDE_MODEL, "model")
    parser.add_lightning_class_args(MolDataModule, "data")

    args = parser.parse_args()

    del args.model.max_len, args.model.vocab_size

    return args


def main():
    args = parse_args()

    L.seed_everything(args.seed)

    output_path = Path(args.output)
    # output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "vae").mkdir(parents=True, exist_ok=True)
    (output_path / "pde").mkdir(parents=True, exist_ok=True)

    dm = MolDataModule(**args.data)
    dm.prepare_data()
    dm.setup()

    print("max len:", dm.max_len)
    print("vocab size:", dm.vocab_size)

    model = VAE_PDE_MODEL(
        max_len=dm.max_len,
        vocab_size=dm.vocab_size,
        **args.model,
    )
    trainer = L.Trainer(
        # gpus=1,
        # accelerator="gpu",
        # devices=1,
        max_epochs=args.epochs,
        # logger=L.loggers.CSVLogger("logs"),
        logger=[WandbLogger(project="soc_vae_pde", entity="soc_mol")],
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                dirpath=output_path,
            ),
        ],
        # enable_checkpointing=False,
    )
    print("Training..")
    trainer.fit(model, datamodule=dm)

    # load best checkpoint
    model = VAE_PDE_MODEL.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        max_len=dm.max_len,
        vocab_len=dm.vocab_size,
    )

    # vae = VAE.load_from_checkpoint(
    #     "checkpoints/vae/zmc/epoch=146-step=536109.ckpt",
    #     max_len=dm.max_len,
    #     vocab_len=dm.vocab_size,
    # )

    print("Saving..")
    torch.save(model.vae.state_dict(), output_path / "vae" / "checkpoint.pt")
    torch.save(model.pde.pde.state_dict(), output_path / "pde" / "checkpoint.pt")


if __name__ == "__main__":
    main()
