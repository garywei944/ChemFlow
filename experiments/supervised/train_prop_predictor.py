import os
import pandas as pd
import numpy as np
from pandarallel import pandarallel

import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, TensorDataset
from lightning import LightningModule, LightningDataModule
import lightning as L
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from tqdm import tqdm

from pathlib import Path

from dict_hash import sha256

from scipy.stats import linregress, pearsonr, spearmanr

from cd2root import cd2root

cd2root()

from src.vae import load_vae, VAE, MolDataModule
from src.utils.scores import *
from src.predictor import Predictor


class Model(LightningModule):
    def __init__(
        self, vae: VAE = None, optimizer: str = "sgd", learning_rate: float = 1e-3
    ):
        super().__init__()
        self.vae = vae
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        self.cls = Predictor(vae.max_len * vae.vocab_size)
        self.loss_fn = nn.MSELoss()

        self.val_y_hat = []
        self.val_y = []

    def forward(self, x: Tensor):
        # return self.cls(self.vae.decode(self.vae.encode(x)[0]).exp())
        return self.cls(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        loss = self.loss_fn(self(x).squeeze(), y)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()

        loss = self.loss_fn(y_hat, y)
        self.val_y_hat.append(y_hat)
        self.val_y.append(y)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_y_hat.clear()
        self.val_y.clear()

    def on_validation_epoch_end(self) -> None:
        # Compute linear regression r coefficient
        val_y_hat = torch.cat(self.val_y_hat).flatten().detach().cpu().numpy()
        val_y = torch.cat(self.val_y).flatten().detach().cpu().numpy()

        try:
            r = linregress(val_y_hat, val_y).rvalue
        except ValueError:
            r = 0.0

        self.log("lin_reg_r", r, prog_bar=True)

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = optim.AdamW(self.cls.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            optimizer = optim.SGD(
                self.cls.parameters(), lr=self.learning_rate, weight_decay=1e-2
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer} not supported")

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 15], gamma=0.1
        )

        return [optimizer], [scheduler]


class PropDataset(LightningDataModule):
    def __init__(
        self,
        vae: VAE = None,
        dm: MolDataModule = None,
        prop: str = "plogp",  # this one should be "1err" or "2iik" for binding affinity
        n: int = 110_000,
        batch_size: int = 1_000,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        output_dir: str = "data/processed/prop",
        seed: int = 42,
        binding_affinity: bool = False,
    ):
        super().__init__()
        self.vae = vae
        self.dm = dm
        self.prop = prop
        self.n = n
        self.batch_size = batch_size
        self.device = device
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.binding_affinity = binding_affinity

        self.save_hyperparameters(
            ignore=["vae", "dm", "device", "prop_fn", "output_dir"]
        )

        # self.id = sha256(self.hparams)

    def setup(self, stage=None):
        file_name = f"data/interim/props/prop_predictor_{self.n}_seed{self.seed}"

        if self.binding_affinity:
            print(f"Binding affinity for {self.prop}")
            file_name += "_binding_affinity"

        x = torch.load(f"{file_name}.pt")
        df = pd.read_csv(f"{file_name}.csv")

        # Gary: The binding affinity values results in very abnormal sum, so we
        # replace all positive values with 0
        if self.binding_affinity:
            df[self.prop] = df[self.prop].apply(lambda e: min(e, 0))

        # normalize the input and target data
        self.y_mean = df[self.prop].mean()
        self.y_std = df[self.prop].std()

        print(f"{self.prop} mean: {self.y_mean}, std: {self.y_std}")
        y = (df[self.prop].values - self.y_mean) / self.y_std
        # y = df[self.prop].values

        # self.dataset = TensorDataset(x, torch.tensor(df[self.prop].values).float())
        self.dataset = TensorDataset(x, torch.tensor(y).float())
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset,
            [10 / 11, 1 / 11],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


def parse_args():
    parser = LightningArgumentParser()

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=20,
    )
    parser.add_argument("-lb", "--load-best", action="store_true")
    parser.add_argument("-tlr", "--tune-learning-rate", action="store_true")
    parser.add_lightning_class_args(Model, "model")
    parser.add_lightning_class_args(PropDataset, "data")

    args = parser.parse_args()

    del args.model.vae, args.data.vae, args.data.dm

    return args


def main():
    args = parse_args()

    # L.seed_everything(args.seed)

    dm, vae = load_vae(
        file_path="data/processed/zmc.smi",
        model_path="checkpoints/vae/zmc/checkpoint.pt",
    )

    dm_prop = PropDataset(vae=vae, dm=dm, **args.data)
    dm_prop.prepare_data()
    dm_prop.setup()

    L.seed_everything(dm_prop.seed)

    model = Model(vae=vae, **args.model)
    # summary(model)

    name = f"{dm_prop.prop}-{args.model.learning_rate}"

    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=WandbLogger(project="soc_prop", entity="soc_mol", name=name),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                # monitor="val_loss",
                monitor="lin_reg_r",
                mode="max",
                save_top_k=1,
                save_last=True,
                save_weights_only=True,
                save_on_train_epoch_end=False,
                dirpath=f"checkpoints/prop_predictor/{dm_prop.prop}",
            ),
            # LearningRateFinder(),
        ],
        # val_check_interval=0.25,
    )

    if args.tune_learning_rate:
        tuner = Tuner(trainer)
        tuner.lr_find(model, dm_prop)

    trainer.fit(model, datamodule=dm_prop)

    if args.load_best:
        model = Model.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path, vae=vae, **args.model
        )
    trainer.validate(model, dm_prop.val_dataloader())

    # save model
    output_dir = Path("checkpoints/prop_predictor") / dm_prop.prop
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.cls.state_dict(), output_dir / "checkpoint.pt")


if __name__ == "__main__":
    # print(torch.cuda.is_initialized())
    # torch.multiprocessing.set_start_method("spawn")
    # pandarallel.initialize(nb_workers=os.cpu_count(), progress_bar=True, verbose=2)
    # print(torch.cuda.is_initialized())
    main()
