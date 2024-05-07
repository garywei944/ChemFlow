import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser

from pathlib import Path

from cd2root import cd2root

cd2root()

# from src.vae.utils import *
from src.vae import VAE
from src.vae.datamodule import MolDataModule


def parse_args():
    parser = LightningArgumentParser()

    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-e", "--epochs", type=int, default=150)
    parser.add_argument("-o", "--output", type=str, default="checkpoints/vae/zmc")

    parser.add_lightning_class_args(VAE, "model")
    parser.add_lightning_class_args(MolDataModule, "data")

    args = parser.parse_args()

    del args.model.max_len, args.model.vocab_size

    return args


def main():
    args = parse_args()

    L.seed_everything(args.seed)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    dm = MolDataModule(**args.data)
    dm.prepare_data()
    dm.setup()

    print("max len:", dm.max_len)
    print("vocab size:", dm.vocab_size)

    vae = VAE(
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
        logger=[WandbLogger(project="soc_vae", entity="soc_mol")],
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
    trainer.fit(vae, datamodule=dm)

    # load best checkpoint
    vae = VAE.load_from_checkpoint(
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
    torch.save(vae.state_dict(), output_path / "checkpoint.pt")


if __name__ == "__main__":
    main()
