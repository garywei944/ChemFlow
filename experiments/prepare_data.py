# train prop uses 110K data points, generating them and compute the properties on the
# fly is too slow, so we generate them and save them in a csv file

import os
import pandas as pd
import numpy as np
from pandarallel import pandarallel

import torch
from tqdm import tqdm, trange
from tap import Tap

from pathlib import Path

from accelerate.utils import set_seed

from cd2root import cd2root

cd2root()

from src.vae import load_vae, VAE, MolDataModule
from src.utils.scores import *


class Args(Tap):
    smiles_file: Path = "data/processed/zmc.smi"

    binding_affinity: bool = False

    def process_args(self):
        self.name = self.smiles_file.stem
        if self.binding_affinity:
            self.name += "_binding"


PROPS = ["plogp", "uplogp", "qed", "sa", "gsk3b", "jnk3", "drd2"]
PROTEIN_FILES = {
    "1err": "1err/1err.maps.fld",
    "2iik": "2iik/2iik.maps.fld",
}


def main():
    df = pd.read_csv(args.smiles_file, header=None, names=["smiles"])

    def func(_x: pd.Series):
        if args.binding_affinity:
            for name, file in PROTEIN_FILES.items():
                _x[name] = smiles2affinity(_x["smiles"], file)
        else:
            for prop in PROPS:
                _x[prop] = PROP_FN[prop](_x["smiles"])
        return _x

    df = df.parallel_apply(func, axis=1)

    df = df.set_index("smiles")
    df.to_csv(output_dir / f"{args.name}.csv")


if __name__ == "__main__":
    args = Args().parse_args()

    pandarallel.initialize(nb_workers=os.cpu_count(), progress_bar=True, verbose=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path("data/interim/props")
    output_dir.mkdir(parents=True, exist_ok=True)

    main()
