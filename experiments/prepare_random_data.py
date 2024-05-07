# train prop uses 110K data points, generating them and compute the properties on the
# fly is too slow, so we generate them and save them in a csv file
import bisect
import os
import pandas as pd
import numpy as np
from pandarallel import pandarallel

import torch
from tqdm import tqdm, trange
from tap import Tap

from pathlib import Path
from rdkit import Chem

from accelerate.utils import set_seed

from cd2root import cd2root

cd2root()

from src.vae import load_vae, VAE, MolDataModule
from src.utils.scores import *
from experiments.utils.utils import partitionIndexes


class Args(Tap):
    smiles_file: str = "data/processed/zmc.smi"
    vae_path: str = "checkpoints/vae/zmc/checkpoint.pt"
    seed: int = 42
    n: int = 110_000
    batch_size: int = 10_000

    binding_affinity: bool = False

    def process_args(self):
        self.name = f"prop_predictor_{self.n}_seed{self.seed}_vae_pde"
        if self.binding_affinity:
            self.name += "_binding_affinity"


PROPS = ["plogp", "uplogp", "qed", "sa", "gsk3b", "jnk3", "drd2"]
PROTEIN_FILES = {
    "1err": "data/raw/1err/1err.maps.fld",
    "2iik": "data/raw/2iik/2iik.maps.fld",
}


def make_dataset():
    dm, vae = load_vae(
        file_path=args.smiles_file,
        model_path=args.vae_path,
    )

    set_seed(args.seed)

    z0 = torch.randn(args.n, vae.latent_dim)
    x0 = torch.zeros(args.n, dm.max_len * dm.vocab_size)

    smiles = []

    # generate all smiles
    for i in trange(0, args.n, args.batch_size):
        z = z0[i : i + args.batch_size].to(device)
        with torch.no_grad():
            x = vae.decode(z).exp()
        x0[i : i + args.batch_size] = x
        smiles.extend(dm.decode(x))

    # save x0
    torch.save(x0, output_dir / f"{args.name}.pt")

    return smiles


def main():
    smiles = make_dataset()
    # pandarallel with split the dataframe into n_cpus chunks
    chunk_idx = list(partitionIndexes(args.n, n_cpus))[1:]

    def func(_x: pd.Series):
        if args.binding_affinity:
            # pandarallel with split the dataframe into n_cpus chunks
            device_idx = bisect.bisect_right(chunk_idx, _x.name)
            mol = Chem.MolFromSmiles(_x["smiles"])
            if mol is None:
                return _x

            for name, file in PROTEIN_FILES.items():
                _x[name] = smiles2affinity(_x["smiles"], file, device_idx=device_idx)
        else:
            for prop in PROPS:
                _x[prop] = PROP_FN[prop](_x["smiles"])
        return _x

    df = pd.DataFrame(smiles, columns=["smiles"])

    print("Finished generating smiles")
    print("Starting to compute properties")

    if args.binding_affinity:
        # tqdm.pandas()
        # df = df.progress_apply(func, axis=1)
        df = df.parallel_apply(func, axis=1)
    else:
        df = df.parallel_apply(func, axis=1)

    df = df.set_index("smiles")
    df.to_csv(output_dir / f"{args.name}.csv")


if __name__ == "__main__":
    args = Args().parse_args()

    n_cpus = torch.cuda.device_count() if args.binding_affinity else os.cpu_count()

    if args.binding_affinity:
        print(f"Using {n_cpus} GPUs for binding affinity computation")

    pandarallel.initialize(
        # nb_workers=8 if args.binding_affinity else os.cpu_count(),
        nb_workers=n_cpus,
        progress_bar=True,
        verbose=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path("data/interim/props")
    output_dir.mkdir(parents=True, exist_ok=True)

    main()
