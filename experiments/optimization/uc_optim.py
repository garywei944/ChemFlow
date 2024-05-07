import torch
import numpy as np
import pandas as pd
from accelerate.utils import set_seed

from absl import logging
import os
from collections import defaultdict
from pandarallel import pandarallel
from tap import Tap
from tqdm import trange
from typing import Literal
import bisect

from pathlib import Path


from cd2root import cd2root

cd2root()

from experiments.utils.traversal_step import Traversal
from src.utils.scores import *
from experiments.utils.utils import partitionIndexes

PROTEIN_FILES = {
    "1err": "data/raw/1err/1err.maps.fld",
    "2iik": "data/raw/2iik/2iik.maps.fld",
}


class Args(Tap):
    prop: str = "plogp"  # property to optimize
    n: int = 110_000  # number of molecules to generate
    steps: int = 10  # number of optimization steps
    batch_size: int = 10_000  # batch size
    method: Literal[
        "random",
        "random_1d",
        "fp",
        "limo",
        "chemspace",
        "wave_sup",
        "wave_unsup",
        "hj_sup",
        "hj_unsup",
    ] = "random"  # optimization method
    step_size: float = 0.1  # step size
    relative: bool = False  # relative step size
    data_name: str = "zmc"  # data name
    seed: int = 42
    binding_affinity: bool = False
    limo: bool = False

    def process_args(self):
        self.model_name = self.prop + "_" + self.method
        self.model_name += f"_{self.step_size}"
        self.model_name += "_relative" if self.relative else "_absolute"


if __name__ == "__main__":
    args = Args().parse_args()

    logging.set_verbosity(logging.INFO)
    n_workers = torch.cuda.device_count() if args.binding_affinity else os.cpu_count()
    # pandarallel with split the dataframe into n_cpus chunks
    chunk_idx = list(partitionIndexes(args.n, n_workers))[1:]
    pandarallel.initialize(nb_workers=n_workers, progress_bar=True, verbose=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    traversal = Traversal(
        method=args.method,
        prop=args.prop,
        data_name=args.data_name,
        step_size=args.step_size,
        relative=args.relative,
        minimize=args.prop in MINIMIZE_PROPS or args.binding_affinity,
        device=device,
    )

    set_seed(args.seed)

    z0 = torch.randn(args.n, traversal.vae.latent_dim)

    print(f"prop: {args.prop}")

    smiles = []
    # generate all smiles
    for i in trange(0, args.n, args.batch_size):
        z = z0[i : i + args.batch_size].to(device)

        optimizer = None
        if args.method == "limo" and args.limo:
            z.requires_grad = True
            optimizer = torch.optim.Adam([z], lr=args.step_size)

        for t in range(args.steps):
            # with torch.no_grad():
            #     # Only for research purpose
            #     u = traversal.generator(z)
            #     print(f"step {t+1} u: {u.sum().item()}")
            if args.method == "limo" and args.limo:
                traversal.step(z, t + 1, optimizer=optimizer)
                continue
            u_z = traversal.step(z, t + 1)
            z += u_z
        with torch.no_grad():
            x = traversal.vae.decode(z).exp()
        smiles.extend(traversal.dm.decode(x))

    df = pd.DataFrame(smiles, columns=["smiles"])
    df_unique = df.drop_duplicates()
    if args.binding_affinity:
        file_name = PROTEIN_FILES[args.prop]

    def func(_x: pd.Series):
        if args.binding_affinity:
            device_idx = bisect.bisect_right(chunk_idx, _x.name)
            _x[args.prop] = smiles2affinity(
                _x["smiles"], file_name, device_idx=device_idx
            )
        else:
            _x[args.prop] = PROP_FN[args.prop](_x["smiles"])
        return _x

    print("Calculating properties, this may take a while...")
    df_unique = df_unique.parallel_apply(func, axis=1)

    # merge with original df to get the original order
    df = df.merge(df_unique, on="smiles", how="left")

    # save results to csv
    output_path = Path("data/interim/uc_optim")
    output_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path / f"{args.model_name}.csv")

    print(f"Results saved to {output_path / args.model_name}.csv")
