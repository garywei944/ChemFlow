import torch
import numpy as np
import pandas as pd
from accelerate.utils import set_seed

from absl import logging
import os
from pandarallel import pandarallel
from tap import Tap
from tqdm import trange
from typing import Literal

from pathlib import Path
import bisect

from cd2root import cd2root

cd2root()

from experiments.utils.traversal_step import Traversal
from src.utils.scores import *
from experiments.utils.utils import partitionIndexes

PROPS = ["plogp", "sa", "qed", "drd2", "jnk3", "gsk3b", "uplogp"]
BINDING_AFFINITY_PROPS = ["1err", "2iik"]


class Args(Tap):
    n: int = 1000  # number of molecules to generate
    steps: int = 10  # number of manipulation steps
    method: Literal["wave_unsup", "hj_unsup"] = "wave_unsup"  # optimization method
    step_size: float = 0.1  # step size
    data_name: str = "zmc"  # data name
    binding_affinity: bool = False  # whether to use binding affinity

    def process_args(self):
        self.model_name = self.method
        self.model_name += f"_{self.step_size}"
        if self.binding_affinity:
            self.model_name += "_binding_affinity"


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    args = Args().parse_args()
    n_workers = torch.cuda.device_count() if args.binding_affinity else os.cpu_count()
    pandarallel.initialize(nb_workers=n_workers, progress_bar=True, verbose=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(42)

    traversal = Traversal(
        method=args.method,
        prop="plogp",
        data_name=args.data_name,
        step_size=args.step_size,
        relative=True,
        device=device,
    )

    z0 = torch.randn(args.n, 1024, device=device)

    results = []

    for idx in trange(10):
        z = z0.clone()
        traversal.idx = idx
        for t in range(args.steps):
            u_z = traversal.step(z, t)
            z += u_z
            smiles = traversal.dm.decode(traversal.vae.decode(z))
            results.extend(
                {
                    "k_idx": idx,
                    "idx": i,
                    "t": t,
                    "smiles": s,
                }
                for i, s in enumerate(smiles)
            )

    df = pd.DataFrame(results)
    df_unique = df[["smiles"]].drop_duplicates("smiles")

    def func(_x: pd.Series):
        if args.binding_affinity:
            device_idx = bisect.bisect_right(chunk_idx, _x.name)
            # print(f"worker {device_idx} working on {_x.name}")
            for prop in BINDING_AFFINITY_PROPS:
                _x[prop] = smiles2affinity(
                    _x["smiles"], PROTEIN_FILES[prop], device_idx=device_idx
                )
        else:
            for prop in PROPS:
                _x[prop] = PROP_FN[prop](_x["smiles"])
        return _x

    print("Calculating properties, this may take a while...")
    # reset index to use it in apply
    df_unique.reset_index(inplace=True)

    print(df_unique.head())
    # pandarallel with split the dataframe into n_cpus chunks
    chunk_idx = list(partitionIndexes(df_unique.shape[0], n_workers))[1:]
    df_unique = df_unique.parallel_apply(func, axis=1)

    # merge with original df to get the original order
    df = df.merge(df_unique, on="smiles", how="left")

    # save results to csv
    output_path = Path("data/interim/corr")
    output_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path / f"{args.model_name}.csv")

    print(f"Results saved to {output_path / args.model_name}.csv")
