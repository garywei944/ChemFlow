from datetime import datetime
from email.policy import strict
import random
from regex import E
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from accelerate.utils import set_seed

import sys
from pathlib import Path
from absl import logging
from tqdm.auto import tqdm, trange
from timeit import default_timer as timer
import pickle
import os
from collections import defaultdict
from pandarallel import pandarallel
from tap import Tap

from rdkit import Chem

from cd2root import cd2root

cd2root()

from src.utils.scores import *
from src.vae import load_vae
from src.pinn.pde import load_wavepde
from src.pinn import PropGenerator, VAEGenerator
from src.predictor import Predictor


class Args(Tap):
    prop: str = "sa"  # property to optimize
    pde: str = "wave"  # pde to solve
    n: int = 1000  # number of molecules to generate


def normalize(x, step_size=None, relative=False):
    if step_size is None:
        return x

    if relative:
        return x * step_size

    try:
        return x / torch.norm(x, dim=-1, keepdim=True) * step_size
    except AttributeError:
        return x


def check_success(
    df: pd.DataFrame,
    smiles: list[str],
    prop: str,
    mode: str = "strict",
    reverse: bool = False,
    both_side: bool = False,
):
    try:

        if len(set(smiles)) <= 2:
            return False

        # TODO: add similarity
        # calculate the similarity using ssim between the first molecule and the rest
        similarity_lst = [ssim(smiles[0], smiles[t]) for t in range(len(smiles))]

        if mode == "strict":
            # if similarity is not monotonicall decreasing, return False
            if not all(
                similarity_lst[i] >= similarity_lst[i + 1]
                for i in range(len(similarity_lst) - 1)
            ):
                return False
        else:
            # if similarity is not monotonicall decreasing, return False
            if not all(
                similarity_lst[i] + 0.1 >= similarity_lst[i + 1]
                for i in range(len(similarity_lst) - 1)
            ):
                return False

        if reverse:
            smiles = smiles[::-1]

        if mode == "strict":
            # print([df.loc[s, prop] for s in smiles])

            r = all(
                df.loc[s1, prop] <= df.loc[s2, prop]
                for s1, s2 in zip(smiles, smiles[1:])
            )
            r = (
                r
                or all(
                    df.loc[s1, prop] >= df.loc[s2, prop]
                    for s1, s2 in zip(smiles, smiles[1:])
                )
                if both_side
                else r
            )
        elif mode == "relaxed_range":
            # print([df.loc[s, prop] for s in smiles])

            r = all(
                df.loc[s1, prop] <= df.loc[s2, prop] + tolerance_range[prop]
                for s1, s2 in zip(smiles, smiles[1:])
            )
            r = (
                r
                or all(
                    df.loc[s1, prop] >= df.loc[s2, prop] - tolerance_range[prop]
                    for s1, s2 in zip(smiles, smiles[1:])
                )
                if both_side
                else r
            )
        elif mode == "relaxed_IQR":
            # print([df.loc[s, prop] for s in smiles])

            r = all(
                df.loc[s1, prop] <= df.loc[s2, prop] + tolerance_IQR[prop]
                for s1, s2 in zip(smiles, smiles[1:])
            )
            r = (
                r
                or all(
                    df.loc[s1, prop] >= df.loc[s2, prop] - tolerance_IQR[prop]
                    for s1, s2 in zip(smiles, smiles[1:])
                )
                if both_side
                else r
            )
        elif mode == "relaxed_std":
            # print([df.loc[s, prop] for s in smiles])

            r = all(
                df.loc[s1, prop] <= df.loc[s2, prop] + tolerance_std[prop]
                for s1, s2 in zip(smiles, smiles[1:])
            )
            r = (
                r
                or all(
                    df.loc[s1, prop] >= df.loc[s2, prop] - tolerance_std[prop]
                    for s1, s2 in zip(smiles, smiles[1:])
                )
                if both_side
                else r
            )
        elif mode == "final":
            r = df.loc[smiles[-1], prop] >= df.loc[smiles[0], prop]
            r = (
                r or df.loc[smiles[-1], prop] <= df.loc[smiles[0], prop]
                if both_side
                else r
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return r

    except Exception as e:
        return 0


def compute_success_rate(
    boundary,
    rand_u_z,
    rand_1d_u_z,
    method: str = "random",
    step_size: float | None = 0.1,
    relative: bool = True,
):
    result_smiles = defaultdict(list)
    unique_smiles = set()

    norms = []

    z = z0.clone()

    # rand_u_z = torch.tensor(np.repeat(np.random.rand(sup_pde.generator.latent_size), 1000, axis=0), dtype=torch.float32).to(device)
    # rand_u_z = normalize(rand_u_z, step_size, relative)
    # rand_u_z = torch.rand(sup_pde.generator.latent_size, dtype=torch.float32).repeat(1000, 1).to(device)

    for t in range(half_range):
        if t == 0:
            u_z = 0
        else:
            if method == "random_1d":
                u_z = rand_1d_u_z
                u_z = normalize(u_z, step_size, relative)
            elif method == "random":
                u_z = rand_u_z
                u_z = normalize(u_z, step_size, relative)
            elif method == "unsup_pde":
                u, u_z = unsup_pde.inference(unsup_pde_idx, z, t % half_range)
                u_z = normalize(u_z, step_size, relative)
            elif method == "sup_pde":
                u, u_z = sup_pde.inference(sup_pde_idx, z, t % half_range)
                u_z = normalize(u_z, step_size, relative)
            elif method == "unsup_hj":
                u, u_z = unsup_hj.inference(unsup_hj_idx, z, t % half_range)
                u_z = normalize(u_z, step_size, relative)
            elif method == "sup_hj":
                u, u_z = sup_hj.inference(sup_hj_idx, z, t % half_range)
                u_z = normalize(u_z, step_size, relative)
            elif method == "fp":
                assert relative and step_size is not None

                z = z.detach().requires_grad_(True)
                u_z = torch.autograd.grad(sup_pde.generator(z).sum(), z)[0]
                u_z = (
                    u_z * step_size
                    + torch.randn_like(u_z) * np.sqrt(2 * step_size) * 0.01
                )
            elif method == "limo":
                z = z.detach().requires_grad_(True)
                u_z = torch.autograd.grad(sup_pde.generator(z).sum(), z)[0]
                u_z = normalize(u_z, step_size, relative)
            elif method == "chemspace":
                u_z = normalize(boundary, step_size, relative)
            else:
                raise ValueError(f"Unknown method {method}")
            norms.append(torch.norm(u_z, dim=-1).mean().item())

        if args.prop in ["sa", "molwt"]:
            u_z = -u_z

        z = z + u_z
        x = vae.decode(z).exp()
        smiles = dm.decode(x)
        # mols = [Chem.MolFromSmiles(s) for s in smiles]
        # props = predictor(x)

        result_smiles[t] = smiles
        # result_smiles[step_size][t] = props.flatten().tolist()
        #
        unique_smiles.update(smiles)

    def func(x: pd.Series):
        # mol = Chem.MolFromSmiles(x["smiles"])
        x[args.prop] = PROP_FN[args.prop](x["smiles"])
        return x

    df = pd.DataFrame({"smiles": sorted(unique_smiles)})
    df = df.parallel_apply(func, axis=1)
    df = df.set_index("smiles")

    success = defaultdict(list)
    for i in range(args.n):
        smiles = [result_smiles[t][i] for t in range(half_range)]
        for mode in ["strict", "relaxed_range", "relaxed_IQR", "relaxed_std", "final"]:
            success[mode].append(
                check_success(
                    df,
                    smiles,
                    args.prop,
                    mode=mode,
                    reverse=args.prop in ["sa", "molwt"],
                )
            )

    success = {k: np.mean(v) for k, v in success.items()}
    success["norm"] = norms

    return success


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    pandarallel.initialize(nb_workers=os.cpu_count(), progress_bar=False, verbose=0)

    args = Args().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm, vae = load_vae(device=device)
    predictor = Predictor(dm.max_len * dm.vocab_size)

    predictor.load_state_dict(
        torch.load(
            f"checkpoints/prop_predictor/{args.prop}/checkpoint.pt", map_location=device
        )
    )
    for p in predictor.parameters():
        p.requires_grad = False

    sup_generator = PropGenerator(vae, predictor).to(device)
    for p in sup_generator.parameters():
        p.requires_grad = False

    unsup_generator = VAEGenerator(vae).to(device)

    unsup_pde = load_wavepde(
        checkpoint=f"checkpoints/wavepde/zmc/checkpoint.pt",
        generator=unsup_generator,
        k=10,
        device=device,
    )
    wave_idx_map = {"plogp": 0, "sa": 6, "qed": 4, "drd2": 2, "jnk3": 0, "gsk3b": 0}
    unsup_pde_idx = wave_idx_map[args.prop]
    for p in unsup_pde.parameters():
        p.requires_grad = False

    sup_pde = load_wavepde(
        checkpoint=f"checkpoints/wavepde_prop/zmc/{args.prop}/checkpoint.pt",
        generator=sup_generator,
        k=1,
        device=device,
    )
    sup_pde_idx = 0
    for p in sup_pde.parameters():
        p.requires_grad = False

    unsup_hj = load_wavepde(
        checkpoint=f"checkpoints/hjpde/zmc/checkpoint.pt",
        generator=unsup_generator,
        k=10,
        device=device,
    )
    hj_idx_map = {"plogp": 5, "sa": 1, "qed": 9, "drd2": 6, "jnk3": 5, "gsk3b": 5}
    unsup_hj_idx = hj_idx_map[args.prop]
    for p in unsup_hj.parameters():
        p.requires_grad = False

    sup_hj = load_wavepde(
        checkpoint=f"checkpoints/hjpde_prop/zmc/{args.prop}/checkpoint.pt",
        generator=sup_generator,
        k=1,
        device=device,
    )
    sup_hj_idx = 0
    for p in sup_hj.parameters():
        p.requires_grad = False

    boundary = np.load(f"src/chemspace/boundaries_zmc/boundary_{args.prop}.npy")
    boundary = torch.tensor(np.repeat(boundary, args.n, axis=0)).to(device)
    # boundary = torch.tensor(boundary, dtype=torch.float32, device=device)

    rand_u_z = torch.randn(sup_pde.generator.latent_size, device=device)
    rand_1d_u_z = torch.zeros(sup_pde.generator.latent_size).to(device)
    rand_1d_u_z[torch.randint(0, sup_pde.generator.latent_size, (1,))] = random.choice(
        [-1, 1]
    )

    k = sup_pde.k
    half_range = sup_pde.half_range
    # half_range = 50
    print(f"half_range: {half_range}")

    set_seed(1991)
    # set_seed(2024)
    z0 = torch.randn((args.n, sup_pde.generator.latent_size), device=device)

    print(f"prop: {args.prop}")

    # load tolerance from pickle file
    with open("experiments/tolerance/relaxed_tolerance_range.pkl", "rb") as f:
        tolerance_range = pickle.load(f)
    with open("experiments/tolerance/relaxed_tolerance_IQR.pkl", "rb") as f:
        tolerance_IQR = pickle.load(f)
    with open("experiments/tolerance/relaxed_tolerance_std.pkl", "rb") as f:
        tolerance_std = pickle.load(f)

    # pandas df save best relative, best step_size, best strict success rate, corresponding relaxed success range rate, corresponding relaxed success IQR rate, corresponding relaxed success std rate, corresponding final success rate
    success_rate_df = pd.DataFrame(
        columns=[
            "method",
            "relative",
            "step_size",
            "result",
            "best strict success rate",
            "relaxed success rate - range",
            "relaxed success rate - IQR",
            "relaxed success rate - std",
            "final success rate",
        ]
    )

    # make a txt file to store the results, dump the results line by line
    with open(f"experiments/success_rate/{args.prop}_{args.n}.txt", "w") as f:
        # record day and time
        now_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Datetime: {now_datetime}\n")
        f.write(
            f"method | relative | step_size | norm | strict | relaxed range | relaxed IQR | relaxed std | final\n"
        )

        for method in [
            "random_1d",
            "random",
            "limo",
            "chemspace",
            "unsup_pde",
            "sup_pde",
            "unsup_hj",
            "sup_hj",
            "fp",
        ]:
            # for method in ["limo"]:
            # create a dict store strict success rate for difference relative and step_size
            strict_success_rate = defaultdict(list)
            for relative in [True]:
                if method == "fp" and not relative:
                    continue
                for step_size in [0.05]:
                    print(
                        f"Running method: {method}, relative: {relative}, step_size: {step_size}"
                    )
                    success = compute_success_rate(
                        boundary,
                        rand_u_z,
                        rand_1d_u_z,
                        method=method,
                        step_size=step_size,
                        relative=relative,
                    )
                    print(
                        f"{method} | {relative} | {step_size} | norm {success['norm']} | strict {success['strict']} | relaxed_range {success['relaxed_range']} | relaxed_IQR {success['relaxed_IQR']} | relaxed_std {success['relaxed_std']} | final {success['final']}\n"
                    )
                    f.write(
                        f"{method} | {relative} | {step_size} | norm {success['norm']} | strict {success['strict']} | relaxed_range {success['relaxed_range']} | relaxed_IQR {success['relaxed_IQR']} | relaxed_std {success['relaxed_std']}  | final {success['final']}\n"
                    )
                    strict_success_rate[(relative, step_size)].extend(
                        [
                            success["strict"],
                            success["relaxed_range"],
                            success["relaxed_IQR"],
                            success["relaxed_std"],
                            success["final"],
                        ]
                    )
            # store the relative and step_size with the highest strict success rate
            best_relative, best_step_size = max(
                strict_success_rate, key=lambda x: strict_success_rate[x][0]
            )
            best_strict_success_rate = strict_success_rate[
                (best_relative, best_step_size)
            ][0]
            corresponding_relaxed_range_success_rate = strict_success_rate[
                (best_relative, best_step_size)
            ][1]
            corresponding_relaxed_IQR_success_rate = strict_success_rate[
                (best_relative, best_step_size)
            ][2]
            corresponding_relaxed_std_success_rate = strict_success_rate[
                (best_relative, best_step_size)
            ][3]
            corresponding_final_success_rate = strict_success_rate[
                (best_relative, best_step_size)
            ][4]
            print(
                f"best relative: {best_relative}, best step_size: {best_step_size}, best strict success rate: {best_strict_success_rate}, corresponding relaxed success rate: relaxed_range {corresponding_relaxed_range_success_rate}, relaxed_IQR {corresponding_relaxed_IQR_success_rate}, relaxed_std {corresponding_relaxed_std_success_rate}, corresponding final succes rate: {corresponding_final_success_rate}\n"
            )
            f.write(
                f"best relative: {best_relative}, best step_size: {best_step_size}, best strict success rate: {best_strict_success_rate}, corresponding relaxed success rate: relaxed_range {corresponding_relaxed_range_success_rate}, relaxed_IQR {corresponding_relaxed_IQR_success_rate}, relaxed_std {corresponding_relaxed_std_success_rate}, corresponding final succes rate: {corresponding_final_success_rate}\n"
            )
            f.write(f"--- | --- | --- | --- | --- | ---\n")

            # save the results to pandas df
            success_rate_df = pd.concat(
                [
                    success_rate_df,
                    pd.DataFrame(
                        [
                            {
                                "method": method,
                                "relative": best_relative,
                                "step_size": best_step_size,
                                "result": "{:.2f}".format(
                                    best_strict_success_rate * 100
                                )
                                + " / "
                                + "{:.2f}".format(
                                    corresponding_relaxed_range_success_rate * 100
                                ),
                                "best strict success rate": best_strict_success_rate
                                * 100,
                                "relaxed success rate - range": corresponding_relaxed_range_success_rate
                                * 100,
                                "relaxed success rate - IQR": corresponding_relaxed_IQR_success_rate
                                * 100,
                                "relaxed success rate - std": corresponding_relaxed_std_success_rate
                                * 100,
                                "final success rate": corresponding_final_success_rate
                                * 100,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

        success_rate_df.to_csv(
            f"experiments/success_rate/{args.prop}_{args.n}.csv",
            float_format="%.2f",
            index=False,
        )
