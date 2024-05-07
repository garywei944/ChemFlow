import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import lightning as L
from pandarallel import pandarallel

import os
import contextlib
from pathlib import Path
from attrs import define
from tqdm.auto import tqdm

import selfies as sf


NOP = "[nop]"


@define
class MolDataset(Dataset):
    dataset: list[list[int]]
    selfies: list[str]
    smiles: list[str]
    vocab: np.ndarray[str]
    token_to_id: dict[str, int]
    max_len: int

    def __attrs_pre_init__(self):
        super().__init__()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return torch.tensor(
            self.dataset[idx]
            + [self.token_to_id[NOP]] * (self.max_len - len(self.dataset[idx]))
        )

    def size(self, idx: int):
        if idx == 0:
            return len(self.dataset)
        elif idx == 1:
            return self.max_len
        else:
            raise ValueError("idx must be 0 or 1")


def make_dataset(file: Path, length_limit: int = 128) -> MolDataset:
    """
    Convert SMILES strings to SELFIES and tokenize them
    :param file: input .smi file
    :param length_limit: maximum length of SELFIES strings
    :return: MolDatasetCheckpoint
    """
    pandarallel.initialize(
        nb_workers=os.cpu_count(),
        progress_bar=True,
    )

    # Load SMILES raw data and convert to SELFIES
    smiles = pd.read_csv(file, header=None, names=["smiles"])

    print(f"Loaded {len(smiles)} SMILES strings from {file}")
    print(f"Example SMILES: {smiles['smiles'].sample(5).tolist()}")

    print("Converting SMILES to SELFIES...")

    def func(x: pd.Series):
        with contextlib.suppress(Exception):
            x["selfies"] = sf.encoder(x["smiles"])
            x["tokens"] = list(sf.split_selfies(x["selfies"]))
            x["vocab"] = set(x["tokens"])
            x["length"] = len(x["tokens"])
        return x

    df = (
        smiles.parallel_apply(func, axis=1).dropna().drop_duplicates(subset=["selfies"])
    )

    # Filter out any SELFIES strings that are too long
    max_len = min(length_limit, df["length"].max())
    df = df.query(f"length <= {max_len}")

    print(f"Max length: {max_len}")

    # update vocab
    vocab = set()
    for v in df["vocab"]:
        vocab.update(v)
    vocab = np.array([NOP] + sorted(vocab))

    token_to_id = {t: i for i, t in enumerate(vocab)}
    selfies = df["selfies"].tolist()
    smiles = df["smiles"].tolist()
    tokens = df.tokens.tolist()

    # Save memory
    del df

    print("Tokenizing... This may take a while.")

    # Gary: the following code will almost definitely exceed memory, don't use
    # multiprocessing for now.

    # def func(x: list[str]):
    #     return [token_to_id.get(t, 0) for t in x]
    #
    # dataset = tokens.parallel_apply(func).tolist()

    dataset = [[token_to_id.get(t, 0) for t in x] for x in tqdm(tokens)]

    return MolDataset(
        dataset=dataset,
        selfies=selfies,
        smiles=smiles,
        vocab=vocab,
        token_to_id=token_to_id,
        max_len=max_len,
    )


class MolDataModule(L.LightningDataModule):
    dataset: MolDataset
    train_data: MolDataset
    test_data: MolDataset

    max_len: int
    vocab_size: int

    def __init__(
        self,
        file: str = "data/processed/zmc.smi",
        batch_size: int = 1024,
        length_limit: int = 72,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.file = Path(file)
        self.name = self.file.stem
        self.length_limit = length_limit

        self.path = Path("data/processed") / f"{self.name}_data.pt"

    def prepare_data(self) -> None:
        try:
            dataset = torch.load(self.path)
            print(f"Loaded tokenized dataset from {self.path}")
        except Exception as err:
            print(f"Failed to load tokenized dataset from {self.path}: {err}")
            print(f"Tokenizing dataset and saving to {self.path}")

            dataset = make_dataset(self.file, self.length_limit)
            torch.save(dataset, self.path)

    def setup(self, stage: str = None) -> None:
        self.dataset = torch.load(self.path)
        self.max_len = self.dataset.max_len
        self.vocab_size = len(self.dataset.vocab)

        self.train_data, self.test_data = random_split(
            self.dataset,
            [0.9, 0.1],
            generator=torch.Generator().manual_seed(42),
        )

    def __len__(self):
        return len(self.dataset)

    def encode(self, x: str | list[str]) -> Tensor:
        if isinstance(x, str):
            x = [x]

        results = torch.zeros((len(x), self.max_len))

        x = [sf.encoder(s) for s in x]
        for i, s in enumerate(x):
            tokens = torch.tensor(
                [self.dataset.token_to_id.get(t, 0) for t in sf.split_selfies(s)]
            )
            results[i, : len(tokens)] = tokens

        results = results.int()

        return results if len(x) > 1 else results.squeeze(0)

    def decode(self, x: Tensor, progress_bar: bool = False) -> str | list[str]:
        token_ids = x.reshape(-1, self.max_len, self.vocab_size).argmax(-1)
        # selfies = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

        smiles = []
        for t in token_ids:
            s = "".join([self.dataset.vocab[i] for i in t if i != 0])
            s = sf.decoder(s)
            smiles.append(s)

        return smiles if x.ndim > 1 else smiles[0]

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=4,
        )


if __name__ == "__main__":
    from cd2root import cd2root

    cd2root()

    # dm = MolDataModule(file="data/processed/moses.smi")
    # dm = MolDataModule(file="data/processed/chembl.smi")
    dm = MolDataModule(file="data/processed/zmc.smi")
    # dm = MolDataModule()
    # dm.prepare_data()
    dm.setup()

    print(dm.vocab_size)
    print(dm.dataset.vocab)

    print(dm.max_len)
    # print(dm.tokenizer.vocab_size)

    for x in dm.train_dataloader():
        print(x)
        break

    s = "[C][C][Branch1][C][C][Branch1][C][C][C][=C][C][=C][O][C][=C][Branch1][S][C][C][=Branch1][C][=O][N][C][=C][C][=C][C][=C][Ring1][=Branch1][F][C][Ring1][S][=C][Ring2][Ring1][Ring2]"
    print(list(sf.split_selfies(s)))
    # tokens = dm.tokenizer.encode(
    #     s,
    #     add_special_tokens=False,
    #     padding="max_length",
    #     max_length=1024,
    #     truncation=True,
    # )
    # print(tokens)
    # print(dm.tokenizer.decode(tokens))

    x = torch.randn(dm.max_len * dm.vocab_size)

    print(dm.decode(x))

    print(len(dm))

    print(dm.encode(["C", "CC", "CCC"]))
