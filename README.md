# ChemFlow: Navigating Chemical Space with Latent Flows

```bibtex
@misc{Wei2024navigating,
    title = {Navigating Chemical Space with Latent Flows},
    author = {Guanghao Wei and Yining Huang and Chenru Duan and Yue Song and Yuanqi Du},
    year = {2024},
    eprint = {N/A},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

## Quick Start

* Install all dependencies with `conda env create -f environment.yml`.
* [Download data](#download-data) and put it in the `data/processed` directory.
* Prepare DataModule to pre-train the model.
* (Optional) Install [AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU) for docking binding affinity. See [Notes on Compiling AutoDock-GPU](#notes-on-compiling-autodock-gpu).


## Download Data & Model Checkpoints

We extract 4,253,577 molecules from the three commonly used datasets for drug discovery
including [MOSES](https://github.com/molecularsets/moses), [ZINC250K](https://zinc.docking.org/)([download](https://www.kaggle.com/datasets/basu369victor/zinc250k/data)),
and [ChEMBL](https://www.ebi.ac.uk/chembl/).
* Download the entire dataset from [Google Drive](https://drive.google.com/file/d/1V0wT2epsG9gF-WtD4XpIc1qBbrg1NUET/view?usp=sharing)(176.9 MiB).
* Data processing notebooks refers to [`notebooks/datasets.ipynb`](notebooks/datasets.ipynb).

## Prepare dataset module

## Notes on Compiling [AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU)

The conda version of `AutoDock-GPU` is not compatible with RTX 3080 & 3090.
So don't use `environment.yml` to install `AutoDock-GPU`.
Make sure to follow
this [issue](https://github.com/ccsb-scripps/AutoDock-GPU/issues/172#issuecomment-1010263229)
to compile the source code.
A good reference to the SM code
is [here](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).

Some commands might be useful:

```bash
export GPU_INCLUDE_PATH=/usr/local/cuda/include
export GPU_LIBRARY_PATH=/usr/local/cuda/lib64

make DEVICE=CUDA NUMWI=128 TARGETS=86
```

To test if the compilation is successful, run the following command:

```bash
obabel -:"CCN(CCCCl)OC1=CC2=C(Cl)C1C3=C2CCCO3" -O demo.pdbqt -p 7.4 --partialcharge gasteiger --gen3d
autodock_gpu_128wi -M data/raw/1err/1err.maps.fld -L demo.pdbqt -s 0 -N demo
```
