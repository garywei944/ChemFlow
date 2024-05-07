import torch
from src.vae import VAE, MolDataModule


def load_vae(
    file_path: str = "data/processed/zmc.smi",
    model_path: str = "checkpoints/vae/zmc/checkpoint.pt",
    latent_dim: int = 1024,
    embedding_dim: int = 128,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[MolDataModule, VAE]:
    dm = MolDataModule(file_path)
    dm.prepare_data()
    dm.setup()

    vae = VAE(
        max_len=dm.max_len,
        vocab_size=dm.vocab_size,
        latent_dim=latent_dim,
        embedding_dim=embedding_dim,
    ).to(device)

    vae.load_state_dict(torch.load(model_path))

    for p in vae.parameters():
        p.requires_grad = False
    vae.eval()

    return dm, vae

def load_vae_demo(
    file_path: str = "data/processed/zmc.smi",
    model_path: str = "checkpoints/vae/zmc/checkpoint.pt",
    latent_dim: int = 1024,
    embedding_dim: int = 128,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[MolDataModule, VAE]:
    dm = MolDataModule(file_path)
    dm.max_len = 72
    dm.vocab_size = 244

    vae = VAE(
        max_len=72,
        vocab_size=244,
        latent_dim=latent_dim,
        embedding_dim=embedding_dim,
    ).to(device)

    vae.load_state_dict(torch.load(model_path))

    for p in vae.parameters():
        p.requires_grad = False
    vae.eval()

    return dm, vae