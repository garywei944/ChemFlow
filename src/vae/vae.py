import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch import optim
import lightning as L


class VAE(L.LightningModule):
    def __init__(
        self,
        max_len: int = None,
        vocab_size: int = None,
        latent_dim: int = 1024,
        embedding_dim: int = 128,
        learning_rate: float = 1e-3,
        hidden_sizes: list[int] = None,
        p: float = 0.1,
        padding_idx: int = 0,
    ):
        if hidden_sizes is None:
            hidden_sizes = [4096, 2048, 1024]

        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.p = p

        self.save_hyperparameters()

        def _block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.Mish())
            return layers

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )

        encoder_layers = _block(
            max_len * embedding_dim, hidden_sizes[0], normalize=False
        )
        for i in range(len(hidden_sizes) - 1):
            encoder_layers.extend(_block(hidden_sizes[i], hidden_sizes[i + 1]))
        encoder_layers.append(nn.Linear(hidden_sizes[-1], latent_dim * 2))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = _block(latent_dim, hidden_sizes[-1])
        for i in range(len(hidden_sizes) - 1, 0, -1):
            decoder_layers.extend(_block(hidden_sizes[i], hidden_sizes[i - 1]))
        decoder_layers.append(nn.Linear(hidden_sizes[0], max_len * vocab_size))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.encoder(self.embedding(x).view((len(x), -1))).view(
            (-1, 2, self.latent_dim)
        )
        mu, log_var = x[:, 0, :], x[:, 1, :]
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        return mu + eps * std, mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        return F.log_softmax(
            self.decoder(z).reshape((-1, self.max_len, self.vocab_size)), dim=2
        ).reshape((-1, self.max_len * self.vocab_size))

    def forward(self, x):
        z, mu, log_var = self.encode(x)
        return self.decode(z), z, mu, log_var

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        # cosine scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
        # return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)

    def loss_function(self, pred, target, mu, log_var, batch_size, p=0.1):
        nll = F.nll_loss(pred, target)
        kld = (
            -0.5
            * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            / (batch_size * pred.shape[1])
        )
        return (1 - p) * nll + p * kld, nll, kld

    def training_step(self, train_batch, batch_idx):
        out, z, mu, log_var = self(train_batch)
        loss, nll, kld = self.loss_function(
            out.reshape((-1, self.vocab_size)),
            train_batch.flatten(),
            mu,
            log_var,
            len(train_batch),
            self.p,
        )
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_nll", nll, sync_dist=True)
        self.log("train_kld", kld, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        out, z, mu, log_var = self(val_batch)
        loss, nll, kld = self.loss_function(
            out.reshape((-1, self.vocab_size)),
            val_batch.flatten(),
            mu,
            log_var,
            len(val_batch),
            0.5,
        )
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_nll", nll, sync_dist=True)
        self.log("val_kld", kld, sync_dist=True)
        self.log("val_mu", torch.mean(mu), sync_dist=True)
        self.log("val_logvar", torch.mean(log_var), sync_dist=True)
        return loss
