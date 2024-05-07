from torch import Tensor, nn


class Block(nn.Module):
    """
    residual block
    """

    def __init__(
        self, in_features: int, out_features: int, hidden_features: int = None
    ):
        super().__init__()

        if hidden_features is None:
            hidden_features = out_features

        self.block = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Mish(),
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.Mish(),
            nn.Linear(hidden_features, out_features),
        )

        if in_features != out_features:
            self.skip = nn.Linear(in_features, out_features, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor):
        return self.block(x) + self.skip(x)


class Predictor(nn.Module):
    def __init__(self, latent_size: int, hidden_sizes: list[int] = None):
        # A typical input latent size for zinc250k vae is ~8136
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [1024, 1024]

        layers = [Block(latent_size, hidden_sizes[0])]
        # preact normalization
        layers.extend(
            Block(hidden_sizes[i - 1], hidden_sizes[i])
            for i in range(1, len(hidden_sizes))
        )
        layers.append(nn.Linear(hidden_sizes[-1], 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.mlp(x)
