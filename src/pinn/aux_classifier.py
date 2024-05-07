from torch import nn, Tensor


class AuxClassifier(nn.Module):
    def __init__(
        self, latent_size: int, num_classes: int, hidden_sizes: list[int] = None
    ):
        # A typical input latent size for zinc250k vae is ~4510
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [512, 256]

        layers = [nn.Linear(latent_size * 2, hidden_sizes[0]), nn.ReLU()]
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))

        # TODO: maybe we can try to use GNN
        # https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/code2/conv.py#L10-L34
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        """

        :param x: concatenated representation of [m_origin, m2_transformed]
        :return:
        """
        return self.mlp(x)
