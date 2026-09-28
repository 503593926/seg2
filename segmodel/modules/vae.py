from torch import nn


class MLP(nn.Sequential):
    def __init__(self, input_features, hidden_dims, dropout=0.0):
        layers = []
        for output_features in hidden_dims:
            layers.extend(
                (
                    nn.Linear(input_features, output_features),
                    nn.LayerNorm(output_features),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            input_features = output_features
        super().__init__(*layers)
        self.output_features = input_features


class GeneEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128, hidden_dims=(512, 256), dropout=0.1):
        super().__init__()
        self.body = MLP(input_dim, hidden_dims, dropout)
        self.mu = nn.Linear(self.body.output_features, latent_dim)
        self.logvar = nn.Linear(self.body.output_features, latent_dim)

    def forward(self, expression):
        hidden = self.body(expression)
        return self.mu(hidden), self.logvar(hidden).clamp(-12, 12)


class GeneDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128, hidden_dims=(512, 256)):
        super().__init__()
        self.body = MLP(latent_dim, reversed(hidden_dims))
        self.head = nn.Linear(self.body.output_features, input_dim)

    def forward(self, latent):
        return self.head(self.body(latent))
