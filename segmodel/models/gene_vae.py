import torch
from torch import nn

from segmodel.modules.vae import GeneDecoder, GeneEncoder


class GeneVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim=128,
        hidden_dims=(512, 256),
        dropout=0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = GeneEncoder(input_dim, latent_dim, hidden_dims, dropout)
        self.decoder = GeneDecoder(input_dim, latent_dim, hidden_dims)

    def encode(self, expression):
        return self.encoder(expression)

    def embedding(self, expression):
        mean, _ = self.encode(expression)
        return mean

    def forward(self, expression):
        mean, logvar = self.encode(expression)
        latent = mean
        if self.training:
            latent = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
        return {
            "reconstruction": self.decoder(latent),
            "mean": mean,
            "logvar": logvar,
            "embedding": mean,
        }
