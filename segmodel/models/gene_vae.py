import torch
from torch import nn

from ..modules import GeneDecoder, GeneEncoder, ProjectionHead


class GeneVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim=128,
        hidden_dims=(512, 256),
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = GeneEncoder(input_dim, latent_dim, hidden_dims, dropout)
        self.decoder = GeneDecoder(input_dim, latent_dim, hidden_dims)
        self.projector = ProjectionHead(latent_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, 2)

    def encode(self, expression):
        return self.encoder(expression)

    def embedding(self, expression):
        mean, _ = self.encode(expression)
        return self.projector(mean)

    def forward(self, expression):
        mean, logvar = self.encode(expression)
        latent = mean
        if self.training:
            latent = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
        mean_embedding = self.projector(mean)
        task_embedding = self.projector(latent) if self.training else mean_embedding
        return {
            "reconstruction": self.decoder(latent),
            "mean": mean,
            "logvar": logvar,
            "embedding": mean_embedding,
            "task_embedding": task_embedding,
            "spot_logits": self.classifier(task_embedding),
        }
