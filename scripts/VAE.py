import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


# Inspired by https://mbernste.github.io/posts/vae/

# Define the loss function
def loss_function(output, x, mu, logvar, beta = 1.0):
    recon_loss = F.mse_loss(output, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, beta * kl_loss


class VAE(nn.Module):
    def __init__(self, input_shape:torch.Size, hidden_features:int, latent_features:int) -> None:
        super(VAE, self).__init__()

        self.input_shape = input_shape
        self.input_features = np.prod(input_shape)
        self.hidden_features = hidden_features
        self.latent_features = latent_features
        

        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.input_features, out_features=self.hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_features, out_features=int(self.hidden_features/2)),
            nn.ReLU(),
            nn.Linear(in_features=int(self.hidden_features/2), out_features=int(self.hidden_features/4)),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.latent_features, out_features=int(self.hidden_features/4)),
            nn.ReLU(),
            nn.Linear(in_features=int(self.hidden_features/4), out_features=int(self.hidden_features/2)),
            nn.ReLU(),
            nn.Linear(in_features=int(self.hidden_features/2), out_features=self.hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=self.input_features),
        )

        # Mu and logvar layers
        self.enc_mu = nn.Linear(int(self.hidden_features/4), self.latent_features)
        self.enc_logvar = nn.Linear(int(self.hidden_features/4), self.latent_features)

        # Xavier init
        torch.nn.init.xavier_uniform_(self.enc_mu.weight)
        torch.nn.init.xavier_uniform_(self.enc_logvar.weight)

        for layer in self.encoder:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

        for layer in self.decoder:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)


    def encode_mu_var(self, x):
        x = self.encoder(x)
        x_mu, x_logvar = x.chunk(2, dim=-1)
        mu = self.enc_mu(x_mu)
        logvar = self.enc_logvar(x_logvar)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def forward(self, x):
        mu, logvar = self.encode_mu_var(x)
        z = self.reparameterize(mu, logvar)
        output = self.decoder(z)
        return output, z, mu, logvar

class VAE_lf(nn.Module):
    def __init__(self, input_shape:torch.Size, hidden_features:int, latent_features:int) -> None:
        super(VAE_lf, self).__init__()

        self.input_shape = input_shape
        self.input_features = np.prod(input_shape)
        self.hidden_features = hidden_features
        self.latent_features = latent_features
        

        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.input_features, out_features=self.latent_features*16),
            nn.ReLU(),
            nn.Linear(in_features=self.latent_features*16, out_features=self.latent_features*8),
            nn.ReLU(),
            nn.Linear(in_features=self.latent_features*8, out_features=int(self.latent_features*4)),
            nn.ReLU(),
            nn.Linear(in_features=int(self.latent_features*4), out_features=int(self.latent_features*2)),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.latent_features, out_features=int(self.latent_features*2)),
            nn.ReLU(),
            nn.Linear(in_features=int(self.latent_features*2), out_features=int(self.latent_features*4)),
            nn.ReLU(),
            nn.Linear(in_features=int(self.latent_features*4), out_features=self.latent_features*8),
            nn.ReLU(),
            nn.Linear(in_features=int(self.latent_features*8), out_features=self.latent_features*16),
            nn.ReLU(),
            nn.Linear(in_features=self.latent_features*16, out_features=self.input_features),
        )

        # Mu and logvar layers
        self.enc_mu = nn.Linear(int(self.latent_features*2), self.latent_features)
        self.enc_logvar = nn.Linear(int(self.latent_features*2), self.latent_features)

        # Xavier init
        torch.nn.init.xavier_uniform_(self.enc_mu.weight)
        torch.nn.init.xavier_uniform_(self.enc_logvar.weight)

        for layer in self.encoder:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

        for layer in self.decoder:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def encode_mu_var(self, x):
        x = self.encoder(x)
        x_mu, x_logvar = x.chunk(2, dim=-1)
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def forward(self, x):
        mu, logvar = self.encode_mu_var(x)
        z = self.reparameterize(mu, logvar)
        output = self.decoder(z)
        return output, z, mu, logvar