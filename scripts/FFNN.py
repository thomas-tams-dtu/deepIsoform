import torch
from torch import nn
import numpy as np
from typing import *

class FeedForwardIsoform_small(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """

    def __init__(self, input_shape:torch.Size, output_shape:torch.Size) -> None:
        super(FeedForwardIsoform_small, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_features = np.prod(input_shape)
        self.output_features = np.prod(output_shape)

        self.FNN = nn.Sequential(
            nn.Linear(in_features=self.input_features, out_features=512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(in_features=1024, out_features=self.output_features)
        )

        for layer in self.FNN:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                #for param in layer.parameters():
                #    print(f"Parameter dtype: {param.dtype}")

    def forward(self, x) -> Dict[str, Any]:
        # flatten the input
        x = x.view(x.size(0), -1)
        x = self.FNN(x)

        return x

class FeedForwardIsoform_medium(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """

    def __init__(self, input_shape:torch.Size, output_shape:torch.Size) -> None:
        super(FeedForwardIsoform_medium, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_features = np.prod(input_shape)
        self.output_features = np.prod(output_shape)

        self.FNN = nn.Sequential(
            nn.Linear(in_features=self.input_features, out_features=512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=1024, out_features=self.output_features)
        )

        for layer in self.FNN:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                #for param in layer.parameters():
                #    print(f"Parameter dtype: {param.dtype}")

    def forward(self, x) -> Dict[str, Any]:
        # flatten the input
        x = x.view(x.size(0), -1)
        x = self.FNN(x)

        return x

class FeedForwardIsoform_large(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """

    def __init__(self, input_shape:torch.Size, output_shape:torch.Size) -> None:
        super(FeedForwardIsoform_large, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_features = np.prod(input_shape)
        self.output_features = np.prod(output_shape)

        self.FNN = nn.Sequential(
            nn.Linear(in_features=self.input_features, out_features=1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=2048, out_features=self.output_features)
        )

        for layer in self.FNN:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                #for param in layer.parameters():
                #    print(f"Parameter dtype: {param.dtype}")

    def forward(self, x) -> Dict[str, Any]:
        # flatten the input
        x = x.view(x.size(0), -1)
        x = self.FNN(x)

        return x
