import torch
from torch import nn
import numpy as np
from typing import *

class FeedForwardIsoform(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """

    def __init__(self, input_shape:torch.Size, output_shape:torch.Size) -> None:
        super(FeedForwardIsoform, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_features = np.prod(input_shape)
        self.output_features = np.prod(output_shape)

        self.FNN = nn.Sequential(
            nn.Linear(in_features=self.input_features, out_features=512).double(),
            nn.ReLU(),
            #nn.Linear(in_features=128, out_features=128).double(),
            #nn.ReLU(),
            nn.Linear(in_features=512, out_features=512).double(),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.output_features).double()
        )

        for layer in self.FNN:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""

        # flatten the input
        x = x.view(x.size(0), -1)
        x = self.FNN(x)

        return x
