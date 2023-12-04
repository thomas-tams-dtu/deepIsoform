#!/usr/bin/env python3

#from FFNN import FeedForwardIsoform_small, FeedForwardIsoform_medium, FeedForwardIsoform_large, FeedForwardIsoform_XL, FeedForwardIsoform_XXL
from VAE import VAE_lf, VAE, loss_function
import numpy as np
import torch.nn.functional as F
import torch

# Set seed for reproducibility
np.random.seed(42)

# Generate two random numeric vectors of length 10
vector1 = np.random.randn(10)
vector2 = np.random.randn(10)

# Convert NumPy arrays to PyTorch tensors
tensor1 = torch.tensor(vector1, dtype=torch.float32)
tensor2 = torch.tensor(vector2, dtype=torch.float32)

# Calculate Mean Squared Error (MSE)
mse_loss = F.mse_loss(tensor1, tensor2, reduction="sum")

print(mse_loss)