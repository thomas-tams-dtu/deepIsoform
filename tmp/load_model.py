#!/usr/bin/env python3

import torch
from small_VAE_code import VariationalAutoencoder, VariationalInference

loaded_model_path = '/zhome/99/d/155947/DeeplearningProject/deepIsoform/models/first_test_2000_1000_50'  # The file path where the model was saved

# Load the model and additional information
checkpoint = torch.load(loaded_model_path)


# Access additional information (optional)
info = checkpoint.get('info')
if info:
    architecture = info.get('architecture')
    hyperparameters = info.get('hyperparameters')
    train_loss = info.get('train_loss')
    val_loss = info.get('val_loss')
    init_values = info.get('init_values')

# Create a new instance of your model
model = VariationalAutoencoder(torch.Size([18965]), latent_features=hyperparameters['layer_size'][2])

# Load the model's state dictionary
model.load_state_dict(checkpoint['model_state_dict'])

for loss in train_loss:
    print(loss)

print(val_loss)