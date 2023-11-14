#!/usr/bin/env python3

import torch
from small_VAE_code import VariationalAutoencoder, VariationalInference
from scripts.FFNN import FeedForwardIsoform
from scripts.plot_loss import plot_loss


loaded_model_path = '/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/bhole_storage/models/PCA_DENSE_l32_lr0.01_e100'

# Load the model and additional information
checkpoint = torch.load(loaded_model_path)

# Access additional information (optional)
info = checkpoint.get('info')
if info:
    MODEL_NAME = info.get('architecture')
    hyperparameters = info.get('hyperparameters')
    train_loss = info.get('train_loss')
    val_loss = info.get('validation_loss')
    init_values = info.get('init_values')


LATENT_FEATURES = hyperparameters['layer_size'][0][0]
OUTPUT_FEATURES = hyperparameters['layer_size'][-1][1]
FNN = FeedForwardIsoform(input_shape=LATENT_FEATURES, output_shape=OUTPUT_FEATURES)
FNN.load_state_dict(checkpoint['model_state_dict'])

print(FNN)


plot_loss(training_loss=train_loss, validation_loss=val_loss, save_path=f'{MODEL_NAME}_loss_plot.png')