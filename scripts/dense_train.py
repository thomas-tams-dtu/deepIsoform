#!/usr/bin/env python3

import torch
import IsoDatasets as IsoDatasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import *
from FFNN import FeedForwardIsoform_small, FeedForwardIsoform_medium, FeedForwardIsoform_large
from write_training_data import write_training_data
from collections import defaultdict
import pickle
from plot_loss import plot_loss
import argparse
import sys

# Arguments
parser = argparse.ArgumentParser(description='Training dense neural network')
parser.add_argument('-ns', type=str, help='Network size. Choose between small, medium, large')
parser.add_argument('-lf', type=int, help='Latents features used for encoding. Choose between 16, 32, 64, 128, 256')
parser.add_argument('-wd', type=float, help='Weight decay used for Adam optimizer')
parser.add_argument('--sm', action='store_true', help='Save model')
args = parser.parse_args()


# Initialize training parameters
NETWORK_SIZE = args.ns
SAVE_MODEL = args.sm
LATENT_FEATURES = args.lf
BATCH_SIZE = 500
LEARNING_RATE = 1e-4
WEIGHT_DECAY = args.wd   # 1e-5
NUM_EPOCHS = 100
PATIENCE = 6
MODEL_NAME = f'PCA_DENSE_l{LATENT_FEATURES}_lr{LEARNING_RATE}_e{NUM_EPOCHS}_wd{WEIGHT_DECAY}_p{PATIENCE}'
IPCA = f'/zhome/99/d/155947/DeeplearningProject/deepIsoform/models/ipca_model_n{LATENT_FEATURES}.pkl'
METADATA_SAVE_PATH = f'/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/training_meta_data/train_data_{NETWORK_SIZE}.tsv'
PLOT_PATH = f'/zhome/99/d/155947/DeeplearningProject/deepIsoform/model_plots/dense_train/{MODEL_NAME}_loss_plot.png'
MODEL_PATH = f'/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/bhole_storage/models/{MODEL_NAME}'

# Check if size is proper
if NETWORK_SIZE not in ['small', 'medium', 'large']:
    print('Network size has to be either small, medium or large. Got', f'\'{NETWORK_SIZE}\'')
    sys.exit(1)

# Check if latent features are proper
if LATENT_FEATURES not in [16, 32, 64, 128, 256]:
    print(f'Number of latents features has to be 16, 32, 64, 128, 256. Got \'{LATENT_FEATURES}\'')
    sys.exit(1)

# Printing some data to std out
print('model name', MODEL_NAME)
print('PCA model', IPCA)


### INIT DATASET
# Load gtex datasets
gtex_train = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/", exclude='brain')
gtex_test = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/", include='brain')

print("gtex training set size:", len(gtex_train))
print("gtex test set size:", len(gtex_test))

gtx_train_dataloader = DataLoader(gtex_train, batch_size=BATCH_SIZE, shuffle=True)
gtx_test_dataloader = DataLoader(gtex_test, batch_size=BATCH_SIZE, shuffle=True)

### INIT PCA MODEL
# Define PCA model
with open(IPCA, 'rb') as file:
    ipca = pickle.load(file)

### INIT FNN
# Grab a sample to initialize latent features and output size for network
gene_expr, isoform_expr = next(iter(gtx_train_dataloader))

# Select corresponding network
if NETWORK_SIZE == 'small':
    FNN = FeedForwardIsoform_small(input_shape = LATENT_FEATURES, 
                             output_shape = isoform_expr[0].size())
elif NETWORK_SIZE == 'medium':
    FNN = FeedForwardIsoform_medium(input_shape = LATENT_FEATURES, 
                             output_shape = isoform_expr[0].size())
elif NETWORK_SIZE == 'large':
    FNN = FeedForwardIsoform_large(input_shape = LATENT_FEATURES, 
                             output_shape = isoform_expr[0].size())

print(FNN)

### INIT TRAINING 
# Loss function
criterion = torch.nn.MSELoss()

# The Adam optimizer works really well with VAEs.
optimizer = torch.optim.Adam(FNN.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Set up learning rate scheduler
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR by a factor of 0.5 every 10 epochs

# Define list to store the loss
training_loss =   []
validation_loss = []

# If GPU available send to gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

FNN = FNN.to(device)
criterion = criterion.to(device)

### TRAINING ...
best_val_loss = float('inf')
epoch = 0
while epoch < NUM_EPOCHS:
    epoch+= 1
    print('Training epoch', epoch)
    training_epoch_data = []
    FNN.train()

    # Go through each batch in the training dataset using the loader
    for x, y in tqdm(gtx_train_dataloader):
        # Send to device and do PCA
        x = ipca.transform(x)
        x = torch.from_numpy(x).float()
        x = x.to(device)
        y = y.to(device)

        # Run through network
        x = FNN.forward(x)

        # Caculate loss and backprop
        loss = criterion(x, y).double()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_epoch_data.append(loss.mean().item())

    training_loss.append(np.mean(training_epoch_data))

    # Evaluate on a single batch, do not propagate gradients
    with torch.no_grad():
        FNN.eval()
        # Grab test data
        x, y = next(iter(gtx_test_dataloader))

        # Run PCA
        x = ipca.transform(x)
        x = torch.from_numpy(x).float()
        x = x.to(device)
        y = y.to(device)

        # Run through network
        x = FNN.forward(x)

        # Calculate loss
        loss = criterion(x, y)

        validation_loss.append(loss.mean().item())
    
    # Early stopping
    if validation_loss[-1] < best_val_loss:
        best_val_loss = validation_loss[-1]
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    # Check if training should stop
    if early_stopping_counter >= PATIENCE:
        print(f"Early stopping! Training stopped at epoch {epoch}.")
        break

### PLOTTING, SAVING METADATA FROM TRAINING AND MODEL
# Plotting val and train data
plot_loss(training_loss=training_loss, validation_loss=validation_loss, save_path=PLOT_PATH)

# Saving model training metadata
write_training_data(file_path=METADATA_SAVE_PATH,
                    network_name=MODEL_NAME,
                    network_size=NETWORK_SIZE,
                    latent_features=LATENT_FEATURES,
                    learning_rate=LEARNING_RATE,
                    weight_decay=WEIGHT_DECAY,
                    patience=PATIENCE,
                    training_runs=epoch,
                    train_loss= training_loss,
                    eval_loss=  validation_loss)

# Saving model
if SAVE_MODEL:
    init_values = {'batch_size': BATCH_SIZE,
                   'num_epochs': NUM_EPOCHS}

    layer_sizes = [(layer.in_features, layer.out_features) for layer in FNN.FNN if isinstance(layer, torch.nn.Linear)]

    # Create a dictionary to save additional information (optional)
    info = {
        'architecture': MODEL_NAME ,
        'init_values': init_values,
        'hyperparameters': {
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'layer_size': layer_sizes
        },
        'patience': PATIENCE,
        'num_epochs': epoch,
        'train_loss': training_loss,
        'validation_loss': validation_loss
    }

    # Save the model and additional information
    torch.save({'model_state_dict': FNN.state_dict(), 'info': info},
                MODEL_PATH)