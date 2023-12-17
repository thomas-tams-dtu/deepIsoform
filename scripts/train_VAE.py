#!/usr/bin/env python3

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import IsoDatasets as IsoDatasets
from VAE2 import VAE_lf, loss_function
from write_training_data import write_training_data
import argparse
import time

parser = argparse.ArgumentParser(description='Training of VAE on the Archs4 dataset')
parser.add_argument('-e', type=int, help='Number of epochs to train')
parser.add_argument('-hl', type=int, help='Hidden layers size')
parser.add_argument('-lf', type=int, help='Latents features used for encoding. Choose between 16, 32, 64, 128, 256, 512, 1024')
parser.add_argument('-lr', type=float, help='Learning rate used for AdamW optimizer')
parser.add_argument('-b', type=float, help="Beta value used KL-divergence in loss function")
parser.add_argument('-bs', type=int, help='Batch size for dataloader')
parser.add_argument('-p', type=int, help='Patience for early stopping')
parser.add_argument('--sm', action='store_true', help='Save model')
args = parser.parse_args()

BATCH_SIZE = args.bs        # 500
NUM_EPOCHS = args.e         # 100
HIDDEN_SIZE = args.hl       # 128, 256, 512, 1024, ect.
LATENT_FEATURES = args.lf   # 16, 32, 64, 128, 256, 512, ect. 
LEARNING_RATE = args.lr     # 1e-3
BETA = args.b               # 0.5
PATIENCE = args.p           # 6
SAVE_MODEL = args.sm

print('BATCH_SIZE        ', BATCH_SIZE      )
print('NUM_EPOCHS        ', NUM_EPOCHS      )
print('HIDDEN_SIZE       ', HIDDEN_SIZE     )
print('LATENT_FEATURES   ', LATENT_FEATURES )
print('LEARNING_RATE     ', LEARNING_RATE   )
print('BETA              ', BETA            )
print('SAVE_MODEL        ', SAVE_MODEL      )
print('PATIENCE          ', PATIENCE        )

# CHANGE PROJECT_DIR TO LOCATION OF deepIsoform
PROJECT_DIR = f'/zhome/99/d/155947/DeeplearningProject/deepIsoform'
#MODEL_NAME = f'VAE_e{NUM_EPOCHS}_lf{LATENT_FEATURES}_b{BETA}_hl{HIDDEN_SIZE}_lr{LEARNING_RATE}'
MODEL_NAME = f'my_VAE_e{NUM_EPOCHS}_lf{LATENT_FEATURES}_b{BETA}_hl{HIDDEN_SIZE}_lr{LEARNING_RATE}'
MODEL_SAVE_PATH = f'{PROJECT_DIR}/data/bhole_storage/models/{MODEL_NAME}'
METADATA_SAVE_PATH = f'{PROJECT_DIR}/data/bhole_storage/training_meta_data/my_VAE_train_metadata.tsv'

# Set up dataset and dataloader for archs4 data
archs4_dataset_train = IsoDatasets.Archs4GeneExpressionDataset('/dtu-compute/datasets/iso_02456/hdf5-row-sorted/')
archs4_dataloader_train = DataLoader(archs4_dataset_train, batch_size = BATCH_SIZE)
archs4_dataset_val = IsoDatasets.Archs4GeneExpressionDataset('/dtu-compute/datasets/iso_02456/hdf5-row-sorted/',
                                                             validation_set=True)
archs4_dataloader_val = DataLoader(archs4_dataset_val, batch_size = BATCH_SIZE)

# Define VAE model
gene_expr = next(iter(archs4_dataloader_train))
vae = VAE_lf(input_shape=gene_expr[0].size(), hidden_features=HIDDEN_SIZE, latent_features=LATENT_FEATURES)
print(vae)

# Count parameters in model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(vae)
print('NUM PARAM', num_params)

# Define the optimizer
optimizer = optim.AdamW(vae.parameters(), lr=LEARNING_RATE)

# Send to device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")
vae = vae.to(device)

# List for training metadata
training_time = []
training_loss = []
training_recon_loss = []
training_beta_kl_loss= []
validation_loss = []
validation_recon_loss = []
validation_beta_kl_loss = []
best_val_loss = float('inf')
early_stopping_counter = 0

#### TRAINING
epoch = 0
while epoch < NUM_EPOCHS:
    epoch += 1
    epoch_loss = []
    epoch_recon = []
    epoch_beta_kl_loss = []
    start_time = time.time()

    for X in tqdm(archs4_dataloader_train):   #tqdm(archs4_train_dataloader):
        # Zero the gradients
        X = X.to(device)
        optimizer.zero_grad()

        # Forward pass
        output, z, mu, logvar = vae(X)

        # Calculate loss
        loss, recon_loss, beta_kl_loss = loss_function(output, X, mu, logvar, beta=BETA)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Save loss
        epoch_loss.append(loss.item())
        epoch_recon.append(recon_loss.item())
        epoch_beta_kl_loss.append(beta_kl_loss.item())

    training_time.append(time.time() - start_time)
    training_loss.append(np.mean(epoch_loss))
    training_recon_loss.append(np.mean(epoch_recon))
    training_beta_kl_loss.append(np.mean(epoch_beta_kl_loss))

    # Evaluate on a single batch, do not propagate gradients
    with torch.no_grad():
        vae.eval()
        # Grab test data
        # Zero the gradients
        X = X.to(device)
        optimizer.zero_grad()

        # Forward pass
        output, z, mu, logvar = vae(X)

        # Calculate loss
        loss, recon_loss, beta_kl_loss = loss_function(output, X, mu, logvar, beta=BETA)

        # Save loss
        validation_loss.append(loss.item())
        validation_recon_loss.append(recon_loss.item())
        validation_beta_kl_loss.append(beta_kl_loss.item())

    # Early stopping
    if validation_loss[-1] < best_val_loss:
        best_val_loss = validation_loss[-1]
        early_stopping_counter = 0
        print('best val loss', best_val_loss)
    else:
        early_stopping_counter += 1

    # Check if training should stop
    if early_stopping_counter >= PATIENCE:
        print(f"Early stopping! Training stopped at epoch {epoch}.")
        break

### PLOTTING, SAVING METADATA FROM TRAINING AND MODEL
# Saving training metadata
metadata_dictionary = {'network_name' : MODEL_NAME,
                       'num_params' : num_params,
                       'batch_size' : BATCH_SIZE,
                       'latent_features': LATENT_FEATURES,
                       'beta' : BETA,
                       'learning_rate' : LEARNING_RATE,
                       'patience' : PATIENCE,
                       'training_runs' : epoch,
                       'training_time' : training_time,
                       'train_loss' : training_loss,
                       'val_loss' : validation_loss,
                       'training_recon_loss' : training_recon_loss,
                       'training_beta_kl_loss' : training_beta_kl_loss,
                       'validation_recon_loss' : validation_recon_loss,
                       'validation_beta_kl_loss': validation_beta_kl_loss
                       }

write_training_data(file_path=METADATA_SAVE_PATH, metadata_dict=metadata_dictionary)

# Saving model
if SAVE_MODEL:
    # Create a dictionary to save additional information (optional)
    # Save the model and additional information
    torch.save({'model_state_dict': vae.state_dict(),
                'training_metadata': metadata_dictionary},
                MODEL_SAVE_PATH)
