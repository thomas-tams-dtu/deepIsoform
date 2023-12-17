#!/usr/bin/env python3

import torch
import IsoDatasets as IsoDatasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import *
from FFNN import FeedForwardIsoform_small, FeedForwardIsoform_medium, FeedForwardIsoform_large, FeedForwardIsoform_XL, FeedForwardIsoform_XXL
from VAE2 import VAE_lf
from write_training_data import write_training_data
from collections import defaultdict
import pickle
from plot_loss import plot_loss
import argparse
import sys
import time
import re

# Arguments
parser = argparse.ArgumentParser(description='Writes the encoded feature space produced by a 2 latent feature encoder')


parser.add_argument('-mp', type=int, help='Path to VAE model checkpoint')

args = parser.parse_args()


# Initialize training parameters
ENCODER_MODEL_PATH = args.mp

# CHANGE PROJECT_DIR TO LOCATION OF deepIsoform
PROJECT_DIR =f'/zhome/99/d/155947/DeeplearningProject/deepIsoform'
METADATA_SAVE_PATH = f'{PROJECT_DIR}/data/bhole_storage/training_meta_data/latent_space_rep_encoder.tsv'

## Set manual for now
ENCODER_PATH = f'{PROJECT_DIR}/data/bhole_storage/models/my_VAE_e15_lf{LATENT_FEATURES}_b{BETA}_hl128_lr0.0001'

# Check if latent features are proper
if LATENT_FEATURES != 2:
    print('Only takes 2 encoders that output 2 latent features')
    sys.exit(1)

# Printing some data to std out
print('model name', MODEL_NAME)
print('encoder model', ENCODER_PATH)


### INIT DATASET
# Load gtex datasets
gtex_train = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5-row-sorted/", exclude=['brain', 'Artery'])
gtex_val = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5-row-sorted/", include='brain')
gtex_test = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5-row-sorted/", include='Artery')

print("gtex training set size:", len(gtex_train))
print("gtex validation set size:", len(gtex_val))
print("gtex test set size:", len(gtex_test))


gtx_train_dataloader = DataLoader(gtex_train, batch_size=BATCH_SIZE, shuffle=True)
gtx_val_dataloader = DataLoader(gtex_val, batch_size=BATCH_SIZE, shuffle=True)
gtx_test_dataloader = DataLoader(gtex_test, batch_size=10, shuffle=True)

### INIT ENCODER MODEL
# Grab a sample to initialize latent features and output size for network
gene_expr, isoform_expr, _ = next(iter(gtx_train_dataloader))

# Encoder model
encoder_model = VAE_lf(input_shape=gene_expr[0].size(),
                       hidden_features=128,
                       latent_features=LATENT_FEATURES)
checkpoint = torch.load(ENCODER_PATH)
encoder_model.load_state_dict(checkpoint['model_state_dict'])

print('VAE structure')
print(encoder_model)


# If GPU available send to gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

encoder_model = encoder_model.to(device)

# Save latent feature space data
latent_space_points_path = f"{PROJECT_DIR}/data/bhole_storage/training_meta_data/latent_space_points/encoder_latent_space_b{BETA}_lf{LATENT_FEATURES}.tsv"
latent_file = open(latent_space_points_path, 'w')
latent_file.write('z1\tz2\tTissue\n')

pattern = re.compile(r'^([^\s]+)')

# Go through each batch in the training dataset using the loader
for x, _, tissue in tqdm(gtx_train_dataloader):
    x = x.to(device)

    # Encode input to latent space
    mu, logvar = encoder_model.encode_mu_var(x)
    z = encoder_model.reparameterize(mu, logvar)

    z1 = z[: , 0].tolist()
    z2 = z[: , 1].tolist()
    
    #tissue_list = [pattern.search(element.decode('utf-8')).group(1) for element in tissue]
    tissue_list = tissue.tolist()

    combined_data = list(zip(z1, z2, tissue_list))
    for z1, z2, lab in combined_data:
        latent_file.write(f'{z1}\t{z2}\t{lab}\n')
    

for x, _, label in tqdm(gtx_val_dataloader):
    x = x.to(device)

    # Encode input to latent space
    mu, logvar = encoder_model.encode_mu_var(x)
    z = encoder_model.reparameterize(mu, logvar)

    z1 = z[: , 0].tolist()
    z2 = z[: , 1].tolist()

    #tissue_list = [pattern.search(element.decode('utf-8')).group(1) for element in tissue]
    tissue_list = tissue.tolist()

    combined_data = list(zip(z1, z2, tissue_list))
    for z1, z2, lab in combined_data:
        latent_file.write(f'{z1}\t{z2}\t{lab}\n')

latent_file.close()

