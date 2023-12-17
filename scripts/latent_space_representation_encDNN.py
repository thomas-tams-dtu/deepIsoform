#!/usr/bin/env python3

import torch
import IsoDatasets as IsoDatasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import *
from VAE2 import VAE_lf
import argparse

# CHANGE PROJECT_DIR TO LOCATION OF deepIsoform
PROJECT_DIR =f'/zhome/99/d/155947/DeeplearningProject/deepIsoform'

# Arguments
parser = argparse.ArgumentParser(description='Writes the encoded feature space produced by a 2 latent feature encoder')
parser.add_argument('-mp', type=str, help='Path to VAE model checkpoint')
parser.add_argument('-bs', type=int, help='Batch size to lazy load dataset')
parser.add_argument('-o', type=str, help='Output path for the embeddings')
args = parser.parse_args()

# Initialize training parameters
ENCODER_PATH = args.mp
BATCH_SIZE = args.bs
OUTPUT_PATH = args.o
print('encoder model path', ENCODER_PATH)
print('output path', OUTPUT_PATH)


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
gtx_test_dataloader = DataLoader(gtex_test, batch_size=100, shuffle=True)

### INIT ENCODER MODEL
# Grab a sample to initialize latent features and output size for network
gene_expr, isoform_expr, _ = next(iter(gtx_train_dataloader))

# Encoder model
encoder_model = VAE_lf(input_shape=gene_expr[0].size(),
                       hidden_features=128,
                       latent_features=2)
checkpoint = torch.load(ENCODER_PATH)
encoder_model.load_state_dict(checkpoint['model_state_dict'])

print('VAE structure')
print(encoder_model)


# If GPU available send to gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

encoder_model = encoder_model.to(device)

# Save latent feature space data
latent_file = open(OUTPUT_PATH, 'w')

# Go through each batch in the training dataset using the loader
for x, _, tissues in tqdm(gtx_train_dataloader):
    x = x.to(device)

    # Encode input to latent space
    mu, logvar = encoder_model.encode_mu_var(x)
    z = encoder_model.reparameterize(mu, logvar)

    z1 = z[: , 0].tolist()
    z2 = z[: , 1].tolist()
    
    tissue_list = [tissue.decode('utf-8') for tissue in tissues]

    combined_data = list(zip(z1, z2, tissue_list))
    for z1, z2, lab in combined_data:
        latent_file.write(f'{z1}\t{z2}\t{lab}\n')
    

for x, _, tissues in tqdm(gtx_val_dataloader):
    x = x.to(device)

    # Encode input to latent space
    mu, logvar = encoder_model.encode_mu_var(x)
    z = encoder_model.reparameterize(mu, logvar)

    z1 = z[: , 0].tolist()
    z2 = z[: , 1].tolist()

    tissue_list = [tissue.decode('utf-8') for tissue in tissues]

    combined_data = list(zip(z1, z2, tissue_list))
    for z1, z2, lab in combined_data:
        latent_file.write(f'{z1}\t{z2}\t{lab}\n')

for x, _, tissues in tqdm(gtx_test_dataloader):
    x = x.to(device)

    # Encode input to latent space
    mu, logvar = encoder_model.encode_mu_var(x)
    z = encoder_model.reparameterize(mu, logvar)

    z1 = z[: , 0].tolist()
    z2 = z[: , 1].tolist()

    tissue_list = [tissue.decode('utf-8') for tissue in tissues]

    combined_data = list(zip(z1, z2, tissue_list))
    for z1, z2, lab in combined_data:
        latent_file.write(f'{z1}\t{z2}\t{lab}\n')

latent_file.close()

