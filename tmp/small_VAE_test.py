#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader
import numpy as np
import sys
sys.path.insert(1, '/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts')
from chunky_loader import ChunkyDataset


from small_VAE_code import VariationalAutoencoder, VariationalInference
from collections import defaultdict

NROWS_DATASET =         167885    # 167885
NROWS_DATASET_TEST =    20000
NROWS_DATASET_TRAIN =   NROWS_DATASET - NROWS_DATASET_TEST
CHUNK_SIZE =            1000
BATCH_SIZE =            1000
LATENT_FEATURES =       50
LEARNING_RATE =         1e-5
NUM_EPOCHS =            20

print('NROWS_DATASET      ',          NROWS_DATASET)
print('NROWS_DATASET_TRAIN',    NROWS_DATASET_TRAIN)
print('NROWS_DATASET_TEST ',     NROWS_DATASET_TEST)
print('CHUNK_SIZE         ',             CHUNK_SIZE)
print('BATCH_SIZE         ',             BATCH_SIZE)
print('LATENT_FEATURES    ',        LATENT_FEATURES)
print('LEARNING_RATE      ',          LEARNING_RATE)
print('NUM_EPOCHS         ',             NUM_EPOCHS)

init_values = [NROWS_DATASET,
               NROWS_DATASET_TRAIN,
               NROWS_DATASET_TEST,
               CHUNK_SIZE,
               BATCH_SIZE,
               LATENT_FEATURES]

# Construct data sets and data loaders
gz_path ="/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/raw_data/archs4_gene_expression_norm_transposed.tsv.gz"
#gz_path = "/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/head500_archs4_gene_expression_norm_transposed.tsv.gz"
GzChunks_train = ChunkyDataset(file_path=gz_path, nrows=NROWS_DATASET_TRAIN, lines_per_chunk=CHUNK_SIZE, skip_lines=NROWS_DATASET_TEST, got_header=True)
GzChunks_test = ChunkyDataset(file_path=gz_path, nrows=NROWS_DATASET_TEST, lines_per_chunk=CHUNK_SIZE, skip_lines=0, got_header=True)

loader_train = DataLoader(GzChunks_train, batch_size=BATCH_SIZE)
loader_test = DataLoader(GzChunks_test, batch_size=BATCH_SIZE)

data_train, _ = next(iter(loader_train))

# Define the models, evaluator and optimizer
# VAE
vae = VariationalAutoencoder(data_train[0].size(), LATENT_FEATURES)

# Evaluator: Variational Inference
beta = 1
vi = VariationalInference(beta=beta)

# The Adam optimizer works really well with VAEs.
optimizer = torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE)

# define dictionary to store the training curves
training_data = defaultdict(list)
validation_data = defaultdict(list)


print(vae)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

# move the model to the device
vae = vae.to(device)

epoch = 0
val_loss = list()
train_loss = list()
# training..
while epoch < NUM_EPOCHS:
    epoch+= 1
    training_epoch_data = defaultdict(list)
    vae.train()

    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known as it is here
    for x, y in loader_train:
        x = x.to(device)

        # perform a forward pass through the model and compute the ELBO
        loss, diagnostics, outputs = vi(vae, x)

        train_loss.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # gather data for the current bach
        for k, v in diagnostics.items():
            training_epoch_data[k] += [v.mean().item()]


    # gather data for the full epoch
    for k, v in training_epoch_data.items():
        training_data[k] += [np.mean(training_epoch_data[k])]

    # Evaluate on a single batch, do not propagate gradients
    with torch.no_grad():
        vae.eval()

        # Just load a single batch from the test loader
        x, y = next(iter(loader_test))
        x = x.to(device)

        # perform a forward pass through the model and compute the ELBO
        loss, diagnostics, outputs = vi(vae, x)
        val_loss.append(loss)

        # gather data for the validation step
        for k, v in diagnostics.items():
            validation_data[k] += [v.mean().item()]
    



# Assuming 'model' is your PyTorch model
model_path = '/zhome/99/d/155947/DeeplearningProject/deepIsoform/models/first_test_2000_1000_50'  # The file path to save the model

# Create a dictionary to save additional information (optional)
info = {
    'architecture': 'First VAE' ,
    'init_values': init_values,
    'hyperparameters': {
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'layer_size': [2000, 1000, LATENT_FEATURES]
    },
    'train_loss': train_loss,
    'val_loss': val_loss
}

# Save the model and additional information
torch.save({
    'model_state_dict': vae.state_dict(),
    'info': info,
}, model_path)