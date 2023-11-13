#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader
import multiprocessing
import numpy as np
import sys
from small_VAE_code import VariationalAutoencoder, VariationalInference
from collections import defaultdict

sys.path.insert(1, '/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts')
from hdf5_load import DatasetHDF5
from plotting import make_vae_plots



BATCH_SIZE =            5000
LEARNING_RATE =         1e-5
BETA =                  5
NUM_EPOCHS =            100
LATENT_FEATURES =       16
MODEL_NAME =            f'e_{NUM_EPOCHS}_b{BETA}_l{LEARNING_RATE}_l512_128_{LATENT_FEATURES}'
MAX_GRAD_NORM =         0.5 # None if not use gradient clipping

print('MODEL_NAME         ',             MODEL_NAME)
print('BATCH_SIZE         ',             BATCH_SIZE)
print('LATENT_FEATURES    ',        LATENT_FEATURES)
print('LEARNING_RATE      ',          LEARNING_RATE)
print('BETA               ',                   BETA)
print('NUM_EPOCHS         ',             NUM_EPOCHS)
print('MAX_GRAD_NORM      ',          MAX_GRAD_NORM)

init_values = {'batch_size': BATCH_SIZE,
               'latent_features': LATENT_FEATURES,
               'learning_rate': LEARNING_RATE,
               'beta': BETA}

# Construct data sets and data loaders
hdf5_path_train = "/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/bhole_storage/train_archs4_gene_expression_norm_transposed.hdf5"
hdf5_path_val = "/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/bhole_storage/val_archs4_gene_expression_norm_transposed.hdf5"
hdf5_path_test = "/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/bhole_storage/test_archs4_gene_expression_norm_transposed.hdf5"
hdf5_dataset_train = DatasetHDF5(hdf5_path_train)                       
hdf5_dataset_val = DatasetHDF5(hdf5_path_val)  
hdf5_dataset_test = DatasetHDF5(hdf5_path_test)

loader_train = DataLoader(hdf5_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
loader_val = DataLoader(hdf5_dataset_val, batch_size=BATCH_SIZE, shuffle=True)
loader_test = DataLoader(hdf5_dataset_test, batch_size=BATCH_SIZE, shuffle=True)

data_train, _ = next(iter(loader_train))

# Define the models, evaluator and optimizer
# VAE
vae = VariationalAutoencoder(data_train[0].size(), LATENT_FEATURES)

# Evaluator: Variational Inference
vi = VariationalInference(beta=BETA)

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

# training..
while epoch < NUM_EPOCHS:
    epoch+= 1
    training_epoch_data = defaultdict(list)
    vae.train()

    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known as it is here
    for x, y in loader_train:
        x = x.double()          # Go from float16 to float64
        x = x.to(device)

        # perform a forward pass through the model and compute the ELBO
        loss, diagnostics, outputs = vi(vae, x)

        optimizer.zero_grad()
        loss.backward()

        if MAX_GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(vae.parameters(), MAX_GRAD_NORM)
        

        optimizer.step()

        # gather data for the current batch
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
        x = x.double()
        x = x.to(device)

        # perform a forward pass through the model and compute the ELBO
        loss, diagnostics, outputs = vi(vae, x)

        # gather data for the validation step
        for k, v in diagnostics.items():
            validation_data[k] += [v.mean().item()]
        
        # Using .format() to create even spacing
        make_vae_plots(vae, x, y, outputs, training_data, validation_data, tmp_img=f"/zhome/99/d/155947/DeeplearningProject/deepIsoform/tmp/plots/{MODEL_NAME}_vae_out.png")
        print("loss: {0:10} elbo: {1:10} beta_elbo: {2:10} log_px: {3:10} log_qz: {4:10} log_pz: {5:10} kl: {6:10} {7:3}".format(
                                                                round(loss.item()),
                                                                round(diagnostics['elbo'].mean().item()     )    ,   
                                                                round(diagnostics['beta_elbo'].mean().item())    , 
                                                                round(diagnostics['log_px'].mean().item()   )    ,
                                                                round(diagnostics['log_qz'].mean().item()   )    ,
                                                                round(diagnostics['log_pz'].mean().item()   )    ,
                                                                round(diagnostics['kl'].mean().item()       )    ,
                                                                epoch)
                                                                    )
    



# Assuming 'model' is your PyTorch model
model_path = f'/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/bhole_storage/models/{MODEL_NAME}'  # The file path to save the model

# Create a dictionary to save additional information (optional)
info = {
    'architecture': MODEL_NAME ,
    'init_values': init_values,
    'hyperparameters': {
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'layer_size': [2000, 1000, LATENT_FEATURES]
    },
    'train_data': training_data,
    'validation_data': validation_data
}

# Save the model and additional information
torch.save({
    'model_state_dict': vae.state_dict(),
    'info': info,
}, model_path)