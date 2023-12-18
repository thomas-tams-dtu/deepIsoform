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
import argparse
import sys
import time

# Arguments
parser = argparse.ArgumentParser(description='Training dense neural network')
parser.add_argument('-ns', type=str, help='Network size. Choose between small, medium, large')
parser.add_argument('-e', type=int, help='Number of epochs to train')
parser.add_argument('-lf', type=int, help='Latents features used for encoding. Choose between 16, 32, 64, 128, 256, 512, 1024')
parser.add_argument('-wd', type=float, help='Weight decay used for Adam optimizer')
parser.add_argument('-bs', type=int, help='Batch size for dataloader')
parser.add_argument('-lr', type=float, help='Learning rate used for Adam optimizer')
parser.add_argument('-p', type=int, help='Patience for early stopping')
parser.add_argument('-b', type=float, help="Beta value used to train encoder network")
parser.add_argument('--sm', action='store_true', help='Save model')
args = parser.parse_args()


# Initialize training parameters
NETWORK_SIZE = args.ns
SAVE_MODEL = args.sm
LATENT_FEATURES = args.lf
BATCH_SIZE = args.bs        # 500
LEARNING_RATE = args.lr     # 1e-4
WEIGHT_DECAY = args.wd      # 1e-5
PATIENCE = args.p           # 6
NUM_EPOCHS = args.e         # 100
BETA = args.b

print('NETWORK_SIZE   ', NETWORK_SIZE       )
print('SAVE_MODEL     ', SAVE_MODEL         )
print('LATENT_FEATURES', LATENT_FEATURES    )
print('BATCH_SIZE     ', BATCH_SIZE         )  # 500
print('LEARNING_RATE  ', LEARNING_RATE      )  # 1e-4
print('WEIGHT_DECAY   ', WEIGHT_DECAY       )  # 1e-5
print('PATIENCE       ', PATIENCE           )  # 6
print('BETA           ', BETA               )
print('NUM_EPOCHS     ', NUM_EPOCHS         )

# CHANGE PROJECT_DIR TO LOCATION OF deepIsoform
PROJECT_DIR =f'/zhome/99/d/155947/DeeplearningProject/deepIsoform'
MODEL_NAME = f'ENCODER_DENSE_l{LATENT_FEATURES}_lr{LEARNING_RATE}_e{NUM_EPOCHS}_wd{WEIGHT_DECAY}_p{PATIENCE}_b{BETA}'
#METADATA_SAVE_PATH = f'{PROJECT_DIR}/data/training_meta_data/encoder_dense_train_metadata_{NETWORK_SIZE}.tsv'
MODEL_PATH = f'{PROJECT_DIR}/data/bhole_storage/models/{MODEL_NAME}'

## Set manual for now
#ENCODER_PATH = f'{PROJECT_DIR}/data/bhole_storage/models/VAE_e100_lf{LATENT_FEATURES}_b{BETA}_hl128_lr0.0001'

ENCODER_PATH = f'{PROJECT_DIR}/data/bhole_storage/models/my_VAE_e30_lf{LATENT_FEATURES}_b{BETA}_hl128_lr0.0001'

METADATA_SAVE_PATH = f'{PROJECT_DIR}/data/bhole_storage/training_meta_data/custom_encoder_dense_train_metadata_lf{LATENT_FEATURES}_{NETWORK_SIZE}.tsv'


print(ENCODER_PATH)


# Check if size is proper
if NETWORK_SIZE not in ['small', 'medium', 'large', 'XL', 'XXL']:
    print('Network size has to be either small, medium or large. Got', f'\'{NETWORK_SIZE}\'')
    sys.exit(1)

# Check if latent features are proper
if LATENT_FEATURES not in [2, 16, 32, 64, 128, 256, 512, 1024, 2048, 4044]:
    print(f'Number of latents features has to be 16, 32, 64, 128, 256, 512, 1024, 2048, 4044. Got \'{LATENT_FEATURES}\'')
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
                       hidden_features=0,
                       latent_features=LATENT_FEATURES)
checkpoint = torch.load(ENCODER_PATH)
encoder_model.load_state_dict(checkpoint['model_state_dict'])

print('VAE structure')
print(encoder_model)


### INIT FNN
# Select corresponding network
if NETWORK_SIZE == 'small':
    fnn = FeedForwardIsoform_small(input_shape = LATENT_FEATURES, 
                             output_shape = isoform_expr[0].size())
elif NETWORK_SIZE == 'medium':
    fnn = FeedForwardIsoform_medium(input_shape = LATENT_FEATURES, 
                             output_shape = isoform_expr[0].size())
elif NETWORK_SIZE == 'large':
    fnn = FeedForwardIsoform_large(input_shape = LATENT_FEATURES, 
                             output_shape = isoform_expr[0].size())
elif NETWORK_SIZE == 'XL':
    fnn = FeedForwardIsoform_XL(input_shape = LATENT_FEATURES, 
                             output_shape = isoform_expr[0].size())
elif NETWORK_SIZE == 'XXL':
    fnn = FeedForwardIsoform_XXL(input_shape = LATENT_FEATURES, 
                             output_shape = isoform_expr[0].size())

print('FNN structure')
print(fnn)

### INIT TRAINING 
# Loss function
criterion = torch.nn.MSELoss()

# The Adam optimizer works really well with VAEs.
optimizer = torch.optim.Adam(fnn.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Set up learning rate scheduler
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR by a factor of 0.5 every 10 epochs

# If GPU available send to gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

fnn = fnn.to(device)
encoder_model = encoder_model.to(device)
criterion = criterion.to(device)

### TRAINING ...
# Define list to store the loss and time
training_loss =   []
validation_loss = []
training_time = []
best_val_loss = float('inf')
epoch = 0
early_stopping_counter = 0

while epoch < NUM_EPOCHS:
    fnn.train()
    epoch+= 1
    train_time = time.time()
    training_epoch_data = []
    print('Training epoch', epoch)
    
    # Go through each batch in the training dataset using the loader
    for x, y, _ in tqdm(gtx_train_dataloader):
        # Send to device
        #x = x.float()
        x = x.to(device)
        y = y.to(device)

        # Encode input to latent space
        mu, logvar = encoder_model.encode_mu_var(x)
        z = encoder_model.reparameterize(mu, logvar)

        # Run through network
        x = fnn.forward(z)

        # Caculate loss and backprop
        loss = criterion(x, y).double()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_epoch_data.append(loss.mean().item())

    training_time.append(time.time() - train_time)
    training_loss.append(np.mean(training_epoch_data))

    # Evaluate on a single batch, do not propagate gradients
    with torch.no_grad():
        fnn.eval()
        # Grab test data
        x, y, _ = next(iter(gtx_val_dataloader))

        # Send to device
        #x = x.float()
        x = x.to(device)
        y = y.to(device)

        # Encode input to latent space
        mu, logvar = encoder_model.encode_mu_var(x)
        z = encoder_model.reparameterize(mu, logvar)

        # Run through network
        x = fnn.forward(z)

        # Caculate loss and backprop
        loss = criterion(x, y).double()

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

### TEST MODEL ON SEPERATE TEST SET
test_loss = []
with torch.no_grad():
    fnn.eval()

    # Go through each batch in the training dataset using the loader
    for x, y, _ in tqdm(gtx_test_dataloader):
        # Send to device
        #x = x.float()
        x = x.to(device)
        y = y.to(device)

        # Encode input to latent space
        mu, logvar = encoder_model.encode_mu_var(x)
        z = encoder_model.reparameterize(mu, logvar)

        # Run through network
        x = fnn.forward(z)

        # Caculate loss and backprop
        loss = criterion(x, y).double()

        test_loss.append(loss.item())
        



### PLOTTING, SAVING METADATA FROM TRAINING AND MODEL
# Count parameters in model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(fnn)

# Saving model training metadata
metadata_dictionary = {
                    'network_name' :MODEL_NAME,
                    'network_size' :NETWORK_SIZE,
                    'num_params':num_params,
                    'batch_size' :BATCH_SIZE,
                    'latent_features' :LATENT_FEATURES,
                    'learning_rate':LEARNING_RATE,
                    'weight_decay' :WEIGHT_DECAY,
                    'beta': BETA,
                    'patience':PATIENCE,
                    'training_runs':epoch,
                    'training_time' :training_time,
                    'train_loss' : training_loss,
                    'eval_loss' : validation_loss,
                    'test_loss' :test_loss
                    }

write_training_data(file_path=METADATA_SAVE_PATH, metadata_dict=metadata_dictionary)


# Saving model
if SAVE_MODEL:
    init_values = {'batch_size': BATCH_SIZE,
                   'num_epochs': NUM_EPOCHS}

    layer_sizes = [(layer.in_features, layer.out_features) for layer in fnn.fnn if isinstance(layer, torch.nn.Linear)]

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
    torch.save({'model_state_dict': fnn.state_dict(), 'info': info},
                MODEL_PATH)
    