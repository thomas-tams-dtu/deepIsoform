#!/usr/bin/env python3

import torch
import IsoDatasets as IsoDatasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import *
from FFNN import FeedForwardIsoform
from collections import defaultdict
import pickle
from plot_loss import plot_loss


def cross_entropy(ys, ts):
    # computing cross entropy per sample
    cross_entropy = -torch.sum(ts * torch.log(ys), dim=1, keepdim=False)
    # averaging over samples
    return torch.mean(cross_entropy)

def accuracy(ys, ts):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    correct_prediction = torch.eq(torch.max(ys, 1)[1], torch.max(ts, 1)[1])
    # averaging the one-hot encoded vector
    return torch.mean(correct_prediction.float())


BATCH_SIZE = 500
LATENT_FEATURES = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100
PATIENCE = 6
MODEL_NAME = f'PCA_DENSE_l{LATENT_FEATURES}_lr{LEARNING_RATE}_e{NUM_EPOCHS}_wd{WEIGHT_DECAY}_p{PATIENCE}_dout_bn_medium+'
IPCA = f'/zhome/99/d/155947/DeeplearningProject/deepIsoform/models/ipca_model_n{LATENT_FEATURES}.pkl'

print('model name', MODEL_NAME)

# Load gtex datasets
gtex_train = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/", exclude='brain')
gtex_test = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5/", include='brain')

print("gtex training set size:", len(gtex_train))
print("gtex test set size:", len(gtex_test))

gtx_train_dataloader = DataLoader(gtex_train, batch_size=BATCH_SIZE, shuffle=True)
gtx_test_dataloader = DataLoader(gtex_test, batch_size=BATCH_SIZE, shuffle=True)

# Define PCA model
with open(IPCA, 'rb') as file:
    ipca = pickle.load(file)

# Define the Dense neural network
gene_expr, isoform_expr = next(iter(gtx_train_dataloader))
FNN = FeedForwardIsoform(input_shape = LATENT_FEATURES, 
                         output_shape = isoform_expr[0].size())

print(FNN)

# Loss function
criterion = torch.nn.MSELoss()

# The Adam optimizer works really well with VAEs.
optimizer = torch.optim.Adam(FNN.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Set up learning rate scheduler
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR by a factor of 0.5 every 10 epochs

# define dictionary to store the training curves
training_data =   [] #defaultdict(list)
validation_data = [] #defaultdict(list)


# If GPU available send to gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

# move the model to the device
FNN = FNN.to(device)
criterion = criterion.to(device)


best_val_loss = float('inf')
# training..
epoch = 0
while epoch < NUM_EPOCHS:
    epoch+= 1
    print('Training epoch', epoch)
    training_epoch_data = [] #[defaultdict(list)]
    FNN.train()

    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known as it is here
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

    training_data.append(np.mean(training_epoch_data))

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

        validation_data.append(loss.mean().item())
    
    # Primitive early stopping
    if validation_data[-1] < best_val_loss:
        best_val_loss = validation_data[-1]
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    # Check if training should stop
    if early_stopping_counter >= PATIENCE:
        print(f"Early stopping! Training stopped at epoch {epoch}.")
        break

### Plotting val and train data
plot_path = f'/zhome/99/d/155947/DeeplearningProject/deepIsoform/model_plots/dense_train/{MODEL_NAME}_loss_plot.png'
plot_loss(training_loss=training_data, validation_loss=validation_data, save_path=plot_path)


### Saving model
# Save path for model
model_path = f'/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/bhole_storage/models/{MODEL_NAME}'

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
    'train_loss': training_data,
    'validation_loss': validation_data
}

# Save the model and additional information
torch.save({'model_state_dict': FNN.state_dict(), 'info': info,},
            model_path)