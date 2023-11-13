#!/usr/bin/env python3

import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from typing import *
from FFNN import FeedForwardIsoform
from collections import defaultdict
import sys
sys.path.insert(1, '/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts')
from hdf5_load import DatasetHDF5_gtex
from plotting import make_vae_plots


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

# gtex_gene_expression_norm_transposed.tsv.gz :     The small paired dataset containing the gene expression  
# gtex_isoform_expression_norm_transposed.tsv.gz :  The small paired dataset containing the isoform expression
# gtex_gene_isoform_annoation.tsv.gz :              The gene-isoform relationship for the small paired dataset
# gtex_annot.tsv.gz :                               The tissue type for each sample in the small dataset


BATCH_SIZE = 200
LATENT_FEATURES = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS = 100
MODEL_NAME = f'PCA_l{LATENT_FEATURES}_lr{LEARNING_RATE}_e{NUM_EPOCHS}'

#hdf5_path_train = "/dtu/blackhole/0b/155947/train_gtex_geneX_isoformY.hdf5"
#hdf5_path_val = "/dtu/blackhole/0b/155947/val_gtex_geneX_isoformY.hdf5"    
#hdf5_path_test = "/dtu/blackhole/0b/155947/test_gtex_geneX_isoformY.hdf5"  
#hdf5_dataset_train = DatasetHDF5(hdf5_path_train)                       
#hdf5_dataset_val = DatasetHDF5(hdf5_path_val)  
#hdf5_dataset_test = DatasetHDF5(hdf5_path_test)
#
#loader_train = DataLoader(hdf5_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
#loader_val = DataLoader(hdf5_dataset_val, batch_size=BATCH_SIZE, shuffle=True)
#loader_test = DataLoader(hdf5_dataset_test, batch_size=BATCH_SIZE, shuffle=True)

hdf5_path_train = "/dtu/blackhole/0b/155947/all_gtex_geneX_isoformY2.hdf5"
all_dataset = DatasetHDF5_gtex(hdf5_path_train)
loader_train = DataLoader(all_dataset, batch_size=BATCH_SIZE, shuffle=True)
loader_val = DataLoader(all_dataset, batch_size=BATCH_SIZE, shuffle=True)


# Define feedfoward model
sklearn_pca = PCA(n_components=LATENT_FEATURES)

gene_expr, isoform_expr = next(iter(loader_train))
FNN = FeedForwardIsoform(input_shape = gene_expr[0].size(), 
                         output_shape = isoform_expr[0].size())


# Loss function
criterion = torch.nn.CrossEntropyLoss()

# The Adam optimizer works really well with VAEs.
optimizer = torch.optim.Adam(FNN.parameters(), lr=LEARNING_RATE)

# define dictionary to store the training curves
training_data = defaultdict(list)
validation_data = defaultdict(list)


print(FNN)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

# move the model to the device
FNN = FNN.to(device)



# training..
epoch = 0
while epoch < NUM_EPOCHS:
    epoch+= 1
    training_epoch_data = defaultdict(list)
    FNN.train()

    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known as it is here
    for gene_expr, isoform_expr in loader_train:
        x = sklearn_pca.fit_transform(gene_expr)
        
        break

        x = x.double()
        x = x.to(device)

        loss = criterion(x, isoform_expr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #training_epoch_data[k] += [v.mean().item()]
    
    break
    #training_data[k] += [np.mean(training_epoch_data[k])]

    # Evaluate on a single batch, do not propagate gradients
    with torch.no_grad():
        FNN.eval()
        gene_expr, isoform_expr = next(iter(loader_test))
        
        x = sklearn_pca.fit_transform(gene_expr)
        
        x = x.double()
        x = x.to(device)

        loss = criterion(x, isoform_expr)

        #training_epoch_data[k] += [loss.mean().item()]

"""

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
"""