#!/usr/bin/env python3

import IsoDatasets as IsoDatasets
from torch.utils.data import DataLoader
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
import pickle
import argparse

parser = argparse.ArgumentParser(description='Training of PCA models on archs4 gene expression data')
parser.add_argument('-nc', type=int, help='Number of components wanted for the PCA')
args = parser.parse_args()

N_COMPONENTS = args.nc

# Archs4 dataloader
archs4_train = IsoDatasets.Archs4GeneExpressionDataset("/dtu-compute/datasets/iso_02456/hdf5/")
archs4_train_dataloader = DataLoader(archs4_train, batch_size=5000, shuffle=True)

# Setup IPCA
ipca = IncrementalPCA(n_components=N_COMPONENTS)

# Train model
for X in tqdm(archs4_train_dataloader):
    ipca.partial_fit(X.numpy())

# Save model
save_path = f'/zhome/99/d/155947/DeeplearningProject/deepIsoform/models/ipca_model_n{N_COMPONENTS}.pkl'
with open(save_path, 'wb') as file:
    pickle.dump(ipca, file)

print('model saved to: ', save_path)