#!/usr/bin/env python3

import IsoDatasets as IsoDatasets
from torch.utils.data import DataLoader
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
import pickle

N_COMPONENTS = 32

archs4_train = IsoDatasets.Archs4GeneExpressionDataset("/dtu-compute/datasets/iso_02456/hdf5/")
archs4_train_dataloader = DataLoader(archs4_train, batch_size=100, shuffle=True)

ipca = IncrementalPCA(n_components=N_COMPONENTS)

# Train model
c = 0
for X in tqdm(archs4_train_dataloader):
    ipca.partial_fit(X.numpy())

# Save model
with open(f'/zhome/99/d/155947/DeeplearningProject/deepIsoform/models/ipca_model_n{N_COMPONENTS}_small.pkl', 'wb') as file:
    pickle.dump(ipca, file)