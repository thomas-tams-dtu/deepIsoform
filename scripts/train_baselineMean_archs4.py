#!/usr/bin/env python3

import IsoDatasets
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Example of making a training set that excludes samples from the brain and a test set with only samples from the brain
# If you have enough memory, you can load the dataset to memory using the argument load_in_mem=True
archs4_dataset_train = IsoDatasets.Archs4GeneExpressionDataset('/dtu-compute/datasets/iso_02456/hdf5-row-sorted/')
archs4_dataset_val = IsoDatasets.Archs4GeneExpressionDataset('/dtu-compute/datasets/iso_02456/hdf5-row-sorted/',
                                                             validation_set=True)
print("archs4 training set size:", len(archs4_dataset_train))
print("archs4 validation set size:", len(archs4_dataset_val))

archs4_train_dataloader = DataLoader(archs4_dataset_train, batch_size=500, shuffle=True)
archs4_val_dataloader = DataLoader(archs4_dataset_val, batch_size=1, shuffle=True)


print('Training means for each attribute')
first = True
for X in tqdm(archs4_train_dataloader):
    
    if first:
        a = torch.sum(X, dim=0)
        first = False
        continue
    
    a = torch.add((torch.sum(X, dim=0)), a)

meanTensor = torch.div(a,len(archs4_dataset_train))
print(meanTensor)
print(meanTensor.size())


print('Calculating distance to validation set')
sum_squared_error = 0
for X in tqdm(archs4_val_dataloader):
    predictions = meanTensor.expand_as(X)
    
    squared_errors =(predictions - X)**2
    
    sum_squared_error += squared_errors.sum()

mse_loss_val = sum_squared_error / (len(archs4_dataset_val) * 156958)



#Print the MSE loss
print(f'MSE Val Loss: {mse_loss_val.item()}')