#!/usr/bin/env python3

import IsoDatasets
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Example of making a training set that excludes samples from the brain and a test set with only samples from the brain
# If you have enough memory, you can load the dataset to memory using the argument load_in_mem=True
gtex_train = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5-row-sorted/", exclude=['brain', 'Artery'])
gtex_val = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5-row-sorted/", include='brain')
gtex_test = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5-row-sorted/", include='Artery')

print("gtex training set size:", len(gtex_train))
print("gtex test set size:", len(gtex_test))

print(len(gtex_test)/(len(gtex_test)+len(gtex_train)) * 100)

gtx_train_dataloader = DataLoader(gtex_train, batch_size=500, shuffle=True)
gtex_val_dataloader = DataLoader(gtex_val, batch_size=1, shuffle=True)
gtx_test_dataloader = DataLoader(gtex_test, batch_size=1, shuffle=True)


print('Training means for each attribute')
first = True
for X, y, _ in tqdm(gtx_train_dataloader):
    
    if first:
        a = torch.sum(y, dim=0)
        first = False
        continue
    
    a = torch.add((torch.sum(y, dim=0)), a)

meanTensor = torch.div(a,len(gtex_train))
print(meanTensor)
print(meanTensor.size())


print('Calculating distance to validation set')
sum_squared_error = 0
totalLength = 0
for X, y, _ in tqdm(gtex_val_dataloader):
    predictions = meanTensor.expand_as(y)
    
    squared_errors =(predictions - y)**2
    
    totalLength += len(X)
    
    sum_squared_error += squared_errors.sum()

mse_loss_val = sum_squared_error / (len(gtex_val) * 156958)

print('Calculating distance to test set')
sum_squared_error = 0
totalLength = 0
for X, y, _ in tqdm(gtx_test_dataloader):
    predictions = meanTensor.expand_as(y)
    
    squared_errors =(predictions - y)**2
    
    totalLength += len(X)
    
    sum_squared_error += squared_errors.sum()

mse_loss_test = sum_squared_error / (len(gtex_test) * 156958)


#Print the MSE loss
#print(f"Largest value:")
print("Total length:" ,totalLength)
print(f'MSE Val Loss: {mse_loss_val.item()}')
print(f'MSE Test Loss: {mse_loss_test.item()}')