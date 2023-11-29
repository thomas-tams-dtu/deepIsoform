#!/usr/bin/env python3

print('hi')
from torch.utils.data import DataLoader
from tqdm import tqdm
import IsoDatasets as IsoDatasets

print('go')
archs4_dataset_val = IsoDatasets.Archs4GeneExpressionDataset('/dtu-compute/datasets/iso_02456/hdf5-row-sorted/')
val_loader = DataLoader(archs4_dataset_val, batch_size = 1000)

print('loaded')

counter = 0
for X in val_loader:
    counter+=1
    print(counter)
