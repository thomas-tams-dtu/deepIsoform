#!/usr/bin/env python3

print('hi')
from torch.utils.data import DataLoader
from torch import var
from tqdm import tqdm
import IsoDatasets as IsoDatasets

BATCH_SIZE = 50

print('go')
#archs4_dataset_val = IsoDatasets.Archs4GeneExpressionDataset('/dtu-compute/datasets/iso_02456/hdf5-row-sorted/')
gtex_train = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5-row-sorted/", exclude=['brain', 'Artery'])
gtex_val = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5-row-sorted/", include='brain')
gtex_test = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5-row-sorted/", include='Artery')

print("gtex training set size:", len(gtex_train))
print("gtex validation set size:", len(gtex_val))
print("gtex test set size:", len(gtex_test))

gtx_train_dataloader = DataLoader(gtex_train, batch_size=BATCH_SIZE, shuffle=True)
gtx_val_dataloader = DataLoader(gtex_val, batch_size=BATCH_SIZE, shuffle=True)
gtx_test_dataloader = DataLoader(gtex_test, batch_size=10, shuffle=True)

print('loaded')

#counter = 0
#for x, y in gtx_train_dataloader:
#    break
