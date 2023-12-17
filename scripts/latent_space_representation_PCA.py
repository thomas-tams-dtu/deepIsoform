#!/usr/bin/env python3

import torch
import IsoDatasets as IsoDatasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle
import re


IPCA = f'/zhome/99/d/155947/DeeplearningProject/deepIsoform/models/ipca_model_n16.pkl'
with open(IPCA, 'rb') as file:
    ipca = pickle.load(file)

gtex_test = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5-row-sorted/", include=".")
print("gtex test set size:", len(gtex_test))

gtx_test_dataloader = DataLoader(gtex_test, batch_size=1, shuffle=True)

gene_expr, isoform_expr, _ = next(iter(gtx_test_dataloader))

with open('pca_output.tsv', 'w') as file:

    pattern = re.compile(r'^([^\s]+)')

    # Go through each batch in the training dataset using the loader
    for x, y, label in tqdm(gtx_test_dataloader):
        x = ipca.transform(x)
        label_decoded = label[0].decode('utf-8')
        match = pattern.search(label_decoded)
        #print(f"{x[0][0]}\t{x[0][1]}\t{match.group(1)}\n")
        file.write(f"{x[0][0]},{x[0][1]},{label_decoded}\n")