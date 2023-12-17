#!/usr/bin/env python3

import IsoDatasets as IsoDatasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import argparse

'pca_output.tsv'
parser = argparse.ArgumentParser(description='Print latent feature representation of the Gtex data, produced by a given PCA model')
parser.add_argument('-pcap', type=str, help='Path to trained pca model')
parser.add_argument('-o', type=str, help='Embedded features output file path')
args = parser.parse_args()

IPCA = args.pcap
OUTPUT_FILE = args.o

# Load PCA model
with open(IPCA, 'rb') as file:
    ipca = pickle.load(file)

# Load gtex data
gtex_test = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5-row-sorted/", include=".")
print("gtex test set size:", len(gtex_test))
gtx_test_dataloader = DataLoader(gtex_test, batch_size=1, shuffle=True)

# Write embeddings
with open(OUTPUT_FILE, 'w') as file:
    for x, y, label in tqdm(gtx_test_dataloader):
        x = ipca.transform(x)
        label_decoded = label[0].decode('utf-8')
        file.write(f"{x[0][0]}\t{x[0][1]}\t{label_decoded}\n")

## EXAMPLE RUN
#latent_space_representation_PCA.py -pcap /zhome/99/d/155947/DeeplearningProject/deepIsoform/models/ipca_model_n16.pkl -o /zhome/99/d/155947/DeeplearningProject/deepIsoform/data/bhole_storage/training_meta_data/latent_space_points/latent_space_representation_PCA.tsv