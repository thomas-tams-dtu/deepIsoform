#!/usr/bin/env python3

import umap
import IsoDatasets as IsoDatasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

PROJECT_DIR =f'/zhome/99/d/155947/DeeplearningProject/deepIsoform'
METADATA_SAVE_PATH = f'{PROJECT_DIR}/data/bhole_storage/training_meta_data/UMAP_embed_archs4_trained_2.tsv'
METADATA_SAVE_PATH_ARCHS4 = f'{PROJECT_DIR}/data/bhole_storage/training_meta_data/UMAP_archs4_embed.tsv'

gtex_all = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5-row-sorted/", include=".")
archs4_dataset_train = IsoDatasets.Archs4GeneExpressionDataset('/dtu-compute/datasets/iso_02456/hdf5-row-sorted/')
#gtex_all = IsoDatasets.GtexDataset("/dtu-compute/datasets/iso_02456/hdf5-row-sorted/", include=".", load_in_mem=True)
print("gtex test set size:", len(gtex_all))
print("archs4 test set size", len(archs4_dataset_train))

reducer = umap.UMAP()

# Load in arch4s
archs4_dataloader = DataLoader(archs4_dataset_train, batch_size=151095, shuffle=True)

embbing_data_file = open(METADATA_SAVE_PATH, 'w')
for x in archs4_dataloader:
    archs4_embed = reducer.fit_transform(x)

    emb1 = archs4_embed[:, 0].tolist()
    emb2 = archs4_embed[:, 1].tolist()

    combined_data = list(zip(emb1, emb2))
    for z1, z2, lab in combined_data:
        embbing_data_file.write(f'{z1},{z2}\n')   

# Run the embedding of gtex
gtex_dataloader = DataLoader(gtex_all, batch_size=17356, shuffle=True)
embbing_data_file = open(METADATA_SAVE_PATH, 'w')
for x, _, tissues in gtex_dataloader:
    gtex_embed = reducer.transform(x)
    tissue_list = [tissue.decode('utf-8') for tissue in tissues]

    emb1 = gtex_embed[:, 0].tolist()
    emb2 = gtex_embed[:, 1].tolist()

    combined_data = list(zip(emb1, emb2, tissue_list))
    for z1, z2, lab in combined_data:
        embbing_data_file.write(f'{z1},{z2},{lab}\n')
    break

print('done')
