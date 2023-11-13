#!/usr/bin/env python3

from torch.utils.data import DataLoader
import numpy as np
import sys
sys.path.insert(1, '/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts')
from hdf5_load import DatasetHDF5
import pandas as pd


hdf5_path = "/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/bhole_storage/archs4_gene_expression_norm_transposed.hdf5"
hdf5_dataset_train = DatasetHDF5(hdf5_path)

nrows = 18965
distance_matrix = np.empty([nrows, nrows], dtype=float)

distance_dict ={}

for i in range(nrows):
    for j in range(nrows):
        row_sample, _ = hdf5_dataset_train[i]
        compare_sample, _ = hdf5_dataset_train[j]

        distance = np.linalg.norm(row_sample-compare_sample)
        idx_dict = {j : round(distance,2)}
        distance_dict.update(idx_dict)
    
    pd.DataFrame({'euclidean_distance':distance_dict.values(), 'hdf5_idx':distance_dict.keys()}).to_csv('distance.csv')
    break

value_sorted_dict = dict(sorted(distance_dict.items(), key=lambda item: item[1], reverse=True))

pd.DataFrame({'euclidean_distance':value_sorted_dict.values(), 'hdf5_idx':value_sorted_dict.keys()}).to_csv('distance_sorted.csv')