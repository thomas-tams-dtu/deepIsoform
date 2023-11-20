#!/usr/bin/env python3

import pandas as pd
import h5py
import numpy as np
from torch.utils.data import DataLoader
import sys
sys.path.insert(1, '/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts')
from hdf5_load import DatasetHDF5
from sklearn.model_selection import train_test_split


tsv_file_path = "/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/raw_data/archs4_gene_expression_norm_transposed.tsv.gz"
#tsv_file_path = "/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/head_archs4_gene_expression_norm_transposed.tsv"
hdf5_file_path_train = "/dtu/blackhole/0b/155947/train_archs4_gene_expression_norm_transposed.hdf5"
hdf5_file_path_val = "/dtu/blackhole/0b/155947/val_archs4_gene_expression_norm_transposed.hdf5"
hdf5_file_path_test = "/dtu/blackhole/0b/155947/test_archs4_gene_expression_norm_transposed.hdf5"

## Read the TSV file into a DataFrame
df = pd.read_csv(tsv_file_path, sep='\t')
print(f'{tsv_file_path} loaded')

sample_ids = df.iloc[:, 0].to_numpy()
expression_levels = df.iloc[:, 1:].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(expression_levels, sample_ids, test_size=0.1, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.11, random_state=1) # 0.111 x 0.9 = 0.1


## Save the DataFrame to HDF5 format
with h5py.File(hdf5_file_path_train, 'w') as f:
    f.create_dataset('sample_ids', data=y_train)
    f.create_dataset('expression_levels', data=X_train)

with h5py.File(hdf5_file_path_val, 'w') as f:
    f.create_dataset('sample_ids', data=y_val)
    f.create_dataset('expression_levels', data=X_val)

with h5py.File(hdf5_file_path_test, 'w') as f:
    f.create_dataset('sample_ids', data=y_test)
    f.create_dataset('expression_levels', data=X_test)



"""

dataset = DatasetHDF5(hdf5_file_path)
loader_train = DataLoader(dataset, batch_size=2, shuffle=True)

for x,y in loader_train:
    print(y)
    print(x)
    break
    
"""

