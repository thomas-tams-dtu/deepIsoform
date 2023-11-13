#!/usr/bin/env python3

import pandas as pd
import h5py
from sklearn.model_selection import train_test_split



isoform_file_path = "/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/raw_data/gtex_isoform_expression_norm_transposed.tsv.gz"
gene_file_path = "/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/raw_data/gtex_gene_expression_norm_transposed.tsv.gz"

#isoform_file_path = "/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/head_gtex_isoform_expression_norm_transposed.tsv"
#gene_file_path =    "/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/head_gtex_gene_expression_norm_transposed.tsv"


hdf5_file_path_train = "/dtu/blackhole/0b/155947/train_gtex_geneX_isoformY2.hdf5"
hdf5_file_path_val = "/dtu/blackhole/0b/155947/val_gtex_geneX_isoformY2.hdf5"    
hdf5_file_path_test = "/dtu/blackhole/0b/155947/test_gtex_geneX_isoformY2.hdf5"
hdf5_file_path_all = "/dtu/blackhole/0b/155947/all_gtex_geneX_isoformY2.hdf5"  

print('variables created')

isoform_df = pd.read_csv(isoform_file_path, sep='\t')
print('isoforms loaded')
gene_df = pd.read_csv(gene_file_path,  sep='\t')
print('gene loaded')


join_inner = pd.merge(isoform_df, gene_df, on='sample_id')

## Save the DataFrame to HDF5 format
with h5py.File(hdf5_file_path_all, 'w') as f:
    f.create_dataset('gene_expression', data=join_inner.iloc[:, 1:156959])
    f.create_dataset('isoform_expression', data=join_inner.iloc[:, 156959:])


"""

print('all dataset loaded')

X_train, X_test, y_train, y_test = train_test_split(gene_expression_levels, isoform_expression_levels, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

print('splits created')

## Save the DataFrame to HDF5 format
with h5py.File(hdf5_file_path_train, 'w') as f:
    f.create_dataset('isoform_expression', data=y_train)
    f.create_dataset('gene_expression', data=X_train)

print('train dataset create')

with h5py.File(hdf5_file_path_val, 'w') as f:
    f.create_dataset('isoform_expression', data=y_val)
    f.create_dataset('gene_expression', data=X_val)

print('val dataset create')

with h5py.File(hdf5_file_path_test, 'w') as f:
    f.create_dataset('isoform_expression', data=y_test)
    f.create_dataset('gene_expression', data=X_test)

print('test dataset create')

"""