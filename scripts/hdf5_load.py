#!/usr/bin/env python3

import h5py
from torch.utils.data import Dataset, DataLoader
import time
from torch import tensor, double, float64
import numpy

class DatasetHDF5(Dataset):
    def __init__(self, hdf5_path, given_length=None):
        self.hdf5_path = hdf5_path
        self.hdf5_file = h5py.File(self.hdf5_path, 'r')
        self.expression_level_key = list(self.hdf5_file.keys())[0]
        self.sample_id_key = list(self.hdf5_file.keys())[1]
        if given_length is not None:
            self.nrows = given_length            
        else:
            self.nrows = self.hdf5_file[self.expression_level_key].len()

    def __len__(self):
        return self.nrows
    
    def __getitem__(self, idx):
        #return self.load_chunk(chunk_size=self.batch_size)
        return self.hdf5_file[self.expression_level_key][idx], self.hdf5_file[self.sample_id_key][idx]
    

class DatasetHDF5_gtex(Dataset):
    def __init__(self, hdf5_path, given_length=None):
        self.hdf5_path = hdf5_path
        self.hdf5_file = h5py.File(self.hdf5_path, 'r')
        self.gene_expression = list(self.hdf5_file.keys())[0]
        self.isoform_expression = list(self.hdf5_file.keys())[1]

        if given_length is not None:
            self.nrows = given_length            
        else:
            self.nrows = self.hdf5_file[self.isoform_expression].len()

    def __len__(self):
        return self.nrows
    
    def __getitem__(self, idx):
        #return self.load_chunk(chunk_size=self.batch_size)
        return self.hdf5_file[self.gene_expression][idx], self.hdf5_file[self.isoform_expression][idx]


BATCH_SIZE = 2

#hdf5_path = "/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/raw_data/hdf5/archs4_gene_expression_norm_transposed.hdf5"
#hdf5_path = "/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/bhole_storage/all_gtex_geneX_isoformY2.hdf5"
#h5py_dataset = DatasetHDF5_gtex(hdf5_path)
#
#data_loader = DataLoader(h5py_dataset, 
#                         batch_size=BATCH_SIZE)
#
#
#start = time.time()
#for x, y in data_loader:
#    print(x)
#    print(y)
#    print(x[0].size())
#    print(x[0].size())
#    break
#
#print('Load all', time.time()-start)