#!/usr/bin/env python3

import sys
sys.path.insert(1, '../scripts')
from chunky_loader import ChunkyDataset
from torch.utils.data import DataLoader
import time


#gz_path ="/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/head5000_archs4.tsv.gz"
#gz_path = "/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/raw_data/archs4_gene_expression_norm_transposed.tsv.gz"



#GzChunks_test = ChunkyDataset(file_path=gz_path, nrows=NROWS_DATASET_TEST, lines_per_chunk=CHUNK_SIZE, skip_lines=0, got_header=True)
#loader_test = DataLoader(GzChunks_test, batch_size=BATCH_SIZE)

#GzChunks_train = ChunkyDataset(file_path=gz_path, nrows=NROWS_DATASET_TRAIN, lines_per_chunk=CHUNK_SIZE, skip_lines=NROWS_DATASET_TEST, got_header=True)
#loader_train = DataLoader(GzChunks_train, batch_size=BATCH_SIZE)
#
#
#
#for x, y in loader_train:
#  #print('train_features', x)
#  #print('sample id train', y)
#  pass
#
#end_all = time.time()
#print('time entire 5000 (500,500) run', end_all-start_all)

#for x, y in loader_test:
#  #print('train_features', x)
#  print('sample id test', y)


import pandas as pd
from torch import from_numpy, tensor
from multiprocessing import Pool
from math import ceil

class ChunkReader():
  def __init__(self, data_path, chunk_size, num_workers, nrows_file):
    self.data_path = data_path
    self.chunk_size = chunk_size
    self.num_workers = num_workers
    self.nrows_per_worker = ceil(chunk_size / num_workers)
    self.last_idx_previous_chunk = 0
    self.nrows_file = nrows_file


  def load_chunk(self, chunk_range):
    start, end = chunk_range
    return pd.read_csv(self.data_path, sep='\t', skiprows=range(1, start), nrows=end-start)

  def next_chunk(self):
    print('last loaded idx', self.last_idx_previous_chunk)
    if self.last_idx_previous_chunk >= self.nrows_file:
      print('EOF', self.last_idx_previous_chunk)
      return None
    
    chunk_ranges = [(start + self.last_idx_previous_chunk, min(start + self.last_idx_previous_chunk + self.nrows_per_worker, self.nrows_file)) for start in range(1, self.chunk_size, self.nrows_per_worker)]
    self.last_idx_previous_chunk = chunk_ranges[-1][1]
    print(chunk_ranges)

    with Pool(processes=self.num_workers) as mpool:
      data_frame = mpool.map(self.load_chunk, chunk_ranges)
    
    chunk_data = pd.concat(data_frame, axis=0)

    return chunk_data
    #return chunk.iloc[:,0].values, tensor(chunk.iloc[:,1:].values)
  
  def __getitem__(self, index):
    return self.next_chunk()


BATCH_SIZE = 1000
NROWS_DATASET = 5000

start_all = time.time()
#gz_path = "/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/head500_archs4_gene_expression_norm_transposed.tsv.gz"
gz_path = "/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/head5000_archs4.tsv.gz"

start_panda = time.time()
chunky = ChunkReader(data_path = gz_path, nrows_file=NROWS_DATASET, chunk_size=BATCH_SIZE, num_workers=10)

for chunk in chunky:
  if chunk is None:
    break


print('pandas time 5000 (500,500)', time.time()-start_panda)
