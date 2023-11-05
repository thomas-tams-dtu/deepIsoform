#!/usr/bin/env python3

import sys
sys.path.insert(1, '../scripts')
from chunky_loader import ChunkyDataset
from torch.utils.data import DataLoader
import time

NROWS_DATASET_TEST = 0
NROWS_DATASET_TRAIN = 5000 - NROWS_DATASET_TEST
CHUNK_SIZE = 500
BATCH_SIZE = 500

start_all = time.time()
#gz_path ="/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/head5000_archs4.tsv.gz"
gz_path = "/zhome/99/d/155947/DeeplearningProject/deepIsoform/data/raw_data/archs4_gene_expression_norm_transposed.tsv.gz"
GzChunks_train = ChunkyDataset(file_path=gz_path, nrows=NROWS_DATASET_TRAIN, lines_per_chunk=CHUNK_SIZE, skip_lines=NROWS_DATASET_TEST, got_header=True)
#GzChunks_test = ChunkyDataset(file_path=gz_path, nrows=NROWS_DATASET_TEST, lines_per_chunk=CHUNK_SIZE, skip_lines=0, got_header=True)


loader_train = DataLoader(GzChunks_train, batch_size=BATCH_SIZE)
#loader_test = DataLoader(GzChunks_test, batch_size=BATCH_SIZE)


for x, y in loader_train:
  #print('train_features', x)
  #print('sample id train', y)
  pass

end_all = time.time()
print('time entire 5000 (500,500) run', end_all-start_all)

#for x, y in loader_test:
#  #print('train_features', x)
#  print('sample id test', y)