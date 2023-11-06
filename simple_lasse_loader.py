path = '/dtu-compute/datasets/iso_02456/archs4_gene_expression_norm_transposed.tsv.gz'
import pandas as pd
import torch
def read_chunks(path, prints=False)
  allbatches = pd.read_csv(path, engine='c',sep='\t', chunksize=500)
  i = 0
  for batch in allbatches:
      batch=batch.iloc[:,1:]
      batch=torch.from_numpy(batch.values)

      if prints:
        i += 500
        print('at line:',i)
      yield batch
    for chunk in read_chunks(path, prints=True):
        pass


