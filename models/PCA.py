import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

df = pd.read_csv("data/head_of_archs4_gene_expression_norm_transposed.tsv", header=0)
print(df.head())
n_components = 2

pca = PCA(n_components=n_components)
pca.fit(X)


