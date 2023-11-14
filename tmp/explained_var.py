#!/usr/bin/env python3

import pickle

LATENT_FEATURES = 32

IPCA = f'/zhome/99/d/155947/DeeplearningProject/deepIsoform/models/ipca_model_n{LATENT_FEATURES}.pkl'

with open(IPCA, 'rb') as file:
    ipca = pickle.load(file)

print(sum(ipca.explained_variance_ratio_))