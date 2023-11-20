#!/bin/bash

# List of values
learning_rates=(0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000 10000)
latent_features=(16 32 64 128 256)

# Loop over the values
for lf in "${latent_features[@]}"; do

for lr in "${learning_rates[@]}"; do
    echo "lr $lr   lf $lf"
done

done
