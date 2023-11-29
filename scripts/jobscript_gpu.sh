#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J encD_test
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 00:10
# request 5GB of system-memory
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -R "span[hosts=1]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o /zhome/99/d/155947/DeeplearningProject/deepIsoform/logs
#BSUB -e /zhome/99/d/155947/DeeplearningProject/deepIsoform/logs
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/11.6

/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

# Source/ activate VAE-env2 conda environment
source activate VAE-env2

# Run command
/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/PCA_dense_train.py -ns XXL -lf 32 -wd 1e-6 -bs 500 -lr 1e-4 -p 6 -e 100


#### TRAIN PCA MODEL
#/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/PCA_train.py -nc 4044


#### TRAIN PCA DENSE NETWORK
#net_size=small
#net_size=medium
#net_size=large
#net_size=XL
#net_size=XXL
#
#weight_decays=(1e-8 5e-8 1e-7 5e-7 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3)
#latent_features=(16 32 64 128 256 512 1024 2048)
#
## Loop over the values
#for lf in "${latent_features[@]}"; do
#for wd in "${weight_decays[@]}"; do
#
#/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/PCA_dense_train.py -ns ${net_size} -lf ${lf} -wd ${wd} -bs 500 -lr 1e-4 -p 6 -e 100
##/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/PCA_dense_train.py -ns XXL -lf 32 -wd 1e-6 -bs 500 -lr 1e-4 -p 6 -e 100
#done
#done
#
#echo end of run

### TRAIN VAE
#beta_values=(0.01 0.05 0.1 0.5 1 5 10 50)
#latent_features=(1024 512 256 128 64 32 16)
#
## Loop over the values
#for lf in "${latent_features[@]}"; do
#for beta in "${beta_values[@]}"; do
#
#/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/VAE_train.py -lf ${lf} -b ${beta} -bs 500 -lr 1e-4 -hl 128 -e 50 -p 6 --sm
##/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/VAE_train.py -lf 16 -b 5 -bs 250 -lr 1e-4 -hl 128 -e 100 -p 6
#done
#done
#
#echo end of run


#### TRAIN ENCODER DENSE NETWORK
#net_size=small
#net_size=medium
#net_size=large
#net_size=XL

#weight_decays=(0.00000001 0.00000005 0.0000001 0.0000005 0.000001 0.000005 0.00001 0.00005 0.0001 0.0005 0.001)
#latent_features=(16 32 64 128 256 512 1024 2048 4096 8192 16384)
#
## Loop over the values
#for lf in "${latent_features[@]}"; do
#for b in "$beta_values[@]"; do
#
#wd = 
#
#/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/encoder_dense_train.py -ns ${net_size} -lf ${lf} -wd ${wd} -b ${b} -bs 500 -lr 1e-4 -p 6 -e 100
#/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/encoder_dense_train.py -ns small -lf 1024 -wd 5e-05 -b 0.05 -bs 500 -lr 1e-4 -p 6 -e 100
#done
#done
#
#echo end of run