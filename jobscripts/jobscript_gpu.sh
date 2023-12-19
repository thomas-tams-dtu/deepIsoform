#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J DNN_xl
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 23:50
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


#### DENSE NETWORK
#net_size=small
#net_size=medium
#net_size=large
#net_size=XL
#
#weight_decays=(1e-8 5e-8 1e-7 5e-7 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3)
#learning_rates=(5e-2 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6 5e-7 1e-7 5e-8 1e-8)
#
### Loop over the values
#for lr in "${learning_rates[@]}"; do
#for wd in "${weight_decays[@]}"; do
#
#/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/dense_train.py -ns ${net_size} -wd ${wd} -bs 500 -lr ${lr} -p 10 -e 100
/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/train_standaloneDNN.py -ns XL -wd 5e-7 -bs 500 -lr 1e-3 -p 10 -e 100 --sm
#done
#done
#
#echo end of run


#### TRAIN PCA DENSE NETWORK
#net_size=small
#lr=0.005

#net_size=medium
#lr=0.001

#net_size=large
#lr=0.005

#net_size=XL
#lr=0.001

##weight_decays=(1e-8 5e-8 1e-7 5e-7 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3)
#weight_decays=(5e-3 1e-2 5e-2 1e-1 5e-1)
#latent_features=(16 32 64 128 256 512 1024 2048)
##latent_features=(512 1024 2048)
#
## Loop over the values
#for lf in "${latent_features[@]}"; do
#for wd in "${weight_decays[@]}"; do
#
#/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/PCA_dense_train.py -ns ${net_size} -lf ${lf} -wd ${wd} -bs 500 -lr ${lr} -p 10 -e 100
/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/train_PCADNN.py -ns XL -lf 1024 -wd 5e-8 -bs 500 -lr 1e-4 -p 10 -e 100 --sm
#done
#done
#
#echo end of run

#### TRAIN VAE
#beta_values=(0.01 0.05 0.1 0.5 1 5 10 50)
##latent_features=(1024 512 256 128 64 32 16)
#latent_features=(256 512)
#
## Loop over the values
#for lf in "${latent_features[@]}"; do
#for beta in "${beta_values[@]}"; do
#
#/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/VAE_train.py -lf ${lf} -b ${beta} -bs 500 -lr 1e-4 -hl 128 -e 100 -p 100 --sm
#/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/VAE_train.py -lf 2 -b 1 -bs 200 -lr 1e-4 -hl 128 -e 10 -p 10
#done
#done
#
#echo end of run



#### TRAIN ENCODER DENSE NETWORK
##net_size=small
##lr=0.005
#
##net_size=medium
##lr=0.001
#
##net_size=large
##lr=0.005
#
#net_size=XL
#lr=0.001
#
##latent_features=(512)
##beta_values=(1)
#latent_features=(1024 256 128 64 32 16)
#beta_values=(0.01 0.01 0.05 0.01 0.5 0.1)
##weight_decays=(5e-6 1e-5 5e-5 1e-4 5e-4 1e-3)
#weight_decays=(1e-8 5e-8 1e-7 5e-7 1e-6 5e-3 1e-2 5e-2 1e-1 5e-1)
#
#for ((i=0; i<${#latent_features[@]}; i++)); do
#    lf=${latent_features[i]}
#    beta=${beta_values[i]}
#
#    for wd in "${weight_decays[@]}"; do
#        /zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/encoder_dense_train.py -ns ${net_size} -lf ${lf} -wd ${wd} -b ${beta} -bs 500 -lr ${lr} -p 10 -e 100
#    done
#done
#
#echo end of run


##### TRAIN CUSTOM ENCODER DENSE
#net_size=small
#lr=0.005

#net_size=medium
#lr=0.001

#net_size=large
#lr=0.005

#net_size=XL
#lr=0.001

#lf=2
##beta_values=(1 0.1 0.001 0.0001 1e-5 1e-6 0)
#beta_values=(1)
#weight_decays=(1e-8 5e-8 1e-7 5e-7 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 5e-3)
#
#for wd in "${weight_decays[@]}"; do
#for beta in "${beta_values[@]}"; do
#
##/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/train_encDNN.py -ns ${net_size} -lf ${lf} -wd ${wd} -b ${beta} -bs 500 -lr ${lr} -p 9 -e 100
#/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/train_encDNN.py -ns ${net_size} -lf ${lf} -wd ${wd} -b ${beta} -bs 500 -lr ${lr} -p 9 -e 100 --sm
#
#done
#done

echo end of run


##### TRAIN VAE AND LATENT REPRESENTATION
#beta_values=(1 0.1 0.001 0.0001 1e-5 1e-6 0)
#
#for beta in "${beta_values[@]}"; do
#
#/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/train_VAE.py -lf 256 -b ${beta} -bs 200 -lr 1e-4 -hl 128 -e 30 -p 10 --sm
#
#done
#
###/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/encoder_dense_latent_rep.py -ns small -lf 16 -wd 5e-7 -b 0 -bs 500 -lr 0.005 -p 10 -e 100 --sm

