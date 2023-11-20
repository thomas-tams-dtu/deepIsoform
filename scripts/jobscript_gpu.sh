#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J pca_train_512
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 03:50
# request 5GB of system-memory
#BSUB -R "rusage[mem=7GB]"
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
#/zhome/99/d/155947/DeeplearningProject/deepIsoform/tmp/VAE_train.py

#{ time /zhome/99/d/155947/DeeplearningProject/deepIsoform/tmp/small_VAE_test.py ; } 2> time_e50_dfull_20000_l2000_1000_50.txt
#/zhome/99/d/155947/DeeplearningProject/deepIsoform/tmp/hdf5_load.py

/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/PCA_train.py -nc 512

#net_size=small
#
#weight_decays=(0.0000001 0.0000005 0.000001 0.000005 0.00001 0.00005 0.0001 0.0005 0.001 0.005)
#latent_features=(16 32 64 128 256 512 1024)
#
## Loop over the values
#for lf in "${latent_features[@]}"; do
#
#for wd in "${weight_decays[@]}"; do
#
#/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/dense_train.py -ns ${net_size} -lf ${lf} -wd ${wd}
#
#done
#
#done

