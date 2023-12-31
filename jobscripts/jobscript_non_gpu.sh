#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J UMAP_archs4_trained
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 100GB of memory per core/slot -- 
#BSUB -R "rusage[mem=100GB]"
### -- specify that we want the job to get killed if it exceeds 100 GB per core/slot -- 
#BSUB -M 100GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 02:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s204540
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o /zhome/99/d/155947/DeeplearningProject/deepIsoform/logs/%J.out 
#BSUB -e /zhome/99/d/155947/DeeplearningProject/deepIsoform/logs/%J.err 

# here follow the commands you want to execute with input.in as the input file
source activate VAE-env2

#{ time /zhome/99/d/155947/DeeplearningProject/deepIsoform/tmp/small_VAE_test.py ; } 2> time_e50_d1000_100_l2000_1000_50.txt

#/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/latent_space_representation_UMAP.py
/zhome/99/d/155947/DeeplearningProject/deepIsoform/scripts/calc_sd.py