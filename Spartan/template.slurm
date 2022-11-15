#!/bin/bash
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time 10:35:00
#SBATCH --mem=30G
## Use an account that has GPGPU access

module load cuda/11.1.1
module load tensorflow/2.7.1

python3 ../code/CT-MRI_LandmarkDetection/JupterNotebook/Spartan_GPU_training.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

