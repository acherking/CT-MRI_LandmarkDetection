#!/bin/bash
#SBATCH --nodes 1
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time 10:35:00
#SBATCH --mem=30G
#SBATCH -A wezw
## Use an account that has gpu-a100 access

module purge
module load cuda/11.1.1
module load tensorflow/2.6.0-python-3.8.6

python3 ../code/CT-MRI_LandmarkDetection/JupterNotebook/Spartan_GPU_training.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

