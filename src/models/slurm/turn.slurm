#!/bin/bash
#SBATCH --nodes 1
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time 150:00:00
#SBATCH --mem=128G
#SBATCH --output ./output/slurm-%j.out

module load CUDA/11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0

export TF_ENABLE_ONEDNN_OPTS=0
#python3 /data/gpfs/projects/punim1836/CT-MRI_LandmarkDetection/src/models/Tuner.py
python3 /data/gpfs/projects/punim1836/CT-MRI_LandmarkDetection/src/models/Tuner_grid.py 4 0

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

