#!/bin/bash
#SBATCH --nodes 1
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time 50:00:00
#SBATCH --mem=100G
#SBATCH --output ./output/slurm-%j.out

module load CUDA/11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0

#python3 /data/gpfs/projects/punim1836/Training/Training_model_dsnt.py
python3 /data/gpfs/projects/punim1836/CT-MRI_LandmarkDetection/src/models/Training_divided_volume.py 2 two_landmarks cov_only_dsnt

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

