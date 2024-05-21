#!/bin/bash
#SBATCH --nodes 1
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time 30:00:00
#SBATCH --mem=100G
#SBATCH --output ./output/slurm-%j.out

module load CUDA/11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0

#echo "***Code version***"
#version=$(git log -1 --oneline)
#echo "Git Commit Version: $version"
#echo "JobId:$SLURM_JOB_ID"
#echo "******************"
#echo

python3 /data/gpfs/projects/punim1836/Training/Training_cropped_volume.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

