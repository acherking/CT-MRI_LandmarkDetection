#!/bin/bash
#SBATCH --nodes 1
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time 15:00:00
#SBATCH --mem=100G
#SBATCH --output ./output/slurm-%j.out

module load OpenMPI/4.1.4
module load TensorFlow/2.11.0-CUDA-11.7.0

python3 /data/gpfs/projects/punim1836/Training/tmp.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

