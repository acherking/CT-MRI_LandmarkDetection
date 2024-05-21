#!/bin/bash
#SBATCH --nodes 1
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time 24:00:00
#SBATCH --mem=100G
#SBATCH --output ./output/slurm-%j.out

module purge
module load cuda/11.1.1
module load tensorflow/2.6.0-python-3.8.6

echo "***Code version***"
version=$(git log -1 --oneline)
echo "Git Commit Version: $version"
echo "JobId:$SLURM_JOB_ID"
echo "******************"
echo

python3 /data/gpfs/projects/punim1836/Training/Training_focus_volume.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

