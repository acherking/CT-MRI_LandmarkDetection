#!/bin/bash
#SBATCH --nodes 1
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time 76:00:00
#SBATCH --mem=300G
#SBATCH --output ./output/slurm-%j.out

module purge
module load cuda/11.1.1
module load tensorflow/2.6.0-python-3.8.6

if output=$(git status --porcelain) && [ -n "$output" ]; then
  echo "Git: some files need to be committed :)"
  echo "$output"
  exit 1
fi
echo "***Code version***"
version=$(git log -1 --oneline)
echo "Git Commit Version: $version"
echo "JobId:$SLURM_JOB_ID"
echo "******************"
echo

python3 /data/gpfs/projects/punim1836/Training/Training_withGPU.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

