#!/bin/bash
#SBATCH --nodes 1
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time 50:00:00
#SBATCH --mem=128G
#SBATCH --output ./output/slurm-%j.out

module load CUDA/11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0

#python try_corrode.py u_net_dsnt 11 09Jun20242158
#python try_corrode.py scn_dsnt 0 07Jun20241709

export TF_ENABLE_ONEDNN_OPTS=0
python3 /data/gpfs/projects/punim1836/CT-MRI_LandmarkDetection/src/models/try_corrode.py scn_dsnt 8 17Jun2024-13:44:55-trainID-8/epoch_6

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

