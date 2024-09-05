#!/bin/zsh

local_base_dir=/Users/achs/PhD/code/CT-MRI_LandmarkDetection/src/data/GenerateData_Matlab
remote_base_dir=wezw@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim1836/CT-MRI_LandmarkDetection/src/data/GenerateData_Matlab

# set rsa pass in local ~/.ssh/id_rsa and remote ~/.ssh/authorized_keys(id_rsa.pub)
# scp -p ../Training_withGPU.py wezw@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim1836/Training
rsync -tv -e ssh ${local_base_dir}/*.m ${remote_base_dir}

# update on spartan before each sbatch training
# rsync -tv -e ssh ${local_base_dir}/slurm/*.slurm ${remote_base_dir}/slurm
