#!/bin/zsh

base_path=/Users/achs/PhD/code/CT-MRI_LandmarkDetection/Spartan
fun_path=/Users/achs/PhD/code/CT-MRI_LandmarkDetection/Functions

# set rsa pass in local ~/.ssh/id_rsa and remote ~/.ssh/authorized_keys(id_rsa.pub)
# scp -p ../Training_withGPU.py wezw@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim1836/Training
rsync -tv -e ssh ${base_path}/*.py ${base_path}/*.slurm wezw@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim1836/Training

rsync -tv -e ssh ${fun_path}/*.py wezw@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim1836/Training/Functions
