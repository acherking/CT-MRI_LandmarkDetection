#!/bin/zsh

base_path=/Users/achs/PhD/code/CT-MRI_LandmarkDetection

# set rsa pass in local ~/.ssh/id_rsa and remote ~/.ssh/authorized_keys(id_rsa.pub)
rsync -zarvm -e ssh --include="*/" --include="*.npy" --exclude="*" wezw@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim1836/CT-MRI_LandmarkDetection/models/ ${base_path}/models/
