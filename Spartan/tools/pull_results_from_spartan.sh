#!/bin/zsh

base_path=/Users/achs/PhD/code/CT-MRI_LandmarkDetection/Spartan

# set rsa pass in local ~/.ssh/id_rsa and remote ~/.ssh/authorized_keys(id_rsa.pub)
rsync -zarvm -e ssh --include="*/" --include="*.npy" --exclude="*" wezw@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim1836/Training/trained_models/ ${base_path}/trained_models/
