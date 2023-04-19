#!/bin/zsh

base_path=/Users/achs/PhD/code/CT-MRI_LandmarkDetection/Spartan

# set rsa pass in local ~/.ssh/id_rsa and remote ~/.ssh/authorized_keys(id_rsa.pub)
rsync -avz -e ssh wezw@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim1836/Training/trained_models/y_test_pred/ ${base_path}/Output/y_test_pred
