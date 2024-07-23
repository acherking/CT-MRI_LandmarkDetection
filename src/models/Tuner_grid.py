import sys

import keras_tuner
import Training
import k_fold_cross_validation
import start_training
import numpy as np
import random


def grid_search():
    learning_rates = np.arange(5e-5, 3e-4, 5e-5)
    decay_steps = np.arange(500, 2000, 250)
    batch_sizes = np.arange(2, 10, 2)
    optimizers = ['Adam', 'SGD']
    full_grid = []
    for lr in learning_rates:
        for ds in decay_steps:
            for bs in batch_sizes:
                for opt in optimizers:
                    full_grid.append(
                        {"learning_rate": lr, "decay_steps": ds, "batch_size": bs, "optimizer": opt})
    return full_grid


def random_pick(pick_num):
    full_grid = grid_search()
    pick_len = int(len(full_grid) / pick_num)
    grid_random_pick = []
    for i in range(pick_num):
        grid_random_pick.append(full_grid[i*pick_len:(i+1)*pick_len])
    for i in range(pick_num):
        random.shuffle(grid_random_pick[i])
    return grid_random_pick


# python Tuner_grid.py 4 [0,1,2,3]
if __name__ == "__main__":
    worker_num = int(sys.argv[1])
    worker_id = int(sys.argv[2])
    try_num = 15

    base_args = start_training.base_args

    train_id = 13
    args_update_list = start_training.train_unet_dsnt_model()

    for args_update in args_update_list:
        if args_update['train_id'] == train_id:
            base_args.update(args_update)

    base_args.update({'save_model': False})
    base_args.update({'model_label_1': "random_grid_opt"})
    base_args.update({'model_label_2': f"worker_{worker_id}/{worker_num}"})

    try_parameter_sets = random_pick(worker_num)

    res = []
    for tid in range(try_num):
        print(f"worker[{worker_id}/{worker_num}], tid[{tid}/{try_num}]")
        print(try_parameter_sets[worker_id][tid])
        base_args.update(try_parameter_sets[worker_id][tid])
        res.append(k_fold_cross_validation.k_fold_cross_validation(8, base_args))

    tid_min_res = np.argmin(res)
    print(try_parameter_sets[worker_id][tid_min_res])
