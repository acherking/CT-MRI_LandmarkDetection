import start_training
import Training

import numpy as np


# 20 patients in total
# [80% Training, 20% Test] which is: 16 patients for training, 4 patients for testing
# k folds is for Training Dataset; if k=8, 14 training and 2 validation; 8 times validation; use the mean
def k_fold_cross_validation(k, args_dict):
    args_dict.update({'k_cross_num': k})
    args_dict.update({'data_split_tag': "cross_val"})

    k_val = []
    for k_idx in range(k):
        print(f"start cross validation: {k_idx} / {k}")
        args_dict.update({'k_cross_idx': k_idx})
        k_val.append(Training.train_model(args_dict))
        print(k_val)

    # return mean of all the folds' validation result
    return np.asarray([k_val]).mean()


if __name__ == "__main__":
    base_args = start_training.base_args

    train_id = 33
    args_update_list = start_training.train_down_net_model()

    for args_update in args_update_list:
        if args_update['train_id'] == train_id:
            base_args.update(args_update)

    k_folds_num = 16
    base_args.update({'model_label_1': "cross_val_train_id_33_tmp"})

    k_fold_cross_validation(k_folds_num, base_args)
