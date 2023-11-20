import sys

import Functions.MyDataset as MyDataset
import Training_divided_volume
import Training_cropped_volume
import Training_focus_volume


def train_divided(k_cross_num, k_cross_idx):
    k_pat_splits = MyDataset.get_k_folds_pat_splits(k_cross_num)
    print("K folds patients split: ", k_pat_splits[k_cross_idx])

    idx_split = MyDataset.get_data_splits(k_pat_splits[k_cross_idx], split=True, aug_num=50)

    args = {
        # prepare Dataset
        "dataset_tag": "divided",
        "rescaled_size":  (176, 176, 48),
        "base_dir": "/data/gpfs/projects/punim1836/Data",
        # training
        "batch_size": 2,
        "epochs": 100,
        # model
        "model_name": "straight_model",
        "model_output_num": 1,
        # record
        "y_tag": "mean_two_landmarks",  # "one_landmark", "two_landmarks", "mean_two_landmarks"
        "save_dir_extend": "",  # can be used for cross validation
    }

    save_dir_extend = {"save_dir_extend": f"{k_cross_num}_cross/{k_cross_idx}"}
    args.update(save_dir_extend)

    Training_divided_volume.train_model(idx_split, args)


def train_cropped(k_cross_num, k_cross_idx):
    k_pat_splits = MyDataset.get_k_folds_pat_splits(k_cross_num)
    print("K folds patients split: ", k_pat_splits[k_cross_idx])

    idx_split = MyDataset.get_data_splits(k_pat_splits[k_cross_idx], split=True, aug_num=100)

    args = {
        # prepare Dataset
        "dataset_tag": "cropped",
        "crop_dataset_size": [75, 75, 75, 75, 50, 50],
        # "cut_layers": [39, 39, 39, 39, 26, 26],
        "cut_layers": [25, 25, 25, 25, 0, 0],
        "has_trans": "",
        "trans_tag": "no_trans_100aug_6medium",
        "base_dir": "/data/gpfs/projects/punim1836/Data/cropped/based_on_truth/augment_exp_pythong",
        # training
        "batch_size": 2,
        "epochs": 100,
        # model
        "model_name": "straight_model",
        "model_output_num": 1,
        # record
        "y_tag": "one_landmark",  # "one_landmark", "two_landmarks", "mean_two_landmarks"
        "save_dir_extend": "",  # can be used for cross validation
    }

    save_dir_extend = {"save_dir_extend": f"{k_cross_num}_cross/{k_cross_idx}"}
    args.update(save_dir_extend)

    Training_focus_volume.train_model(idx_split, args)


if __name__ == "__main__":

    k = int(sys.argv[1])
    i = int(sys.argv[2])
    model_tag = str(sys.argv[3])

    i = i - 1  # start from 0

    if model_tag == "divided":
        train_divided(k, i)
    elif model_tag == "cropped":
        train_cropped(k, i)
    else:
        print("There is no training mode: ", model_tag)


