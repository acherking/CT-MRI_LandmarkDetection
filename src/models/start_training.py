import sys

import Training

base_args = {
    # prepare Dataset
    "dataset_tag": "divided",  # "divided", "cropped"
    ## for divided dataset ##
    "input_shape": (176, 88, 48),
    "cut_layers": [[25, 25], [25, 25], [0, 0]],
    "base_dir": "/data/gpfs/projects/punim1836/Data",
    "dataset_label_1": "identical_voxel_distance",
    "data_split_tag": "general",  # "general" - train 14, val 2, test 4; "cross_val"
    "data_split_static": True,
    # training
    "write_log": True,
    "batch_size": 2,
    "epochs": 100,
    "loss_name": "MSE_res",
    "optimizer": "Adam",
    "learning_rate": 0.0001,
    "decay_steps": 10000,
    "decay_rate": 0.96,
    # model
    "model_name": "u_net_dsnt",
    "model_output_num": 2,
    # record
    "save_base_dir": "/data/gpfs/projects/punim1836/CT-MRI_LandmarkDetection/models",
    "y_tag": "two_landmarks",  # "one_landmark_[1/2]", "two_landmarks", "mean_two_landmarks"
    "model_label_1": "",  # Cross validation, different parameter...
    "model_label_2": "",
    "save_model": True,
}


## straight_model
def train_straight_model():
    # record the args?
    update_args_dict_list = [
        # divided data, 176x88x48
        ## Identical Voxel distance with MSE_res
        {"train_id": 0, "model_name": "straight_model", "model_output_num": 1, "y_tag": "mean_two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001"},
        {"train_id": 1, "model_name": "straight_model", "model_output_num": 1, "y_tag": "mean_two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005"},
        ## Variable Voxel distance with MSE_res
        {"train_id": 2, "model_name": "straight_model", "model_output_num": 1, "y_tag": "mean_two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_label_1": "variable_voxel_distance"},
        {"train_id": 3, "model_name": "straight_model", "model_output_num": 1, "y_tag": "mean_two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_label_1": "variable_voxel_distance"},
        {"train_id": 4, "model_name": "straight_model", "model_output_num": 1, "y_tag": "mean_two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "loss_name": "MSE"},
        # cropped data 100x100x100
        {"train_id": 5, "model_name": "straight_model", "model_output_num": 1, "y_tag": "one_landmark_1", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
        {"train_id": 6, "model_name": "straight_model", "model_output_num": 1, "y_tag": "one_landmark_2", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
        {"train_id": 7, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
        {"train_id": 8, "model_name": "straight_model", "model_output_num": 1, "y_tag": "mean_two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
        # divided data, 176x88x96
        ## Identical Voxel distance with MSE_res
        # this is the best for stage 1, it seems
        {"train_id": 9, "model_name": "straight_model", "model_output_num": 1, "y_tag": "mean_two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "input_shape": (176, 88, 96)},
        {"train_id": 10, "model_name": "straight_model", "model_output_num": 1, "y_tag": "one_landmark_1", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "input_shape": (176, 88, 96)},
        {"train_id": 11, "model_name": "straight_model", "model_output_num": 1, "y_tag": "one_landmark_2", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "input_shape": (176, 88, 96)},
        {"train_id": 12, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "input_shape": (176, 88, 96)},
    ]
    return update_args_dict_list


def train_unet_model():
    update_args_dict_list = [
        {"model_name": "u_net", "model_output_num": 2, "y_tag": "two_landmarks", "model_label_1": "basic"}
    ]
    return update_args_dict_list


# u_net_dsnt
def train_unet_dsnt_model():
    update_args_dict_list = [
        ## Identical Voxel distance with MSE_res
        {"train_id": 0, "model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001"},
        {"train_id": 1, "model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005"},
        {"train_id": 2, "model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00001,
         "model_label_1": "learning_rate", "model_label_2": "0.00001"},
        {"train_id": 3, "model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.000005,
         "model_label_1": "learning_rate", "model_label_2": "0.000005"},
        {"train_id": 4, "model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.000001,
         "model_label_1": "learning_rate", "model_label_2": "0.000001"},
        ## Variable Voxel distance with MSE_res
        {"train_id": 5, "model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_label_1": "variable_voxel_distance"},
        {"train_id": 6, "model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_label_1": "variable_voxel_distance"},
        {"train_id": 7, "model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00001,
         "model_label_1": "learning_rate", "model_label_2": "0.00001", "dataset_label_1": "variable_voxel_distance"},
        {"train_id": 8, "model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.000005,
         "model_label_1": "learning_rate", "model_label_2": "0.000005", "dataset_label_1": "variable_voxel_distance"},
        {"train_id": 9, "model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.000001,
         "model_label_1": "learning_rate", "model_label_2": "0.000001", "dataset_label_1": "variable_voxel_distance"},
        ## Identical Voxel distance with MSE
        {"train_id": 10, "model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "loss_name": "MSE"},
        # cropped data
        # still overfitting, landmark_2 not good
        # MSE_res or MSE not change much, it seems
        {"train_id": 11, "model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
        {"train_id": 12, "model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]], "loss_name": "MSE"},
    ]
    return update_args_dict_list


# cov_only_dsnt
def train_covonly_dsnt_model():
    update_args_dict_list = [
        ## Identical Voxel distance with MSE_res
        {"train_id": 0, "model_name": "cov_only_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate_K5", "model_label_2": "0.0001"},
        {"train_id": 1, "model_name": "cov_only_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate_K5", "model_label_2": "0.00005"},
        {"train_id": 2, "model_name": "cov_only_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00001,
         "model_label_1": "learning_rate_K5", "model_label_2": "0.00001"},
        {"train_id": 3, "model_name": "cov_only_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.000005,
         "model_label_1": "learning_rate_K5", "model_label_2": "0.000005"},
        {"train_id": 4, "model_name": "cov_only_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.000001,
         "model_label_1": "learning_rate_K5", "model_label_2": "0.000001"},
        ## Variable Voxel distance with MSE_res
        {"train_id": 5, "model_name": "cov_only_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate_K5", "model_label_2": "0.0001", "dataset_label_1": "variable_voxel_distance"},
        {"train_id": 6, "model_name": "cov_only_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate_K5", "model_label_2": "0.00005", "dataset_label_1": "variable_voxel_distance"},
        {"train_id": 7, "model_name": "cov_only_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00001,
         "model_label_1": "learning_rate_K5", "model_label_2": "0.00001", "dataset_label_1": "variable_voxel_distance"},
        {"train_id": 8, "model_name": "cov_only_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.000005,
         "model_label_1": "learning_rate_K5", "model_label_2": "0.000005", "dataset_label_1": "variable_voxel_distance"},
        {"train_id": 9, "model_name": "cov_only_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.000001,
         "model_label_1": "learning_rate_K5", "model_label_2": "0.000001", "dataset_label_1": "variable_voxel_distance"}
        # cropped dataset 100x100x100

    ]
    return update_args_dict_list


# scn_dsnt
def train_scn_dsnt_model():
    # looks like learning_rate 0.0001 works best
    # MSE_res or MSE doesn't influence much
    update_args_dict_list = [
        # cropped data
        {"train_id": 0, "model_name": "scn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
        {"train_id": 1, "model_name": "scn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
        {"train_id": 2, "model_name": "scn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00001,
         "model_label_1": "learning_rate", "model_label_2": "0.00001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
        {"train_id": 3, "model_name": "scn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0005,
         "model_label_1": "learning_rate", "model_label_2": "0.0005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
        {"train_id": 4, "model_name": "scn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]], "loss_name": "MSE"},
    ]
    return update_args_dict_list


if __name__ == "__main__":
    train_model_name = sys.argv[1]
    train_id_list = sys.argv.copy()
    # remove function name and model name
    train_id_list.pop(0)
    train_id_list.pop(0)
    # convert to int
    train_id_list = [int(i) for i in train_id_list]

    print(f"Training model: [{train_model_name}] with ids_list: {train_id_list}.")

    if train_model_name == "straight_model":
        args_list = train_straight_model()
    elif train_model_name == "u_net":
        args_list = train_unet_model()
    elif train_model_name == "u_net_dsnt":
        args_list = train_unet_dsnt_model()
    elif train_model_name == "cov_only_dsnt":
        args_list = train_covonly_dsnt_model()
    elif train_model_name == "scn_dsnt":
        args_list = train_scn_dsnt_model()
    else:
        print("Unknown model name: ", train_model_name)

    for args_update in args_list:
        train_id = args_update.get("train_id")
        if train_id_list.count(train_id) == 1:
            print(f"Found train_id: [{train_id}] in [{train_model_name}] args list, start training.")
            args = base_args.copy()
            args.update(args_update)
            Training.train_model(args)
