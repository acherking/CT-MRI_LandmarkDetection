import sys

import Training

args = {
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
}


def train_straight_model(idx=-1):
    # record the args?
    update_args_dict_list = [
        {"model_name": "straight_model", "model_output_num": 1, "y_tag": "one_landmark_1", "model_label_1": "basic"}]

    if idx < 0:
        for i in range(len(update_args_dict_list)):
            args.update(update_args_dict_list[i])
            Training.train_model(args)
    else:
        args.update(update_args_dict_list[idx])
        Training.train_model(args)

    Training.train_model(args)


def train_unet_model(idx=-1):
    update_args_dict_list = [
        {"model_name": "u_net", "model_output_num": 2, "y_tag": "two_landmarks", "model_label_1": "basic"}]

    if idx < 0:
        for i in range(len(update_args_dict_list)):
            args.update(update_args_dict_list[i])
            Training.train_model(args)
    else:
        args.update(update_args_dict_list[idx])
        Training.train_model(args)

    Training.train_model(args)


def train_unet_dsnt_model(idx=-1):
    update_args_dict_list = [
        {"model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005"},
        {"model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00001,
         "model_label_1": "learning_rate", "model_label_2": "0.00001"},
        {"model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.000005,
         "model_label_1": "learning_rate", "model_label_2": "0.000005"},
        {"model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.000001,
         "model_label_1": "learning_rate", "model_label_2": "0.000001"}]

    if idx < 0:
        for i in range(len(update_args_dict_list)):
            args.update(update_args_dict_list[i])
            Training.train_model(args)
    else:
        args.update(update_args_dict_list[idx])
        Training.train_model(args)

    Training.train_model(args)


if __name__ == "__main__":
    train_model_name = sys.argv[1]
    train_id = int(sys.argv[2])
    if train_model_name == "straight_model":
        train_straight_model(train_id)
    elif train_model_name == "u_net":
        train_unet_model(train_id)
    elif train_model_name == "u_net_dsnt":
        train_unet_dsnt_model(train_id)
    else:
        print("Unknown model name: ", train_model_name)
