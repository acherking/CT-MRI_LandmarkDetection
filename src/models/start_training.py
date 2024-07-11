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
    "save_model": False,
}


# down_net_org
def train_down_net_org_model():
    # record the args?
    update_args_dict_list = [
        ## Variable Voxel distance with MSE_res
        {"train_id": 1, "model_name": "down_net_org", "model_output_num": 1, "y_tag": "mean_two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_label_1": "variable_voxel_distance"},
        {"train_id": 2, "model_name": "down_net_org", "model_output_num": 1, "y_tag": "mean_two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_label_1": "variable_voxel_distance"},
        # cropped data 100x100x100
        {"train_id": 3, "model_name": "down_net_org", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
        # new crop size
        {"train_id": 4, "model_name": "down_net_org", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]]},
        {"train_id": 5, "model_name": "down_net_org", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0002,
         "model_label_1": "learning_rate", "model_label_2": "0.0002", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]]},
        {"train_id": 6, "model_name": "down_net_org", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00008,
         "model_label_1": "learning_rate", "model_label_2": "0.00008", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]]}
    ]
    return update_args_dict_list


## down_net
def train_down_net_model():
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
        # new crop size
        {"train_id": 13, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[18, 13], [25, 21], [15, 18]]},
        {"train_id": 14, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[21, 21], [29, 25], [22, 22]]},
        {"train_id": 15, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[23, 23], [31, 27], [25, 27]]},
        {"train_id": 16, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]]},
        {"train_id": 17, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[26, 29], [35, 31], [29, 33]]},
        ## try new noises (less noises)
        {"train_id": 18, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[18, 13], [25, 21], [15, 18]]},
        {"train_id": 19, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[21, 21], [29, 25], [22, 22]]},
        {"train_id": 20, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[23, 23], [31, 27], [25, 27]]},
        {"train_id": 21, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]]},  # **
        {"train_id": 22, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[26, 29], [35, 31], [29, 33]]},
        ## try even more (noise S1)
        {"train_id": 23, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[27, 31], [37, 33], [30, 37]]},
        {"train_id": 24, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[29, 33], [38, 35], [32, 41]]},
        {"train_id": 25, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[29, 37], [39, 35], [34, 43]]},
        {"train_id": 26, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[31, 38], [41, 37], [35, 45]]},
        {"train_id": 27, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[32, 41], [41, 38], [37, 47]]},
        ## try even more (noise S1.5)
        {"train_id": 28, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[27, 31], [37, 33], [30, 37]]},
        {"train_id": 29, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[29, 33], [38, 35], [32, 41]]},
        {"train_id": 30, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[29, 37], [39, 35], [34, 43]]},
        {"train_id": 31, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[31, 38], [41, 37], [35, 45]]},
        {"train_id": 32, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[32, 41], [41, 38], [37, 47]]},
        ## low scale, noise S1
        ### {'mean_dis_all': 0.238, 'std_dev_all': 0.124, 'mean_dis_1': 0.182, 'std_dev_1': 0.103, 'mean_dis_2': 0.294, 'std_dev_2': 0.118}
        {"train_id": 33, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]]},  # ***
        {"train_id": 33.1, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "batch_size": 6},
        ### {'mean_dis_all': 0.264, 'std_dev_all': 0.134, 'mean_dis_1': 0.197, 'std_dev_1': 0.103, 'mean_dis_2': 0.33, 'std_dev_2': 0.129}
        {"train_id": 33.2, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00025609,
         "model_label_1": "turner-results", "model_label_2": "training-process", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "batch_size": 10, "decay_steps": 300, "epochs": 72,
         "data_split_tag": "test"},
        ### {'mean_dis_all': 0.239, 'std_dev_all': 0.125, 'mean_dis_1': 0.177, 'std_dev_1': 0.096, 'mean_dis_2': 0.301, 'std_dev_2': 0.121}
        {"train_id": 33.3, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00027237,
         "model_label_1": "turner-results", "model_label_2": "training-process", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "batch_size": 8, "decay_steps": 1000, "epochs": 78,
         ### just learning rate, and decay steps
         ### learning rate: 0.00015939, decay steps: 9000
         "data_split_tag": "test"},
        {"train_id": 34, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[22, 21], [30, 26], [23, 25]]},
        {"train_id": 35, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[24, 25], [33, 29], [26, 29]]},
        {"train_id": 36, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 29], [34, 30], [29, 33]]},
        ## low scale, noise S1.5
        {"train_id": 37, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]]},
        {"train_id": 38, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[22, 21], [30, 26], [23, 25]]},
        {"train_id": 39, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[24, 25], [33, 29], [26, 29]]},
        {"train_id": 40, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 29], [34, 30], [29, 33]]},
        # corrode exp with S1.5 noise
        {"train_id": 46, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[9, 31], [19, 30], [28, 11]]},
        {"train_id": 47, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[10, 35], [21, 31], [30, 13]]},
        {"train_id": 48, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[12, 36], [21, 31], [31, 14]]},
        {"train_id": 49, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[13, 37], [23, 32], [32, 15]]},
        {"train_id": 50, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[14, 38], [25, 32], [32, 16]]},
        {"train_id": 51, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[15, 38], [27, 32], [33, 16]]},
        {"train_id": 52, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[16, 39], [37, 33], [34, 17]]},
        {"train_id": 53, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[16, 39], [37, 33], [35, 17]]},
        {"train_id": 54, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[18, 40], [39, 33], [35, 18]]},
        # *********
        # try to optimize the model with choice of cut layer
        # dataset noises_s1_test_dis   [[20, 17], [27, 23], [19, 20]]
        # dataset noises_s1.5_test_dis [[25, 27], [33, 29], [28, 30]]
        # *********
        ## S1
        ### learning rate 0.0001 is better
        {"train_id": 41, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0003,
         "model_label_1": "learning_rate", "model_label_2": "0.0003", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]]},
        {"train_id": 42, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00006,
         "model_label_1": "learning_rate", "model_label_2": "0.00006", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]]},
        ## S1.5
        ### 0.0003 same as 0.0001, 0.00006 little bit decrease, but not many differences. can stick on 0.0001
        ### more stable than S1 seems
        {"train_id": 43, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0003,
         "model_label_1": "learning_rate", "model_label_2": "0.0003", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]]},
        {"train_id": 44, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00006,
         "model_label_1": "learning_rate", "model_label_2": "0.00006", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]]},
        ## S1.5 with 100x100x100
        {"train_id": 45, "model_name": "straight_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
    ]
    return update_args_dict_list


# down_net_dsnt
def train_down_net_dsnt_model():
    update_args_dict_list = [
        ## 1.061, bias?
        {"train_id": 1.1, "model_name": "down_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
        ## 2.051, no overfitting, bias? worse than S1 interesting
        {"train_id": 1.2, "model_name": "down_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
        # *********
        # try to optimize the model with choice of cut layer
        # dataset noises_s1_test_dis   [[20, 17], [27, 23], [19, 20]]
        # dataset noises_s1.5_test_dis [[25, 27], [33, 29], [28, 30]]
        # *********
        ## S1
        ### 0.423 not overfitting, interesting
        {"train_id": 2, "model_name": "down_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "save_model": False},
        ### 0.389, seems not overfitting
        {"train_id": 3, "model_name": "down_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "save_model": False},
        ### 0.447, seems learning rate should between 1e-5 to 5e-5
        {"train_id": 3.1, "model_name": "down_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00001,
         "model_label_1": "learning_rate", "model_label_2": "0.00001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "save_model": False},
        ## S1.5
        ### 0.791, seems no overfitting
        {"train_id": 4, "model_name": "down_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False},
        ### 0.798, learning rate not influence much
        {"train_id": 5, "model_name": "down_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False},
    ]
    return update_args_dict_list


# down_net_short
def train_down_net_short_model():
    update_args_dict_list = [
        ### 0.494, seems no overfitting, good enough in S1 for a very simple model
        ### if I use this to do overfitting, will it focus more on import area?
        {"train_id": 1.1, "model_name": "down_net_short", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
        ### 0.373, seems no overfitting, good
        {"train_id": 1.2, "model_name": "down_net_short", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
        # *********
        # try to optimize the model with choice of cut layer
        # dataset noises_s1_test_dis   [[20, 17], [27, 23], [19, 20]]
        # dataset noises_s1.5_test_dis [[25, 27], [33, 29], [28, 30]]
        # *********
        ## S1
        ### 0.406, overfitting about 2x
        {"train_id": 2, "model_name": "down_net_short", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "save_model": False},
        ### 0.49, overfitting about 2x, learning rate should increase?
        {"train_id": 3, "model_name": "down_net_short", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "save_model": False},
        ### 0.391, less overfitting, can try more learning rate? 0.001?
        {"train_id": 2.1, "model_name": "down_net_short", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0005,
         "model_label_1": "learning_rate", "model_label_2": "0.0005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "save_model": False},
        ## S1.5
        ### 0.215, seems no overfitting
        {"train_id": 4, "model_name": "down_net_short", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False},
        ### 0.258, no overfitting, seems should decrease learning rate
        {"train_id": 5, "model_name": "down_net_short", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False},
    ]
    return update_args_dict_list


# down_net_short_dsnt
def train_down_net_short_dsnt_model():
    update_args_dict_list = [
        ### 0.467 overfitting x5
        {"train_id": 1.1, "model_name": "down_net_short_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
        ### 0.427 overfitting x10
        {"train_id": 1.2, "model_name": "down_net_short_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
        # *********
        # try to optimize the model with choice of cut layer
        # dataset noises_s1_test_dis   [[20, 17], [27, 23], [19, 20]]
        # dataset noises_s1.5_test_dis [[25, 27], [33, 29], [28, 30]]
        # *********
        ## S1
        ### 0.475 no overfitting
        {"train_id": 2, "model_name": "down_net_short_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "save_model": False},
        ### 0.491, seems not overfitting
        {"train_id": 3, "model_name": "down_net_short_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "save_model": False},
        ## S1.5
        ### 0.812, seems not overfitting
        {"train_id": 4, "model_name": "down_net_short_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False},
        ### 0.806, seems not overfitting
        {"train_id": 5, "model_name": "down_net_short_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False},
    ]
    return update_args_dict_list


# u_net
def train_unet_model():
    update_args_dict_list = [
        {"train_id": 1, "model_name": "u_net", "model_output_num": 2, "y_tag": "two_landmarks", "model_label_1": "basic"},
        # *********
        # try to optimize the model with choice of cut layer
        # dataset noises_s1_test_dis   [[20, 17], [27, 23], [19, 20]]
        # dataset noises_s1.5_test_dis [[25, 27], [33, 29], [28, 30]]
        # *********
        ### precision worse than 100x100x100, but less overfitting...
        ## S1
        ### 0.426, emmm why do I try this?
        {"train_id": 2, "model_name": "u_net", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "save_model": False},
        ### 0.448, learning rate not influence much?
        {"train_id": 3, "model_name": "u_net", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "save_model": False},
        ## S1.5
        ### not finish
        {"train_id": 4, "model_name": "u_net", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False},
        {"train_id": 5, "model_name": "u_net", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False},
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
        # *********
        # try to optimize the model with choice of cut layer
        # dataset noises_s1_test_dis   [[20, 17], [27, 23], [19, 20]]
        # dataset noises_s1.5_test_dis [[25, 27], [33, 29], [28, 30]]
        # *********
        ### precision worse than 100x100x100, but less overfitting...
        ## S1
        {"train_id": 13, "model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]]},
        {"train_id": 14, "model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]]},
        ## S1.5
        {"train_id": 15, "model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]]},
        {"train_id": 16, "model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]]},
        # u_net_mini_dsnt
        ## S1
        ### worse than u_net_dsnt, seems not because of the learning rate
        {"train_id": 17, "model_name": "u_net_mini_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]]},
        {"train_id": 19, "model_name": "u_net_mini_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]]},
        ## S1.5
        ### similar to u_net_dsnt
        {"train_id": 18, "model_name": "u_net_mini_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]]},
        ## try 100x100x100
        ## 0.504
        {"train_id": 20, "model_name": "u_net_mini_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
        ## 0.516
        {"train_id": 21, "model_name": "u_net_mini_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[5, 5], [5, 5], [5, 5]]},
        ## 0.492
        {"train_id": 22, "model_name": "u_net_mini_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[10, 10], [10, 10], [10, 10]]},
        ## 0.654
        {"train_id": 23, "model_name": "u_net_mini_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[15, 15], [15, 15], [15, 15]]},
        # u_net_mini_bn_dsnt
        ## S1
        ### precision not improved compare to without bn
        {"train_id": 24, "model_name": "u_net_mini_bn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]]},
        {"train_id": 25, "model_name": "u_net_mini_bn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]]},
        ## S1.5
        ### precision not improved compare to without bn
        {"train_id": 26, "model_name": "u_net_mini_bn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]]},
        ## try 100x100x100
        {"train_id": 27, "model_name": "u_net_mini_bn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
        # u_net_mini_upsample_rep_dsnt
        {"train_id": 28, "model_name": "u_net_mini_upsample_rep_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
        {"train_id": 29, "model_name": "u_net_mini_upsample_rep_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]]},
        # try increase batch size
        ## no improvement, 1.378
        {"train_id": 30, "model_name": "u_net_mini_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00001,
         "model_label_1": "batch_size", "model_label_2": "10", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "batch_size": 10, "save_model": False},
        ## no improvement, 1.312
        {"train_id": 31, "model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "batch_size", "model_label_2": "10", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "batch_size": 10, "save_model": False},
        ## no improvement, 1.358
        {"train_id": 32, "model_name": "u_net_mini_bn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "batch_size", "model_label_2": "10", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]],  "batch_size": 10, "save_model": False},
        # try % 2x2 shape
        ## may not work, can try later
        {"train_id": 33, "model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[19, 17], [28, 24], [17, 19]], "save_model": False},
        ## no improvement, 1.279
        {"train_id": 34, "model_name": "u_net_mini_bn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[19, 17], [28, 24], [17, 19]], "batch_size": 10, "save_model": False},
        # try u-net corrode experiment based cut
        ## 0.895, better, but just because of bigger size window
        {"train_id": 35, "model_name": "u_net_mini_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 5], [15, 10], [10, 10]], "batch_size": 2, "save_model": False},
        # try different optimizer
        ## 1.608, maybe better after fine turn but emmm not much useful?
        {"train_id": 36, "model_name": "u_net_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[19, 17], [28, 24], [17, 19]], "save_model": False, "optimizer": "SGD"}

    ]
    return update_args_dict_list


# cov_only
def train_covonly_model():
    update_args_dict_list = [
        # *********
        # try to optimize the model with choice of cut layer
        # dataset noises_s1_test_dis   [[20, 17], [27, 23], [19, 20]]
        # dataset noises_s1.5_test_dis [[25, 27], [33, 29], [28, 30]]
        # *********
        ### precision worse than 100x100x100, but less overfitting...
        ## S1
        ### 0.475
        {"train_id": 2, "model_name": "cov_only_fc", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "save_model": False},
        ### 0.439
        {"train_id": 3, "model_name": "cov_only_fc", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "save_model": False},
        ## S1.5
        ### 0.236
        {"train_id": 4, "model_name": "cov_only_fc", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False},
        ### 0.235
        {"train_id": 5, "model_name": "cov_only_fc", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False},
    ]
    return update_args_dict_list


# cov_only_dsnt
def train_covonly_dsnt_model():
    update_args_dict_list = [
        # it seems identical voxel distance works better (stable) for con_only
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
         "model_label_1": "learning_rate_K5", "model_label_2": "0.000001", "dataset_label_1": "variable_voxel_distance"},
        # cropped dataset 100x100x100
        {"train_id": 10, "model_name": "cov_only_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
        # *********
        # try to optimize the model with choice of cut layer
        # dataset noises_s1_test_dis   [[20, 17], [27, 23], [19, 20]]
        # dataset noises_s1.5_test_dis [[25, 27], [33, 29], [28, 30]]
        # *********
        ## S1
        ### precision increased than 100x100x100 little bit, learning rate 0.0001 looks better
        {"train_id": 11, "model_name": "cov_only_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]]},
        {"train_id": 12, "model_name": "cov_only_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]]},
        ## S1.5
        ### precision much better, learning rate not very different seems
        {"train_id": 13, "model_name": "cov_only_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]]},
        {"train_id": 14, "model_name": "cov_only_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]]},
        ## s1.5 with 100x100x100
        {"train_id": 15, "model_name": "cov_only_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]]},
    ]
    return update_args_dict_list


# scn
def train_scn_model():
    update_args_dict_list = [
        # *********
        # try to optimize the model with choice of cut layer
        # dataset noises_s1_test_dis   [[20, 17], [27, 23], [19, 20]]
        # dataset noises_s1.5_test_dis [[25, 27], [33, 29], [28, 30]]
        # *********
        ### precision worse than 100x100x100, but less overfitting...
        ## S1
        ### 0.819
        {"train_id": 2, "model_name": "scn", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "save_model": False},
        ### 0.844
        {"train_id": 3, "model_name": "scn", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "save_model": False},
        ## S1.5
        ### 0.529
        {"train_id": 4, "model_name": "scn", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False},
        ### 0.405, looks like can improve by turing learning rate
        {"train_id": 5, "model_name": "scn", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False},
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
        # try new local_kernel_size and spatial_kernel_size
        ## big local kernel size work better? seems
        {"train_id": 5, "model_name": "scn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "kernel_size", "model_label_2": "local_5x5x5_spatial_15x15x15", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]], "local_kernel_size": (5, 5, 5), "spatial_kernel_size": (15, 15, 15)},
        {"train_id": 6, "model_name": "scn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "kernel_size", "model_label_2": "local_5x5x5_spatial_9x9x5", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]], "local_kernel_size": (5, 5, 5), "spatial_kernel_size": (9, 9, 5)},
        {"train_id": 7, "model_name": "scn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "kernel_size", "model_label_2": "local_3x3x3_spatial_15x15x15", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]], "local_kernel_size": (3, 3, 3), "spatial_kernel_size": (15, 15, 15)},
        ## try even bigger local kernel size
        ## not change much
        {"train_id": 8, "model_name": "scn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "kernel_size", "model_label_2": "local_7x7x7_spatial_9x9x5", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]], "local_kernel_size": (7, 7, 7), "spatial_kernel_size": (9, 9, 5)},
        {"train_id": 9, "model_name": "scn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "LR_0.00005_local_7x7x7_spatial_9x9x5", "dataset_tag": "cropped",
         "dataset_label_1": "noises_s1_test_dis", "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]],
         "local_kernel_size": (7, 7, 7), "spatial_kernel_size": (9, 9, 5)},
        {"train_id": 10, "model_name": "scn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0005,
         "model_label_1": "learning_rate", "model_label_2": "LR_0.0005_local_7x7x7_spatial_9x9x5", "dataset_tag": "cropped",
         "dataset_label_1": "noises_s1_test_dis", "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]],
         "local_kernel_size": (7, 7, 7), "spatial_kernel_size": (9, 9, 5)},
        # *********
        # try to optimize the model with choice of cut layer
        # dataset noises_s1_test_dis   [[20, 17], [27, 23], [19, 20]]
        # dataset noises_s1.5_test_dis [[25, 27], [33, 29], [28, 30]]
        # use local (5,5,5), global (9, 9, 5)
        # *********
        ## S1
        ### precision improved compare to 100x100x100 S1
        {"train_id": 11, "model_name": "scn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]],
         "local_kernel_size": (5, 5, 5), "spatial_kernel_size": (9, 9, 5)},
        {"train_id": 12, "model_name": "scn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]],
         "local_kernel_size": (5, 5, 5), "spatial_kernel_size": (9, 9, 5)},
        ## S1.5
        ### precision better than S1, learning rate 5E-5 seems better? not much different; precision improved compare to 100x100x100 S1.5
        {"train_id": 13, "model_name": "scn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]],
         "local_kernel_size": (5, 5, 5), "spatial_kernel_size": (9, 9, 5)},
        {"train_id": 14, "model_name": "scn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]],
         "local_kernel_size": (5, 5, 5), "spatial_kernel_size": (9, 9, 5)},
        ## S1.5 100x100x100
        {"train_id": 15, "model_name": "scn_dsnt", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]],
         "local_kernel_size": (5, 5, 5), "spatial_kernel_size": (9, 9, 5)},
    ]
    return update_args_dict_list


# cpn
def train_cpn_model():
    update_args_dict_list = [
        ### 1.215, overfitting, model too complex?
        {"train_id": 1.1, "model_name": "cpn_fc_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]], "save_model": False, "loss_name": "2_MSE_res",
         'eval_name': "my_eval_cpn"},
        ### 0.502, overfitting still, seems shifting influenced a lot
        {"train_id": 1.2, "model_name": "cpn_fc_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]], "save_model": False, "loss_name": "2_MSE_res",
         'eval_name': "my_eval_cpn"},
        # *********
        # try to optimize the model with choice of cut layer
        # dataset noises_s1_test_dis   [[20, 17], [27, 23], [19, 20]]
        # dataset noises_s1.5_test_dis [[25, 27], [33, 29], [28, 30]]
        # *********
        ## S1
        ### 1.138 overfitting
        {"train_id": 2, "model_name": "cpn_fc_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "save_model": False, "loss_name": "2_MSE_res",
         'eval_name': "my_eval_cpn"},
        ### 1.137 overfitting
        {"train_id": 3, "model_name": "cpn_fc_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "save_model": False, "loss_name": "2_MSE_res",
         'eval_name': "my_eval_cpn"},
        ## S1.5
        ### 0.445 overfitting, seems cpn can be influenced a lot by the centre translation
        {"train_id": 4, "model_name": "cpn_fc_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False, "loss_name": "2_MSE_res",
         'eval_name': "my_eval_cpn"},
        ### 0.44 overfitting
        {"train_id": 5, "model_name": "cpn_fc_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False, "loss_name": "2_MSE_res",
         'eval_name': "my_eval_cpn"},
        ### 0.417 little bit overfitting
        {"train_id": 6, "model_name": "cpn_fc_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0002,
         "model_label_1": "learning_rate", "model_label_2": "0.0002", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False, "loss_name": "2_MSE_res",
         'eval_name': "my_eval_cpn"},
        ### 0.447, overfitting 10x
        {"train_id": 7, "model_name": "cpn_fc_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0005,
         "model_label_1": "learning_rate", "model_label_2": "0.0005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False, "loss_name": "2_MSE_res",
         'eval_name': "my_eval_cpn"},
    ]
    return update_args_dict_list


# cpn_dsnt
def train_cpn_dsnt_model():
    update_args_dict_list = [
        ### 1.807, no overfitting
        {"train_id": 1.1, "model_name": "cpn_dsnt_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]], "save_model": False, "loss_name": "2_MSE_res",
         'eval_name': "my_eval_cpn"},
        ### 1.722, no overfitting
        {"train_id": 1.2, "model_name": "cpn_dsnt_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[0, 0], [0, 0], [0, 0]], "save_model": False, "loss_name": "2_MSE_res",
         'eval_name': "my_eval_cpn"},
        # *********
        # try to optimize the model with choice of cut layer
        # dataset noises_s1_test_dis   [[20, 17], [27, 23], [19, 20]]
        # dataset noises_s1.5_test_dis [[25, 27], [33, 29], [28, 30]]
        # *********
        ## S1
        ### 1.508, under fitting?
        {"train_id": 2, "model_name": "cpn_dsnt_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "save_model": False, "loss_name": "2_MSE_res",
         'eval_name': "my_eval_cpn"},
        ### 1.701, under fitting?
        {"train_id": 3, "model_name": "cpn_dsnt_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "save_model": False, "loss_name": "2_MSE_res",
         'eval_name': "my_eval_cpn"},
        ### 1.509, seems no overfitting
        {"train_id": 2.1, "model_name": "cpn_dsnt_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.001,
         "model_label_1": "learning_rate", "model_label_2": "0.001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "save_model": False, "loss_name": "2_MSE_res",
         'eval_name': "my_eval_cpn"},
        ### 1.561, bigger overfitting
        {"train_id": 2.2, "model_name": "cpn_dsnt_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.005,
         "model_label_1": "learning_rate", "model_label_2": "0.005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[20, 17], [27, 23], [19, 20]], "save_model": False, "loss_name": "2_MSE_res",
         'eval_name': "my_eval_cpn"},
        ## S1.5
        ### 1.694, under fitting, bias (too general)
        {"train_id": 4, "model_name": "cpn_dsnt_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0001,
         "model_label_1": "learning_rate", "model_label_2": "0.0001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False, "loss_name": "2_MSE_res",
         'eval_name': "my_eval_cpn"},
        ### 2.018, under fitting, bias, seems learning rate will influence
        {"train_id": 5, "model_name": "cpn_dsnt_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.00005,
         "model_label_1": "learning_rate", "model_label_2": "0.00005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False, "loss_name": "2_MSE_res",
         'eval_name': "my_eval_cpn"},
        ### 1.638, no overfitting
        {"train_id": 6, "model_name": "cpn_dsnt_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0002,
         "model_label_1": "learning_rate", "model_label_2": "0.0002", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False, "loss_name": "2_MSE_res",
         'eval_name': "my_eval_cpn"},
        ### 1.513, no overfitting, learning rate do influence
        {"train_id": 7, "model_name": "cpn_dsnt_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.0005,
         "model_label_1": "learning_rate", "model_label_2": "0.0005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False, "loss_name": "2_MSE_res",
         'eval_name': "my_eval_cpn"},
        ### 1.585, look like learning rate should bigger than 0.0001
        {"train_id": 4.1, "model_name": "cpn_dsnt_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.001,
         "model_label_1": "learning_rate", "model_label_2": "0.001", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False, "loss_name": "2_MSE_res",
         'eval_name': "my_eval_cpn"},
        ### 1.516, learning rate can bigger than 0.005
        {"train_id": 4.2, "model_name": "cpn_dsnt_model", "model_output_num": 2, "y_tag": "two_landmarks", "learning_rate": 0.005,
         "model_label_1": "learning_rate", "model_label_2": "0.005", "dataset_tag": "cropped", "dataset_label_1": "noises_s1.5_test_dis",
         "input_shape": (100, 100, 100), "cut_layers": [[25, 27], [33, 29], [28, 30]], "save_model": False, "loss_name": "2_MSE_res",
         'eval_name': "my_eval_cpn"},
    ]
    return update_args_dict_list


if __name__ == "__main__":
    train_model_name = sys.argv[1]
    train_id_list = sys.argv.copy()
    # remove function name and model name
    train_id_list.pop(0)
    train_id_list.pop(0)
    # convert to int
    train_id_list = [float(i) for i in train_id_list]

    print(f"Training model: [{train_model_name}] with ids_list: {train_id_list}.")

    if train_model_name == "down_net":
        args_list = train_down_net_model()
    elif train_model_name == "down_net_org":
        args_list = train_down_net_org_model()
    elif train_model_name == "down_net_dsnt":
        args_list = train_down_net_dsnt_model()
    elif train_model_name == "down_net_short":
        args_list = train_down_net_short_model()
    elif train_model_name == "down_net_short_dsnt":
        args_list = train_down_net_short_dsnt_model()
    elif train_model_name == "u_net":
        args_list = train_unet_model()
    elif train_model_name == "u_net_dsnt":
        args_list = train_unet_dsnt_model()
    elif train_model_name == "cov_only":
        args_list = train_covonly_model()
    elif train_model_name == "cov_only_dsnt":
        args_list = train_covonly_dsnt_model()
    elif train_model_name == "scn":
        args_list = train_scn_model()
    elif train_model_name == "scn_dsnt":
        args_list = train_scn_dsnt_model()
    elif train_model_name == "cpn":
        args_list = train_cpn_model()
    elif train_model_name == "cpn_dsnt":
        args_list = train_cpn_dsnt_model()
    else:
        print("Unknown model name: ", train_model_name)

    for args_update in args_list:
        train_id = args_update.get("train_id")
        if train_id_list.count(train_id) == 1:
            print(f"Found train_id: [{train_id}] in [{train_model_name}] args list, start training.")
            args = base_args.copy()
            args.update(args_update)
            Training.train_model(args)
