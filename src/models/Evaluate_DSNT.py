import tensorflow as tf
from tensorflow import keras
import numpy as np

import models


def get_record_dir(args_dict):
    cut = args_dict.get("cut_layers")
    crop_layers = np.asarray([[cut[0], cut[1]], [cut[2], cut[3]], [cut[4], cut[5]]])

    trans_tag = args_dict.get("trans_tag")
    dataset_tag = args_dict.get("dataset_tag")

    model_name = args_dict.get("model_name")

    crop_size = args_dict.get("crop_dataset_size")
    input_shape = (crop_size[0]+crop_size[1]-crop_layers[0, 0]-crop_layers[0, 1],
                   crop_size[2]+crop_size[3]-crop_layers[1, 0]-crop_layers[1, 1],
                   crop_size[4]+crop_size[5]-crop_layers[2, 0]-crop_layers[2, 1])
    model_size = f"{input_shape[0]}x{input_shape[1]}x{input_shape[2]}"
    # y_tag: "one_landmark", "two_landmarks", "mean_two_landmarks"
    y_tag = args_dict.get("y_tag")

    save_base = "/data/gpfs/projects/punim1836/Training/trained_models"
    save_dir = f"{save_base}/{dataset_tag}_dataset/{model_name}/{y_tag}/{model_size}/{trans_tag}"
    save_dir_extend = args_dict.get("save_dir_extend")
    save_dir = f"{save_dir}/{save_dir_extend}"

    return save_dir


args = {
    # prepare Dataset
    "dataset_tag": "cropped",
    "crop_dataset_size": [75, 75, 75, 75, 50, 50],
    "cut_layers": [25, 25, 25, 25, 0, 0],
    "has_trans": "_trans/tmp",
    "trans_tag": "s1_test_dis",
    "base_dir": "/data/gpfs/projects/punim1836/Data/cropped/based_on_truth",
    # training
    "batch_size": 2,
    "epochs": 100,
    # model
    "model_name": "cpn_fc_model",
    "model_output_num": 2,
    # record
    "y_tag": "two_landmarks_res",  # "one_landmark", "two_landmarks", "mean_two_landmarks"
    "save_dir_extend": "",  # can be used for cross validation
}

save_dir = get_record_dir(args)

dataset_tag = args.get("dataset_tag")
crop_size = args.get("crop_dataset_size")
cut = args.get("cut_layers")
has_trans = args.get("has_trans")
trans_tag = args.get("trans_tag")
base_dir = args.get("base_dir")

crop_layers = np.asarray([[cut[0], cut[1]], [cut[2], cut[3]], [cut[4], cut[5]]])
crop_tag = f"x{crop_size[0]}{crop_size[1]}y{crop_size[2]}{crop_size[3]}z{crop_size[4]}{crop_size[5]}"
crop_tag_dir = crop_tag + has_trans

x_path = f"{base_dir}/{crop_tag_dir}/cropped_volumes_{crop_tag}_truth_{trans_tag}.npy"
y_path = f"{base_dir}/{crop_tag_dir}/cropped_points_{crop_tag}_truth_{trans_tag}.npy"
cropped_length_path = f"{base_dir}/{crop_tag_dir}/cropped_length_{crop_tag}_truth_{trans_tag}.npy"

print("Read Volumes from:   ", x_path)
print("Read Points from:    ", y_path)
print("Read Length from:    ", cropped_length_path)

batch_size = args.get("batch_size", 2)
epochs = args.get("epochs", 100)

# Get model.
model_name = args.get("model_name")
input_shape = (crop_size[0]+crop_size[1]-crop_layers[0, 0]-crop_layers[0, 1],
               crop_size[2]+crop_size[3]-crop_layers[1, 0]-crop_layers[1, 1],
               crop_size[4]+crop_size[5]-crop_layers[2, 0]-crop_layers[2, 1])
model_output_num = args.get("model_output_num")

model_size = f"{input_shape[0]}x{input_shape[1]}x{input_shape[2]}"
model_label = f"{model_name}_{dataset_tag}_{model_size}"

base_cor_rcs = models.coordinate_3d(batch_size, model_output_num, input_shape[0], input_shape[1], input_shape[2])

model_path = f"{save_dir}/bestVal_{model_label}"

model = keras.models.load_model(model_path)