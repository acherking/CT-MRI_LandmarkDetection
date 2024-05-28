import os

import tensorflow as tf
from tensorflow import keras
import numpy as np

import Functions.MyDataset as MyDataset
from src.models.common import TrainingSupport as supporter
import models

mse = tf.keras.losses.MeanSquaredError()
mse_res = models.mse_with_res
test_mse_metric = keras.metrics.Mean()


@tf.function
def test_step(x, y, res, eva_model):
    y_pred = eva_model(x, training=False)
    mse_mm2 = mse_res(y, y_pred, res)
    # Update test metrics
    test_mse_metric.update_state(mse_mm2)
    return y_pred


def my_evaluate(eva_model, eva_dataset):
    # evaluate the model on test dataset, here: cropped volume of test_dataset based on previous steps' predictions
    for step, (x_batch_test, y_batch_test, res_batch_test) in enumerate(eva_dataset):
        y = test_step(x_batch_test, y_batch_test, res_batch_test, eva_model)
        if step == 0:
            y_test_pred = np.copy(y)
        else:
            y_test_pred = np.concatenate((y_test_pred, np.copy(y)), axis=0)

    test_mse_res_f = test_mse_metric.result()
    test_mse_metric.reset_states()
    return test_mse_res_f.numpy(), y_test_pred


###
# Start main process
###
crop_layers = np.asarray([[0, 0], [0, 0], [0, 0]])
crop_size = (100, 100, 100)

crop_tag = "x5050y5050z5050"
base_dir = "/data/gpfs/projects/punim1836/Data/cropped/based_on_pred"

X_path = f"{base_dir}/{crop_tag}/cropped_volumes_{crop_tag}_pred.npy"
Y_path = f"{base_dir}/{crop_tag}/cropped_points_{crop_tag}_pred.npy"
Cropped_length_path = f"{base_dir}/{crop_tag}/cropped_length_{crop_tag}_pred.npy"

pat_splits = MyDataset.get_pat_splits(static=True)

X_test, Y_test, length_test = supporter.load_dataset_crop_test_only(X_path, Y_path, Cropped_length_path, pat_splits, crop_layers)

# Y_train_one = np.asarray(Y_train)[:, 0, :].reshape((1400, 1, 3))
# Y_val_one = np.asarray(Y_val)[:, 0, :].reshape((200, 1, 3))
Y_test_one = np.asarray(Y_test)[:, 0, :].reshape((400, 1, 3))

# res_train = (np.ones((1400, 1, 3)) * 0.15).astype('float32')
# res_val = (np.ones((200, 1, 3)) * 0.15).astype('float32')
res_test = (np.ones((400, 1, 3)) * 0.15).astype('float32')

# y_tag: "one_landmark" -> OL, "two_landmarks" -> TL, "mean_two_landmarks" -> MTL
y_tag = "one_landmark_res"
model_name = "straight_model"
model_tag = "cropped_trans"
trans_tag = "/s1_test_dis"  # /s1_all
model_size = f"{crop_size[0]}x{crop_size[1]}x{crop_size[2]}"
model_label = f"{model_name}_{model_tag}_{model_size}"
base_dir = "/data/gpfs/projects/punim1836/Training/trained_models"
model_dir = f"{base_dir}/{model_tag}_dataset/{model_name}/{y_tag}/{model_size}{trans_tag}"
save_dir = f"{model_dir}/final_results"

# create the dir if not exist
if os.path.exists(save_dir):
    print("Save final results to: ", save_dir)
else:
    os.makedirs(save_dir)
    print("Create dir and save final results in it: ", save_dir)

pred_file_path = f"{save_dir}/{model_label}_{y_tag}_Ytest_pred"

# some configuration (make life easier?)
model_path = f"{model_dir}/final_{model_label}"
model = keras.models.load_model(model_path)

dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test_one, res_test)).batch(2)
mse_test_dataset_mm, Y_test_one_pred = my_evaluate(model, dataset)

np.save(pred_file_path, Y_test_one_pred)
print("Saved: ", pred_file_path)

print("test_dataset MSE(mm^2): ", mse_test_dataset_mm)