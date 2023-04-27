import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np

import Functions.MyDataset as MyDataset
import support_modules as supporter


def fun_1():
    return


@tf.function
def predict(x, model_f):
    y_pred_f = model_f(x, training=False)
    return y_pred_f


def predict_dataset(model_path_f, x_test):
    # Load the Trained Model
    model = keras.models.load_model(model_path_f)
    model.summary()

    test_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(2)

    for step, x_batch_test in enumerate(test_dataset):
        # print("Processing step: ", step)
        if step == 0:
            y_test_pred = predict(x_batch_test, model)
        else:
            y_test = predict(x_batch_test, model)
            y_test_pred = np.concatenate((y_test_pred, y_test), axis=0)

    return y_test_pred


# Get the Test Dataset Prediction Results
# size = (176, 176, 48)
# with_res = True
#
# str_size = str(size[0]) + "_" + str(size[1]) + "_" + str(size[2])
# if with_res:
#     str_size = str_size + "_PD"
#
# X_test, Y_test, res_test = \
#     supporter.load_dataset("/data/gpfs/projects/punim1836/Data/rescaled_data/" + str_size + "/",
#                            size, pat_splits=MyDataset.get_pat_splits(static=True), with_res=with_res,
#                            only_test=True)

rescaled_size = (176, 176, 48)
w = np.ceil(rescaled_size[1]/2).astype(int)
str_size = str(rescaled_size[0]) + "_" + str(rescaled_size[1]) + "_" + str(rescaled_size[2])
dataset_dir = f"/data/gpfs/projects/punim1836/Data/divided/" \
              f"{str(rescaled_size[0])}{str(rescaled_size[1])}{str(rescaled_size[2])}/"

pat_splits = MyDataset.get_pat_splits(static=True)
X_train, Y_train, res_train, length_train, X_val, Y_val, res_val, length_val, X_test, Y_test, res_test, length_test = \
    supporter.load_dataset_divide(dataset_dir, rescaled_size, pat_splits)

# y_tag: "one_landmark" -> OL, "two_landmarks" -> TL, "mean_two_landmarks" -> MTL
# mapped: mapped back to original Volume
y_tag = "mean_two_landmarks"
model_name = "straight_model"
model_tag = "divided"
model_size = f"{rescaled_size[0]}_{w}_{rescaled_size[2]}"
model_label = f"{model_name}_{model_tag}_{model_size}"
save_dir = f"/data/gpfs/projects/punim1836/Training/trained_models/{model_tag}_dataset/{model_name}/{y_tag}/"

# some configuration (make life easier?)
has_y_test_pred = False
model_path = f"{save_dir}bestVal_{model_label}"
file_name = f"{save_dir}bestVal_{model_label}_{y_tag}_MTL_p"


if not has_y_test_pred:
    Y_test_pred = predict_dataset(model_path, X_test)
    # map back to original rescaled volume
    Y_test_pred_mapped = np.copy(Y_test_pred)
    Y_test_pred_mapped[:, :, 0] = Y_test_pred[:, :, 0] + length_test
    Y_test_pred_mapped = Y_test_pred_mapped * res_test / 0.15
    np.save(file_name, Y_test_pred_mapped)
    print("Saved: ", file_name)
else:
    Y_test_pred = np.load(sys.argv[1])

print("Y_test_pred Shape: ", np.shape(Y_test_pred))
# print(Y_test_pred_mapped[0:10])

Y_test_mean = np.mean(Y_test, axis=1).reshape((400, 1, 3))
Y_test_mean_mapped = Y_test_mean
err_diff = Y_test_mean - Y_test_pred
square_err_diff = tf.pow(err_diff, 2)
sum_square_err_diff = tf.reduce_sum(square_err_diff, axis=[1, 2])

min_err_idx = np.argmin(sum_square_err_diff, axis=0)
max_err_idx = np.argmax(sum_square_err_diff, axis=0)
print(f"Min[{min_err_idx}]: {sum_square_err_diff[min_err_idx]}")
print(f"Max[{max_err_idx}]: {sum_square_err_diff[max_err_idx]}")
print(f"Mean: {np.mean(sum_square_err_diff)}")

# Evaluation
