import tensorflow as tf
from tensorflow import keras
import numpy as np

import common.MyDataset as MyDataset
import TrainingSupport as supporter

import start_training
import loss_optimizer
import models as my_models

# model_name = "down_net_dsnt"
# model_tag = "dsnt"
# train_id = 13.1
# time_tag = "14Aug2024-04:36:02-trainID-13.1"

model_name = "down_net"
model_tag = "fc"
train_id = 33.3
time_tag = "19Nov2025-13:17:24-trainID-33.3"

args_dict = start_training.base_args.copy()

# args_list = start_training.train_unet_dsnt_model()
args_list = start_training.train_down_net_model()

for args_update in args_list:
    if args_update["train_id"] == train_id:
        args_dict.update(args_update)
        print("Using args: ", args_dict)
        break

eval_metrics_fn = loss_optimizer.eval_metric_manager(args_dict)

@tf.function
def test_step(model, x):
    y_pred = model(x, training=False)
    return y_pred

def my_evaluate(model, eval_dataset):
    # evaluate the val or test dataset
    for s, (x_batch_test, y_batch_test, res_batch_test) in enumerate(eval_dataset):
            if s == 0:
                (y_pred, y_true, res) = (test_step(model, x_batch_test), y_batch_test, res_batch_test)
            else:
                y_pred = np.concatenate((y_pred, test_step(model, x_batch_test)), axis=0)
                y_true = np.concatenate((y_true, y_batch_test), axis=0)
                res = np.concatenate((res, res_batch_test), axis=0)
        # prepare the evaluation metrics
    evms = eval_metrics_fn(y_true, y_pred, res, args_dict)

    return evms, y_true, y_pred, res


###
# Start main process
###
crop_layers = np.asarray(args_dict.get("cut_layers", [[0, ], [0, 0], [0, 0]]))
crop_size = (100, 100, 100)

# for ct_pre_14
crop_tag = "100x100x100"
base_dir = "/data/gpfs/projects/punim1836/Data/train/CT_Pre_14/cropped/100x100x100/noises_s1_test_dis"
# base_dir = "/data/gpfs/projects/punim1836/Data/cropped/100x100x100/noises_s1_test_dis"

X_path = f"{base_dir}/cropped_volumes_{crop_tag}.npy"
Y_path = f"{base_dir}/cropped_points_{crop_tag}.npy"
Cropped_length_path = f"{base_dir}/cropped_length_{crop_tag}.npy"

pat_splits = MyDataset.get_pat_splits(static=True)

X_test, Y_test, length_test = supporter.load_dataset_crop_test_only(X_path, Y_path, Cropped_length_path, crop_layers)

ins_num = X_test.shape[0]

# Y_train_one = np.asarray(Y_train)[:, 0, :].reshape((1400, 1, 3))
# Y_val_one = np.asarray(Y_val)[:, 0, :].reshape((200, 1, 3))
# Y_test_one = np.asarray(Y_test)[:, 0, :].reshape((400, 1, 3))

# res_train = (np.ones((1400, 1, 3)) * 0.15).astype('float32')
# res_val = (np.ones((200, 1, 3)) * 0.15).astype('float32')
res_test = (np.ones((ins_num, 1, 3)) * 0.15).astype('float32')

# adjust Y for dsnt you know, if the model is dsnt haha
if model_tag == "dsnt":
    (row_size, column_size, slice_size) = (X_test.shape[1], X_test.shape[2], X_test.shape[3])
    Y_test = (2 * Y_test - [column_size + 1, row_size + 1, slice_size + 1]) / [column_size, row_size,
                                                                                        slice_size]
    res_test = (res_test / [2 / column_size, 2 / row_size, 2 / slice_size])

    # convert to float32
    Y_test = Y_test.astype(np.float32)
    res_test = res_test.astype(np.float32)

print("Test dataset shape: ", X_test.shape, Y_test.shape, res_test.shape)   
print(Y_test[:5])
print(res_test[:5])

save_dir = supporter.get_record_dir(args_dict, get_dir=True)
# Change the last part of the save_dir
save_dir = save_dir.rsplit('/', 1)[0] + '/' + time_tag
print("Save dir: ", save_dir)

true_Y_path = f"{save_dir}/Y_CTPre14_true.npy"
np.save(true_Y_path, Y_test)
print("Saved: ", true_Y_path)

res_path = f"{save_dir}/res_CTPre14.npy"
np.save(res_path, res_test)
print("Saved: ", res_path)

length_path = f"{save_dir}/length_CTPre14.npy"
np.save(length_path, length_test)
print("Saved: ", length_path)

exit()

pred_file_path = f"{save_dir}/best_val_Y_CTPre14_pred.npy"

input_shape = X_test.shape[1:]  # (depth, height, width)
model_output_num = Y_test.shape[1]  # number of landmarks
batch_size = args_dict.get("batch_size", 2)

# model = my_models.u_net_dsnt_model(input_shape[0], input_shape[1], input_shape[2], model_output_num, batch_size, 0)
model = my_models.down_net_model(input_shape[0], input_shape[1], input_shape[2], model_output_num)
weight_path = f"{save_dir}/best_val_model.weights.h5"
model.load_weights(weight_path)

dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test, res_test)).batch(batch_size)
evms, y_true, Y_test_pred, res = my_evaluate(model, dataset)

np.save(pred_file_path, Y_test_pred)

print("Saved: ", pred_file_path)
print("test_dataset MSE(mm^2): ", evms)