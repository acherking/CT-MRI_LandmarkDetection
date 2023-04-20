import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np

import Functions.MyDataset as MyDataset
import support_modules as supporter

# some configuration (make life easier?)
has_y_test_pred = True
file_name = "bestVal_first_model_y_test"


def fun_1():
    return


@tf.function
def predict(x, model_f):
    y_pred_f = model_f(x, training=False)
    return y_pred_f


def predict_dataset(x_test):
    # Load the Trained Model
    model_path = sys.argv[1]
    model = keras.models.load_model(model_path)
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
size = (176, 176, 48)
with_res = True

str_size = str(size[0]) + "_" + str(size[1]) + "_" + str(size[2])
if with_res:
    str_size = str_size + "_PD"

X_test, Y_test, res_test = \
    supporter.load_dataset("/data/gpfs/projects/punim1836/Data/rescaled_data/" + str_size + "/",
                           size, pat_splits=MyDataset.get_pat_splits(static=True), with_res=with_res,
                           only_test=True)

if not has_y_test_pred:
    Y_test_pred = predict_dataset(X_test)
    np.save(file_name, Y_test_pred)
else:
    Y_test_pred = np.load(sys.argv[1])

print("Y_test_pred Shape: ", np.shape(Y_test_pred))

# Evaluation
