import Functions.MyDataset as MyDataset
import Training_divided_volume

if __name__ == "__main__":

    k = 5
    k_idx_splits = MyDataset.get_k_folds_data_splits(k)

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

    for i in range(k):
        save_dir_extend = {"save_dir_extend": f"{k}_cross/{i}"}
        args.update(save_dir_extend)

        Training_divided_volume.train_model(k_idx_splits[i], args)
