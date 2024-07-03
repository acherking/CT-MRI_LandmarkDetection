import start_training
import Training

base_args = start_training.base_args

train_id = 33
args_update_list = start_training.train_down_net_model()

for args_update in args_update_list:
    if args_update['train_id'] == train_id:
        base_args.update(args_update)

k_folds_num = 20
base_args.update({'k_cross_num': k_folds_num})
base_args.update({'data_split_tag': "cross_val"})
base_args.update({'model_label_1': "cross_val_train_id_33"})

for idx in range(k_folds_num):
    print(f"start cross validation: {idx} / {k_folds_num}")
    base_args.update({'k_cross_idx': idx})
    base_args.update({'model_label_2': str(idx)})
    Training.train_model(base_args)
