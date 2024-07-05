import keras_tuner
import Training
import start_training

base_args = start_training.base_args

train_id = 33
args_update_list = start_training.train_down_net_model()

for args_update in args_update_list:
    if args_update['train_id'] == train_id:
        base_args.update(args_update)

base_args.update({'model_label_1': "keras_tuner_try"})


class MyTuner(keras_tuner.RandomSearch):
    def run_trial(self, trial, **kwargs):
        hp = trial.hyperparameters
        lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        base_args.update({'model_label_2': f"{trial.trial_id}.keras_tuner"})
        base_args.update({'learning_rate': lr})
        return Training.train_model(base_args)


tuner = MyTuner(
    max_trials=10,
    overwrite=True,
    directory="my_dir",
    project_name="try_keras_tuner",
)
tuner.search()
# Retraining the model
best_hp = tuner.get_best_hyperparameters()[0]