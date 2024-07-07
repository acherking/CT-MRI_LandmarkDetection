import keras_tuner
import Training
import k_fold_cross_validation
import start_training

base_args = start_training.base_args

train_id = 33
args_update_list = start_training.train_down_net_model()

for args_update in args_update_list:
    if args_update['train_id'] == train_id:
        base_args.update(args_update)

base_args.update({'model_label_1': "keras_tuner_training_process"})


class MyTuner(keras_tuner.RandomSearch):
    def run_trial(self, trial, **kwargs):
        hp = trial.hyperparameters
        # hyper-parameter
        lr = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log")
        bs = hp.Int("bs", min_value=2, max_value=10, step=2)
        optimizer = hp.Choice("optimizer", ['Adam', 'SGD'])
        decay_steps = hp.Int("decay_steps", min_value=100, max_value=1000, step=50)
        base_args.update({'model_label_2': f"{trial.trial_id}.keras_tuner"})
        base_args.update({'learning_rate': lr, "batch_size": bs, "optimizer": optimizer, "decay_steps": decay_steps})
        return k_fold_cross_validation.k_fold_cross_validation(8, base_args)


class MyTunerExtra(keras_tuner.RandomSearch):
    def run_trial(self, trial, **kwargs):
        hp = trial.hyperparameters
        # hyper-parameter
        lr = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log")
        bs = hp.Int("bs", min_value=2, max_value=10, step=2)
        optimizer = hp.Choice("optimizer", ['Adam', 'SGD'])
        decay_steps = hp.Int("decay_steps", min_value=100, max_value=1000, step=50)
        base_args.update({'model_label_2': f"{trial.trial_id}.keras_tuner"})
        base_args.update({'learning_rate': lr, "batch_size": bs, "optimizer": optimizer, "decay_steps": decay_steps})
        # tune the model's architecture
        base_args.update({'model_name': "down_net_dynamic", 'hp': hp})
        return k_fold_cross_validation.k_fold_cross_validation(8, base_args)


if __name__ == "__main__":

    # tuner = MyTuner(
    #     max_trials=50,
    #     overwrite=True,
    #     directory="keras_tuner_dir",
    #     project_name="keras_tuner_training_process",
    # )
    tuner = MyTunerExtra(
        max_trials=100,
        overwrite=True,
        directory="keras_tuner_dir",
        project_name="keras_tuner_training_arch",
    )
    tuner.search()
    # Retraining the model
    best_hp = tuner.get_best_hyperparameters()[0]