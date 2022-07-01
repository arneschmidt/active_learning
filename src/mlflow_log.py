from typing import Dict
import mlflow
import os
import copy
import globals
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import backend as K

result_dataframe = pd.DataFrame()

def start_logging():
    mlflow.set_tracking_uri(globals.config["logging"]["tracking_url"])
    experiment_id = mlflow.set_experiment(experiment_name='active_' + globals.config["data"]["dataset_name"])
    mlflow.start_run(experiment_id=experiment_id, run_name=globals.config["logging"]["run_name"])
    config_logging()

def config_logging():
    config = copy.deepcopy(globals.config)
    fe_config = config['model'].pop("feature_extractor")
    log_fe_config = {}
    for key in fe_config:
        log_fe_config['fe_' + key] = fe_config[key]

    head_config = config['model'].pop("head")
    log_head_config = {}
    for key in head_config:
        log_head_config['head_' + key] = head_config[key]

    al_config = config['data'].pop('active_learning')
    mlflow.log_params(al_config)
    acquisition_config = config['model'].pop('acquisition')
    mlflow.log_params(acquisition_config)
    mlflow.log_params(log_fe_config)
    mlflow.log_params(log_head_config)
    mlflow.log_params(config['model'])
    mlflow.log_params(config['data'])


def data_logging(data_dict):
    mlflow.log_params(data_dict)


def log_and_store_metrics(dict: Dict, step):
    mlflow.log_metrics(dict, step)
    for key in dict.keys():
        result_dataframe.loc[step, key] = dict[key]


def log_dict_results(results, mode, step=None):

    formatted_results = {}

    for key in results.keys():
        new_key = mode + '_' + key
        formatted_results[new_key] = results[key]

    log_and_store_metrics(formatted_results, step=step)


def log_artifacts():
    mlflow.log_artifacts(globals.config['logging']['experiment_folder']) # change to experiment dict


def save_results():
    path = os.path.join(globals.config['logging']['experiment_folder'], 'results.csv')
    result_dataframe.to_csv(path)


class MLFlowCallback(tf.keras.callbacks.Callback):
    """
    Object that is used in the keras training procedure to log metrics at the end of an batch/epoch while training.
    """
    def __init__(self, metric_calculator):
        super().__init__()
        self.finished_epochs = 0
        self.acquisition_steps = 0
        self.acquisition_step_metric = {}
        self.best_result = 0.0
        self.best_result_epoch = 0
        self.best_weights = None
        self.metric_calculator = metric_calculator
        self.model_converged = False

    # def on_batch_end(self, batch: int, logs=None):
    #     if batch % 100 == 0:
    #         current_step = int((self.finished_epochs * self.params['steps']) + batch)
    #         metrics_dict = format_metrics_for_mlflow(logs.copy())
    #         mlflow.log_metrics(metrics_dict, step=current_step)

    def on_epoch_end(self, epoch: int, logs=None):
        current_step = int(self.finished_epochs * (self.acquisition_steps+1))
        self.finished_epochs = self.finished_epochs + 1
        metrics_dict = format_metrics_for_mlflow(logs.copy())
        mlflow.log_metrics(metrics_dict, step=current_step)
        metrics_for_monitoring = globals.config['model']['metrics_for_monitoring']
        acc_threshold = globals.config['model']['acquisition']['after_acc_above']

        # If logging interval reached, calculate validation metrics
        if self.finished_epochs % globals.config['logging']['interval'] == 0:
            metrics_dict, _ = self.metric_calculator.calc_metrics(mode='val')
            mlflow.log_metrics(metrics_dict, step=current_step)
            mlflow.log_metric('finished_epochs', self.finished_epochs, step=current_step)
            mlflow.log_metric('acquisition_steps', self.acquisition_steps, step=current_step)
            mlflow.log_metric("lr", float(K.get_value(self.model.optimizer.lr)), step=current_step)

            # If new best result, save model and set best epoch
            if metrics_dict[metrics_for_monitoring] > self.best_result:
                self.best_result_epoch = self.finished_epochs
                self.best_result = metrics_dict[metrics_for_monitoring]
                self.best_metrics = metrics_dict
                self.best_weights = self.model.get_weights()
                if globals.config["model"]["save_model"]:
                    print("\n New best model! Saving model..")
                    self._save_model('acquisition_' + str(self.acquisition_steps))
                mlflow.log_metric("best_" + metrics_for_monitoring, metrics_dict[metrics_for_monitoring], step=current_step)
                mlflow.log_metric("best_epoch", self.finished_epochs, step=current_step)
            lr_decrease_epoch = globals.config['model']['lr_decrease_epochs']
            if lr_decrease_epoch != -1 and self.finished_epochs == lr_decrease_epoch:
                old_lr = float(K.get_value(self.model.optimizer.lr))
                new_lr = old_lr * 0.5
                K.set_value(self.model.optimizer.lr, new_lr)
                print('Reducing learning rate to: ' + str(new_lr))

            # If not, check if model has converged
            # else:
            #     patience = globals.config['model']['acquisition']['after_epochs_of_no_improvement']
            #     if patience < self.finished_epochs - self.best_result_epoch and logs['accuracy'] > acc_threshold:
            #         # self.model.set_weights(self.best_weights)
            #         # self.acquisition_step_metric[self.acquisition_steps] = metrics_dict
            #         old_lr = float(K.get_value(self.model.optimizer.lr))
            #         new_lr = old_lr * 0.5
            #         K.set_value(self.model.optimizer.lr, new_lr)
            #         print('Reducing learning rate to: ' + str(new_lr))

    def on_train_end(self, logs=None):
        self.best_result_epoch = 0
        self.best_result = 0.0
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
        self.best_weights = None

    def _save_model(self, name: str):
        save_dir = os.path.join(globals.config["output_dir"], "models/" + name)
        os.makedirs(save_dir, exist_ok=True)
        fe_path = os.path.join(save_dir, "feature_extractor.h5")
        head_path = os.path.join(save_dir, "head.h5")
        self.model.layers[0].save_weights(fe_path)
        self.model.layers[1].save_weights(head_path)


def format_metrics_for_mlflow(metrics_dict):
    """
    Transform metrics to a format suitable for mlflow.
    """
    # for now, just format f1 score which comes in as an array
    metrics_name = 'f1_score'
    if 'val_f1_score' in metrics_dict.keys():
        prefixes = ['val_', '']
    else:
        prefixes = ['']
    for prefix in prefixes:
        f1_score = metrics_dict.pop(prefix + metrics_name)
        for class_id in range(len(f1_score)):
            key = prefix + 'f1_class_id_' + str(class_id)
            metrics_dict[key] = f1_score[class_id]

        metrics_dict[prefix + 'f1_mean'] = np.mean(f1_score)

    return metrics_dict
