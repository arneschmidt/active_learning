from typing import Dict
import mlflow
import tensorflow
import os
import numpy as np

class MLFlowLogger:
    """
    Object to collect configuration parameters, metrics and artifacts and log them with mlflow.
    """
    def __init__(self, config: Dict):
        mlflow.set_tracking_uri(config["logging"]["tracking_url"])
        experiment_id = mlflow.set_experiment(experiment_name='active_' + config["data"]["dataset_name"])
        mlflow.start_run(experiment_id=experiment_id, run_name=config["logging"]["run_name"])
        self.config = config

    def config_logging(self):
        mlflow.log_params(self.config['model'])
        mlflow.log_params(self.config['model']['feature_extractor'])
        mlflow.log_params(self.config['data'])
        head_type = self.config['model']['head']['type']
        mlflow.log_param('head_type', head_type)
        mlflow.log_params(self.config['model']['head'][head_type])

    def data_logging(self, data_dict):
        mlflow.log_params(data_dict)

    def test_logging(self, metrics: Dict):
        mlflow.log_metrics(metrics)

    def log_artifacts(self):
        mlflow.log_artifacts(self.config['output_dir'])


class MLFlowCallback(tensorflow.keras.callbacks.Callback):
    """
    Object that is used in the keras training procedure to log metrics at the end of an batch/epoch while training.
    """
    def __init__(self, config, metric_calculator_val, metric_calculator_test = None):
        super().__init__()
        self.finished_epochs = 0
        self.acquisition_steps = 0
        self.acquisition_step_metric = {}
        self.best_result = 0.0
        self.best_result_epoch = 0
        self.best_weights = None
        self.config = config
        self.metric_calculator_val = metric_calculator_val
        self.metric_calculator_test = metric_calculator_test
        self.params = {}
        self.params['steps'] = 0

    def on_batch_end(self, batch: int, logs=None):
        pass
        # if batch % 100 == 0:
        #     current_step = int((self.finished_epochs * self.params['steps']) + batch)
        #     # metrics_dict = format_metrics_for_mlflow(logs.copy())
        #     metrics_dict = logs.copy()
        #     mlflow.log_metrics(metrics_dict, step=current_step)

    def on_epoch_end(self, epoch: int, logs=None):
        current_step = int(self.finished_epochs * self.params['steps'])
        self.finished_epochs = self.finished_epochs + 1
        metrics_dict = logs.copy()
        mlflow.log_metrics(metrics_dict, step=current_step)
        metrics_for_monitoring = self.config['model']['metrics_for_monitoring']

        # If logging interval rached, calculate validation metrics
        if self.finished_epochs % self.config['logging']['interval'] == 0:
            metrics_dict, _ = self.metric_calculator_val.calc_metrics(mode='val')
            mlflow.log_metrics(metrics_dict, step=current_step)
            mlflow.log_metric('finished_epochs', self.finished_epochs, step=current_step)
            mlflow.log_metric('acquisition_steps', self.acquisition_steps, step=current_step)

            # If new best result, save model and set best epoch
            if metrics_dict[metrics_for_monitoring] > self.best_result:
                self.best_result_epoch = self.finished_epochs
                self.best_result = metrics_dict[metrics_for_monitoring]
                self.best_weights = self.model.get_weights()
                if self.config["model"]["save_model"]:
                    print("\n New best model! Saving model..")
                    self._save_model('acquisition_' + str(self.acquisition_steps))
                mlflow.log_metric("best_" + metrics_for_monitoring, metrics_dict[metrics_for_monitoring])
                mlflow.log_metric("saved_model_epoch", self.finished_epochs)
            # If not, check if model has converged
            else:
                patience = self.config['data']['active_learning']['acquisition']['after_epochs_of_no_improvement']
                if patience < self.finished_epochs - self.best_result_epoch and logs['accuracy'] > 0.9:
                    self.model.set_weights(self.best_weights)
                    metrics_dict, _ = self.metric_calculator_val.calc_metrics(mode='test')
                    mlflow.log_metrics(metrics_dict, step=current_step)
                    self.model.stop_training = True
                    self.acquisition_step_metric[self.acquisition_steps] = metrics_dict
                    self.best_result = 0.0

    def data_acquisition_logging(self, acquisition_step, data_aquisition_dict):
        current_step = int(self.finished_epochs * self.params['steps'])
        mlflow.log_metrics(data_aquisition_dict, step=current_step)
        self.acquisition_steps = acquisition_step

    def _save_model(self, name: str):
        save_dir = os.path.join(self.config["output_dir"], "models/" + name)
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
