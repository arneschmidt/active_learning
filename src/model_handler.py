import os
import globals
import mlflow
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
from mlflow_log import MLFlowCallback, log_and_store_metrics, log_artifacts
from model_architecture import create_model
from sklearn.neighbors import LocalOutlierFactor
from utils.save_utils import save_dataframe_with_output, save_metrics_artifacts, save_acquired_images
from metrics import MetricCalculator
from typing import Dict, Optional, Tuple
from data import DataGenerator


class ModelHandler:
    """
    Class that contains the classification model. Wraps keras model.
    """
    def __init__(self, n_training_points: int):
        tf.random.set_seed(globals.config['random_seed'])
        self.n_training_points = n_training_points
        self.acquisition_step = 0
        self.uncertainty_logs = {}
        self.class_weights = {}
        config = globals.config
        self.batch_size = config["model"]["batch_size"]
        self.num_classes = config["data"]["num_classes"]
        self.model = create_model(n_training_points)
        if config["model"]["load_model"] != 'None':
            self._load_combined_model(config["model"]["load_model"])
        self.ood_estimator = None
        self._compile_model()
        self.highest_unc_indices = {}
        self.highest_unc_values = {}

        print(self.model.layers[0].summary())
        print(self.model.layers[1].summary())

    def train(self, data_gen: DataGenerator):
        """
        Train the model with the parameters specified in the config. Log progress to mlflow (see README)
        :param data_gen: data generator object to provide the image data generators and dataframes
        """
        # initialize callback for training procedure: logging and metrics calculation at the end of each epoch
        metric_calculator = MetricCalculator(self, data_gen, globals.config)
        mlflow_callback = MLFlowCallback(metric_calculator)
        callbacks = [mlflow_callback]

        # acquisition loop
        if globals.config['data']['supervision'] == 'active_learning':
            total_acquisition_steps = int(globals.config["data"]["active_learning"]["step"]["total_steps"])
        else:
            total_acquisition_steps = 1

        self.n_training_points = data_gen.get_number_of_training_points()

        for acquisition_step in range(total_acquisition_steps):

            self.acquisition_step = acquisition_step
            self.class_weights = data_gen.calculate_class_weights()
            mlflow_callback.acquisition_step = acquisition_step

            mlflow.log_metrics(data_gen.get_labeling_statistics(), self.n_training_points)
            # Optional: class-weighting based on groundtruth and estimated labels
            if globals.config["model"]["class_weighted_loss"]:
                class_weights = self.class_weights
            else:
                class_weights = []

            steps = np.ceil(data_gen.get_number_of_training_points() / self.batch_size)
            self.model.fit(
                data_gen.train_generator_labeled,
                epochs=globals.config['model']['epochs'],
                class_weight=class_weights,
                steps_per_epoch=steps,
                callbacks=callbacks,
            )
            # create_wsi_dataset
            # self.wsi_model.fit(wsi_data)
            if globals.config['model']['test_on_the_fly']:
                self.test(data_gen, step=self.n_training_points)

            if globals.config['data']['supervision'] == 'active_learning':
                mlflow.log_metrics(self.uncertainty_logs, self.n_training_points)
                selected_wsis, train_indices = self.select_data_for_labeling(data_gen)
                data_gen.query_from_oracle(selected_wsis, train_indices)
                self.n_training_points = data_gen.get_number_of_training_points()
                self.update_model(self.n_training_points)

                if globals.config['logging']['save_images']:
                    save_acquired_images(data_gen, self.highest_unc_indices, self.highest_unc_values, acquisition_step)

            if globals.config['logging']['log_artifacts']:
                log_artifacts()

    def test(self, data_gen: DataGenerator, step=None):
        """
        Test the model with the parameters specified in the config. Log progress to mlflow (see README)
        :param data_gen: data generator object to provide the image data generators and dataframes
        :return: dict of metrics from testing
        """
        metric_calculator = MetricCalculator(self.model, data_gen, globals.config)
        metrics, artifacts = metric_calculator.calc_metrics(mode='test')
        save_metrics_artifacts(artifacts, globals.config['logging']['experiment_folder'])
        log_and_store_metrics(metrics, step)

    def predict(self, data):
        """
        Save predictions of a single random batch for demonstration purposes.
        :param data_gen:  data generator object to provide the image data generators and dataframes
        """
        features = self.get_features(data)
        preds = self.get_predictions(features)

        if globals.config['model']['head']['type'] == 'bnn':
            preds = np.mean(preds, axis=0)
        return preds

    def get_features(self, ds):
        feature_extractor = self.model.layers[0]
        features = feature_extractor.predict(ds, verbose=1)
        return features

    def get_predictions(self, features):
        head = self.model.layers[1]

        if globals.config['model']['head']['type'] == 'bnn':
            num_samples = 20
            preds = []
            for i in range(num_samples):
                preds.append(head.predict(features, batch_size=globals.config['model']['batch_size']))
            preds = np.array(preds)
        else: # assume else it is deterministic
            preds = head.predict(features)

        return preds

    def select_data_for_labeling(self, data_gen: DataGenerator):
        print('Select data to be labeled..')
        unlabeled_dataframe = data_gen.train_df.loc[np.logical_not(data_gen.train_df['labeled'])]
        wsi_dataframe = data_gen.wsi_df

        wsis_per_acquisition = globals.config['data']['active_learning']['step']['wsis']
        labels_per_wsi = globals.config['data']['active_learning']['step']['labels_per_wsi']

        if not globals.config['model']['acquisition']['random']:
            features = self.get_features(data_gen.train_generator_unlabeled)
            preds = self.get_predictions(features)

            features_labeled = self.get_features(data_gen.train_generator_labeled)

            if globals.config['model']['acquisition']['focussed_epistemic']:
                self.update_ood_estimator(features_labeled)

            acquisition_scores = self.get_acquisition_scores(preds, features)
            sorted_rows = np.argsort(acquisition_scores)[::-1]

            unlabeled_wsis = np.array(wsi_dataframe['slide_id'].loc[np.logical_and(np.logical_not(wsi_dataframe['labeled']),
                                                                          wsi_dataframe['Partition'] == 'train')])
            mean_uncertainties= np.zeros_like(unlabeled_wsis)
            for i in range(len(unlabeled_wsis)):
                rows = unlabeled_dataframe['wsi'] == unlabeled_wsis[i]
                mean_uncertainties[i] = np.mean(acquisition_scores[rows])
            sorted_wsi_rows = np.argsort(mean_uncertainties)[::-1]
            selected_wsis = unlabeled_wsis[sorted_wsi_rows[0:wsis_per_acquisition]]
        else:
            unlabeled_wsis = wsi_dataframe['slide_id'].loc[np.logical_and(np.logical_not(wsi_dataframe['labeled']),
                                                                          wsi_dataframe['Partition'] == 'train')]
            selected_wsis = np.random.choice(unlabeled_wsis, size=wsis_per_acquisition, replace=False)

        # get the highest uncertainties of the selected WSIs

        if not globals.config['data']['active_learning']['step']['flexible_labeling']:
            ids = np.array([]) # reference to unlabeled dataframe
            for wsi in selected_wsis:
                wsi_rows = np.array([])
                if not globals.config['model']['acquisition']['random']:
                    for row in sorted_rows:
                        if unlabeled_dataframe['wsi'].iloc[row] == wsi:
                            wsi_rows = np.concatenate([row, wsi_rows], axis=None)
                        if wsi_rows.size >= labels_per_wsi:
                            break
                else:
                    candidates = np.squeeze(np.argwhere(np.array(unlabeled_dataframe['wsi']==wsi)))
                    wsi_rows = np.random.choice(candidates, size=labels_per_wsi, replace=False)
                wsi_ids = unlabeled_dataframe['index'].iloc[wsi_rows].values[:] # convert to train_df reference
                ids = np.concatenate([ids, wsi_ids], axis=None)
            if ids.size != wsis_per_acquisition*labels_per_wsi:
                print('Expected labels: ', wsis_per_acquisition*labels_per_wsi)
                print('Requested labels: ', ids.size)
                print('Not enough labels obtained!')
        else:
            unlabeled_ids = [] # reference to unlabeled dataframe
            for row in sorted_rows:
                if unlabeled_dataframe['wsi'].iloc[row] in selected_wsis:
                    unlabeled_ids.append(row)
                if len(unlabeled_ids) > wsis_per_acquisition*labels_per_wsi:
                    break
            ids = unlabeled_dataframe['index'].iloc[unlabeled_ids].values[:] # convert to train_df reference

        return selected_wsis, ids

    def update_model(self, num_training_points: int):
        weights = self.model.get_weights()
        new_model = create_model(num_training_points)
        if globals.config['model']['acquisition']['keep_trained_weights']:
            new_model.set_weights(weights)
        self.model = new_model
        self._compile_model()

    def update_class_weights(self, class_weights):
        self.class_weights = class_weights

    def update_ood_estimator(self, features_labeled):
        features = np.expand_dims(np.mean(features_labeled, axis=1), axis=1)

        ood_k_neighbors = globals.config['model']['acquisition']['ood_k_neighbors']
        ood_estimator = LocalOutlierFactor(n_neighbors=ood_k_neighbors, novelty=True)
        ood_estimator.fit(features)

        self.ood_estimator = ood_estimator

    def _compile_model(self):
        """
        Compile keras model.
        """
        input_shape = (self.batch_size, globals.config["data"]["image_target_size"][0],
                       globals.config["data"]["image_target_size"][1], 3)
        self.model.build(input_shape)

        if globals.config['model']['optimizer'] == 'sgd':
            optimizer = tf.optimizers.SGD(learning_rate=globals.config["model"]["learning_rate"])
        else:
            optimizer = tf.optimizers.Adam(learning_rate=globals.config["model"]["learning_rate"])

        if globals.config['model']['loss_function'] == 'focal_loss':
            loss = tfa.losses.SigmoidFocalCrossEntropy()
        else:
            loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)


        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=['accuracy',
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.Recall(),
                                    tfa.metrics.F1Score(num_classes=self.num_classes),
                                    tfa.metrics.CohenKappa(num_classes=self.num_classes, weightage='quadratic')
                                    ])


    def _calculate_most_uncertain_class(self, val_metrics: Dict):
        metrics_name = 'val_f1_class_id_'
        metrics = []
        for i in range(self.num_classes):
            class_f1 = val_metrics[metrics_name + str(i)]
            metrics.append(class_f1)
        uncertain_class_id = np.argmin(metrics)
        return uncertain_class_id


    def _load_combined_model(self, artifact_path: str = "./models/"):
        model_path = os.path.join(artifact_path, "models")
        self.model.layers[0].load_weights(os.path.join(model_path, "feature_extractor.h5"))
        self.model.layers[1].load_weights(os.path.join(model_path, "head.h5"))
        self.model.summary()

    def _save_predictions(self, image_batch: np.array, predictions: np.array, output_dir: str):
        for i in range(image_batch[0].shape[0]):
            plt.figure()
            image = image_batch[0][i]
            ground_truth = image_batch[1][i][1]
            prediction = predictions[i][1]
            plt.imshow(image.astype(int))
            plt.title("Ground Truth: " + str(ground_truth) + "    Prediction: " + str(prediction))
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, str(i) + ".png"))
            
            
    def _calc_aleatoric_unc(self, p_hat):
        uncertainty_calculation = globals.config['model']['acquisition']['uncertainty_calculation']

        if uncertainty_calculation == 'variance_based':
            unc_matrices = []
            for t in range(p_hat.shape[0]):
                mat = np.diag(p_hat[t]) - np.outer(p_hat[t], p_hat[t])
                unc_matrices.append(mat)
            aleatoric_unc_matrix = np.mean(np.array(unc_matrices), axis=0)
            aleatoric_total = np.trace(aleatoric_unc_matrix)
        elif uncertainty_calculation == 'entropy_based':
            aleatoric_total = - np.sum(np.sum(np.multiply(p_hat, np.log(p_hat)), axis=-1), axis=0)
        else:
            raise Exception('Invalid uncertainty_calulation: ' + uncertainty_calculation)

        return aleatoric_total

    def _calc_epistemic_unc(self, p_hat):
        uncertainty_calculation = globals.config['model']['acquisition']['uncertainty_calculation']

        if uncertainty_calculation == 'variance_based':
            p_bar = np.mean(p_hat, axis=0)
            unc_matrices = []
            for t in range(p_hat.shape[0]):
                mat = np.outer(p_hat[t] - p_bar, p_hat[t] - p_bar)
                unc_matrices.append(mat)
            epistemic_unc_matrix = np.mean(np.array(unc_matrices), axis=0)

            if globals.config['model']['acquisition']['focussed_epistemic']:
                c_weights = list(self.class_weights.values())
                epistemic_total = np.inner(c_weights, np.diag(epistemic_unc_matrix))
            else:
                epistemic_total = np.trace(epistemic_unc_matrix)
        elif uncertainty_calculation == 'entropy_based':
            mean = np.mean(p_hat, axis=0)
            entropy = - np.sum(np.multiply(mean, np.log(mean)), axis=0)
            epistemic_total = entropy + np.sum(np.sum(np.multiply(p_hat, np.log(p_hat)), axis=-1), axis=0)
        else:
            raise Exception('Invalid uncertainty_calulation: ' + uncertainty_calculation)
        return epistemic_total

    def _calc_deterministic_entropy(self, p):
        entropy = - np.sum(np.multiply(p, np.log(p)), axis=0)
        return entropy

    def get_aleatoric_and_epistemic_uncertainty(self, preds):
        n_predictions = preds.shape[-2]

        aleatoric_unc = []
        epistemic_unc = []

        # Iterate over datapoints to make calculation easier
        for i in range(n_predictions):
            if globals.config['model']['head']['type'] == 'bnn':
                vector_samples = preds[:, i, :]  # dimensions of vector_samples: [n_samples, n_classes]
                aleatoric_unc.append(self._calc_aleatoric_unc(vector_samples))
                epistemic_unc.append(self._calc_epistemic_unc(vector_samples))
            else:
                aleatoric_unc.append(0)
                epistemic_unc.append(self._calc_deterministic_entropy(preds[i]))


        aleatoric_unc = np.array(aleatoric_unc)
        epistemic_unc = np.array(epistemic_unc)

        self.store_highest_uncertainty_indices(aleatoric_unc, 'aleatoric_unc')
        self.store_highest_uncertainty_indices(epistemic_unc, 'epistemic_unc')

        self.uncertainty_logs['aleatoric_unc_mean'] = np.mean(aleatoric_unc)
        self.uncertainty_logs['aleatoric_unc_min'] = np.min(aleatoric_unc)
        self.uncertainty_logs['aleatoric_unc_max'] = np.max(aleatoric_unc)

        self.uncertainty_logs['epistemic_unc_mean'] = np.mean(epistemic_unc)
        self.uncertainty_logs['epistemic_unc_min'] = np.min(epistemic_unc)
        self.uncertainty_logs['epistemic_unc_max'] = np.max(epistemic_unc)

        return aleatoric_unc, epistemic_unc

    def get_ood_probabilities(self, features):
        if globals.config['model']['acquisition']['focussed_epistemic']:
            features = np.expand_dims(np.mean(features, axis=1), axis=1)
            in_distribution_prob = self.ood_estimator.score_samples(features)
            in_dist_normalized = (in_distribution_prob - np.min(in_distribution_prob))/\
                                 (np.max(in_distribution_prob) - np.min(in_distribution_prob))
            ood_score = 1 - in_dist_normalized

            self.store_highest_uncertainty_indices(ood_score, 'ood_score')

            self.uncertainty_logs['ood_score_mean'] = np.mean(ood_score)
            self.uncertainty_logs['ood_score_min'] = np.min(ood_score)
            self.uncertainty_logs['ood_score_max'] = np.max(ood_score)
        else:
            ood_score = np.zeros(shape=features.shape[0])

        return ood_score

    def get_acquisition_scores(self, preds, features):

        aleatoric_unc, epistemic_unc = self.get_aleatoric_and_epistemic_uncertainty(preds)
        ood_prob = self.get_ood_probabilities(features)

        aleatoric_factor = globals.config['model']['acquisition']['aleatoric_factor']
        ood_factor = globals.config['model']['acquisition']['ood_factor']

        acq_scores = epistemic_unc - aleatoric_factor*aleatoric_unc - ood_factor*ood_prob

        self.store_highest_uncertainty_indices(acq_scores, 'acq_scores')
        self.uncertainty_logs['acq_scores_mean'] = np.mean(acq_scores)
        self.uncertainty_logs['acq_scores_min'] = np.min(acq_scores)
        self.uncertainty_logs['acq_scores_max'] = np.max(acq_scores)

        return acq_scores

    def store_highest_uncertainty_indices(self, unc, name):
        n = 10
        sorted_ids = np.argsort(unc)[::-1]
        self.highest_unc_indices[name] = sorted_ids[0:n]
        self.highest_unc_values[name] = unc[sorted_ids[0:n]]

