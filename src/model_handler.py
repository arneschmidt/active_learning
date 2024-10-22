import os
import globals
import mlflow
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
from mlflow_log import MLFlowCallback, log_and_store_metrics, log_artifacts
from model_architecture import create_model, create_wsi_level_model
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
        self.highest_uncertainty_dfs = {}
        self.class_weights = {}
        config = globals.config
        self.batch_size = config["model"]["batch_size"]
        self.num_classes = config["data"]["num_classes"]
        self.patch_model = create_model(n_training_points)
        self.wsi_model = create_wsi_level_model(globals.config['data']['active_learning']['start']['wsis_per_class'])
        if config["model"]["load_model"] != 'None':
            self._load_combined_model(config["model"]["load_model"])
        self.ood_estimator = None
        self._compile_models()

        print(self.patch_model.layers[0].summary())
        print(self.patch_model.layers[1].summary())

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
            print('### Train patch model ### ')
            self.patch_model.fit(
                data_gen.train_generator_labeled,
                epochs=globals.config['model']['epochs'],
                class_weight=class_weights,
                steps_per_epoch=steps,
                callbacks=callbacks,
            )
            print('### Make feature predictions ### ')
            train_feat, val_feat, test_feat = self.make_feature_predictions(data_gen)
            if globals.config['model']['wsi_level_model']['use']:
                self.train_wsi_level_model(data_gen, train_feat, val_feat, test_feat)
            if globals.config["model"]["save_model"]:
                print("\n Saving model..")
                self._save_models(acquisition_step)
            if globals.config['model']['test_on_the_fly']:
                self.test(data_gen, step=self.n_training_points)
            if globals.config['data']['supervision'] == 'active_learning':
                mlflow.log_metrics(self.uncertainty_logs, self.n_training_points)
                selected_wsis, train_indices = self.select_data_for_labeling(data_gen, train_feat)
                data_gen.query_from_oracle(selected_wsis, train_indices)
                if len(globals.config['logging']['test_pred_wsis']) > 0:
                    self.save_test_predictions(data_gen, test_feat, acquisition_step)
                self.n_training_points = data_gen.get_number_of_training_points()
                self.update_model(self.n_training_points, int(np.sum(data_gen.wsi_df['labeled'])))

                if globals.config['logging']['save_images']:
                    save_acquired_images(data_gen, train_indices, self.highest_uncertainty_dfs, acquisition_step)

            if globals.config['logging']['log_artifacts']:
                log_artifacts()

    def train_wsi_level_model(self, data_gen, train_feat, val_feat, test_feat):
        print('### Create WSI level data ### ')
        if not globals.config['model']['wsi_level_model']['access_to_all_wsis']:
            train_feat = train_feat[np.logical_not(data_gen.train_df['available_for_query'])]
        data_gen.create_wsi_level_dataset(train_feat, val_feat, test_feat)
        print('### Train WSI level model ### ')
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_cohen_kappa',patience=1000, mode='max',
                                                    restore_best_weights=True) #only take the best weights
        self.wsi_model.fit(data_gen.train_feat_gen,
                           validation_data=data_gen.val_feat_gen,
                           epochs=300,
                           callbacks=[callback])

    def test(self, data_gen: DataGenerator, step=None):
        """
        Test the model with the parameters specified in the config. Log progress to mlflow (see README)
        :param data_gen: data generator object to provide the image data generators and dataframes
        :return: dict of metrics from testing
        """
        metric_calculator = MetricCalculator(self, data_gen, globals.config)
        metrics, artifacts = metric_calculator.calc_metrics(mode='test')
        save_metrics_artifacts(artifacts, globals.config['logging']['experiment_folder'])
        log_and_store_metrics(metrics, step, to_save_dataframe=True)

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
        feature_extractor = self.patch_model.layers[0]
        features = feature_extractor.predict(ds, verbose=1)
        return features

    def get_predictions(self, features):
        head = self.patch_model.layers[1]

        if globals.config['model']['head']['type'] == 'bnn':
            num_samples = 20
            preds = []
            for i in range(num_samples):
                preds.append(head.predict(features, batch_size=globals.config['model']['batch_size']))
            preds = np.array(preds)
        else: # assume else it is deterministic
            preds = head.predict(features)

        return preds

    def make_feature_predictions(self, data_gen: DataGenerator):
        train_instances_ordered = data_gen.data_generator_from_dataframe(data_gen.train_df,
                                                                        image_augmentation=False,
                                                                        shuffle=False)
        train_feat = self.get_features(train_instances_ordered)
        val_feat = self.get_features(data_gen.validation_generator)
        test_feat = self.get_features(data_gen.test_generator)
        return train_feat, val_feat, test_feat

    def select_data_for_labeling(self, data_gen: DataGenerator, train_features):
        print('Select data to be labeled..')
        unlabeled_dataframe = data_gen.train_df.loc[data_gen.train_df['available_for_query']]
        wsi_dataframe = data_gen.wsi_df

        wsis_per_acquisition = globals.config['data']['active_learning']['step']['wsis']
        wsi_independent_labeling = globals.config['data']['active_learning']['step']['wsi_independent_labeling']
        if globals.config['data']['active_learning']['step']['acceleration']['use']:
            if self.acquisition_step > globals.config['data']['active_learning']['step']['acceleration']['after_step']:
                wsis_per_acquisition = globals.config['data']['active_learning']['step']['acceleration']['wsis']

        labels_per_wsi = globals.config['data']['active_learning']['step']['labels_per_wsi']

        # features_unlabeled = self.get_features(data_gen.train_generator_unlabeled)
        features_unlabeled = train_features[data_gen.train_df['available_for_query']]
        features_labeled = train_features[data_gen.train_df['labeled']]
        if globals.config['model']['acquisition']['focal']['focussed_epistemic']:
            self.update_ood_estimator(features_labeled)

        acquisition_scores, epistemic_unc, aleatoric_unc, ood_unc = self.get_acquisition_scores(features_unlabeled)

        sorted_rows = np.argsort(acquisition_scores)[::-1]
        if not wsi_independent_labeling:
            unlabeled_wsis = np.array(wsi_dataframe['slide_id'].loc[np.logical_and(np.logical_not(wsi_dataframe['labeled']),
                                                                          wsi_dataframe['Partition'] == 'train')])
            wsi_acq_scores= np.zeros_like(unlabeled_wsis)
            for i in range(len(unlabeled_wsis)):
                rows = unlabeled_dataframe['wsi'] == unlabeled_wsis[i]
                if globals.config['model']['acquisition']['wsi_selection'] == 'uncertainty_mean':
                    wsi_acq_scores[i] = np.mean(acquisition_scores[rows])
                else:
                    wsi_acq_scores[i] = np.max(acquisition_scores[rows])
            sorted_wsi_rows = np.argsort(wsi_acq_scores)[::-1]
            selected_wsis = unlabeled_wsis[sorted_wsi_rows[0:wsis_per_acquisition]]
        else:
            selected_wsis = []

        # get the highest uncertainties of the selected WSIs
        if not globals.config['data']['active_learning']['step']['flexible_labeling']:
            unlabeled_ids = np.array([]) # reference to unlabeled dataframe
            for wsi in selected_wsis:
                wsi_rows = np.array([])
                if not globals.config['model']['acquisition']['strategy'] == 'random':
                    for row in sorted_rows:
                        if unlabeled_dataframe['wsi'].iloc[row] == wsi:
                            wsi_rows = np.concatenate([row, wsi_rows], axis=None)
                        if wsi_rows.size >= labels_per_wsi:
                            break
                else:
                    candidates = np.squeeze(np.argwhere(np.array(unlabeled_dataframe['wsi']==wsi)))
                    wsi_rows = np.random.choice(candidates, size=labels_per_wsi, replace=False)
                unlabeled_ids = np.concatenate([unlabeled_ids, wsi_rows], axis=None)
            if unlabeled_ids.size != wsis_per_acquisition*labels_per_wsi:
                print('Expected labels: ', wsis_per_acquisition*labels_per_wsi)
                print('Requested labels: ', unlabeled_ids.size)
                print('Not enough labels obtained!')
        else:
            unlabeled_ids = [] # reference to unlabeled dataframe
            for row in sorted_rows:
                if unlabeled_dataframe['wsi'].iloc[row] in selected_wsis or wsi_independent_labeling:
                    unlabeled_ids.append(row)
                if len(unlabeled_ids) >= wsis_per_acquisition*labels_per_wsi:
                    break
        ids = unlabeled_dataframe['index'].iloc[unlabeled_ids].values[:] # convert to train_df reference
        if not globals.config['model']['acquisition']['strategy'] == 'random':
            self.store_dataframes_for_logging(data_gen, unlabeled_ids, acquisition_scores, epistemic_unc, aleatoric_unc, ood_unc)
        return selected_wsis, ids

    def update_model(self, num_training_points: int, num_labeled_wsi: int):
        patch_model_weights = self.patch_model.get_weights()

        new_model = create_model(num_training_points)
        if globals.config['model']['acquisition']['keep_trained_weights']:
            new_model.set_weights(patch_model_weights)
        self.patch_model = new_model
        if globals.config['model']['wsi_level_model']['use']:
            wsi_model_weights = self.wsi_model.get_weights()
            new_wsi_model = create_wsi_level_model(num_labeled_wsi)
            if globals.config['model']['acquisition']['keep_trained_weights']:
                new_wsi_model.set_weights(wsi_model_weights)
            self.wsi_model = new_wsi_model
        self._compile_models()

    def update_class_weights(self, class_weights):
        self.class_weights = class_weights

    def update_ood_estimator(self, features_labeled):
        features = features_labeled
        ood_k_neighbors = globals.config['model']['acquisition']['focal']['ood_k_neighbors']
        ood_estimator = LocalOutlierFactor(n_neighbors=ood_k_neighbors, novelty=True)
        ood_estimator.fit(features)

        self.ood_estimator = ood_estimator

    def _compile_models(self):
        """
        Compile keras model.
        """
        input_shape = (self.batch_size, globals.config["data"]["image_target_size"][0],
                       globals.config["data"]["image_target_size"][1], 3)
        self.patch_model.build(input_shape)

        if globals.config['model']['optimizer'] == 'sgd':
            optimizer = tf.optimizers.SGD(learning_rate=globals.config["model"]["learning_rate"])
        else:
            optimizer = tf.optimizers.Adam(learning_rate=globals.config["model"]["learning_rate"])

        if globals.config['model']['loss_function'] == 'focal_loss':
            loss = tfa.losses.SigmoidFocalCrossEntropy()
        else:
            loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

        self.patch_model.compile(optimizer=optimizer,
                                 loss=loss,
                                 metrics=['accuracy',
                                    tfa.metrics.F1Score(num_classes=self.num_classes),
                                    tfa.metrics.CohenKappa(num_classes=self.num_classes, weightage='quadratic')
                                    ])
        if globals.config['model']['wsi_level_model']['use']:
            wsi_classes = 6
            self.wsi_model.build(input_shape)
            self.wsi_model.compile(optimizer=tf.optimizers.Adam(learning_rate=globals.config['model']['wsi_level_model']['learning_rate']),
                               loss=tf.keras.losses.CategoricalCrossentropy(),
                               metrics=['accuracy',
                                        tfa.metrics.F1Score(num_classes=wsi_classes),
                                        tfa.metrics.CohenKappa(num_classes=wsi_classes, weightage='quadratic')
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
        self.patch_model.layers[0].load_weights(os.path.join(model_path, "feature_extractor.h5"))
        self.patch_model.layers[1].load_weights(os.path.join(model_path, "head.h5"))
        self.patch_model.summary()

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

    def _calc_deterministic_entropy(self, p):
        entropy = - np.sum(np.multiply(p, np.log(p+0.00001)), axis=0)
        return entropy

    def _calc_probabilistic_entropy(self, p_hat):
        mean = np.mean(p_hat, axis=0)
        entropy = - np.sum(np.multiply(mean, np.log(mean+0.00001)), axis=0)
        return entropy
    def _calc_aleatoric_unc(self, preds):
        uncertainty_calculation = globals.config['model']['acquisition']['focal']['uncertainty_calculation']
        aleatoric_unc = []
        for i in range(preds.shape[1]):
            p_hat = preds[:, i, :]
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
            aleatoric_unc.append(aleatoric_total)
        aleatoric_unc = np.array(aleatoric_unc)
        return aleatoric_unc

    def _calc_epistemic_unc(self, preds, type='variance_based', focused=False):
        epistemic_unc = []
        for i in range(preds.shape[1]):
            p_hat = preds[:, i, :]
            if type == 'variance_based':
                p_bar = np.mean(p_hat, axis=0)
                unc_matrices = []
                for t in range(p_hat.shape[0]):
                    mat = np.outer(p_hat[t] - p_bar, p_hat[t] - p_bar)
                    unc_matrices.append(mat)
                epistemic_unc_matrix = np.mean(np.array(unc_matrices), axis=0)

                if focused:
                    c_weights = list(self.class_weights.values())
                    epistemic_total = np.inner(c_weights, np.diag(epistemic_unc_matrix))
                else:
                    epistemic_total = np.trace(epistemic_unc_matrix)
            elif type == 'entropy_based':
                entropy = self._calc_probabilistic_entropy(p_hat)
                epistemic_total = entropy + np.mean(np.sum(np.multiply(p_hat, np.log(p_hat + 0.00001)), axis=-1), axis=0)
            else:
                raise Exception('Invalid uncertainty_calulation: ' + type)
            epistemic_unc.append(epistemic_total)
        epistemic_unc = np.array(epistemic_unc)
        return epistemic_unc

    def get_aleatoric_and_epistemic_uncertainty(self, features):
        acquisition_strategy = globals.config['model']['acquisition']['strategy']
        n_predictions = features.shape[0]
        p_hat = self.get_predictions(features)

        if acquisition_strategy == 'focal':
            # Iterate over datapoints to make calculation easier
            aleatoric_unc = self._calc_aleatoric_unc(p_hat)
            epistemic_unc = self._calc_epistemic_unc(p_hat,
                                                     globals.config['model']['acquisition']['focal'][
                                                         'uncertainty_calculation'],
                                                     globals.config['model']['acquisition']['focal'][
                                                         'focussed_epistemic'])
        else:
            aleatoric_unc = np.zeros(n_predictions)
            epistemic_unc = np.zeros(n_predictions)

        self.uncertainty_logs['aleatoric_unc_mean'] = np.mean(aleatoric_unc)
        self.uncertainty_logs['aleatoric_unc_min'] = np.min(aleatoric_unc)
        self.uncertainty_logs['aleatoric_unc_max'] = np.max(aleatoric_unc)

        self.uncertainty_logs['epistemic_unc_mean'] = np.mean(epistemic_unc)
        self.uncertainty_logs['epistemic_unc_min'] = np.min(epistemic_unc)
        self.uncertainty_logs['epistemic_unc_max'] = np.max(epistemic_unc)

        return aleatoric_unc, epistemic_unc

    def get_ood_probabilities(self, features):
        if globals.config['model']['head']['type'] == 'bnn':
            lof = -self.ood_estimator.score_samples(features) # score_samples returns negative lof, therefore the minus
            ood_score = 0.1 * lof # scaling
        else:
            ood_score = np.zeros(shape=features.shape[0])
        self.uncertainty_logs['ood_score_mean'] = np.mean(ood_score)
        self.uncertainty_logs['ood_score_min'] = np.min(ood_score)
        self.uncertainty_logs['ood_score_max'] = np.max(ood_score)

        return ood_score

    def get_complete_entropy(self, preds, mode='probabilistic'):
        entropy = []
        for i in range(preds.shape[-2]):
            if mode=='probabilistic':
                p_hat = preds[:,i,:]
                entropy.append(self._calc_probabilistic_entropy(p_hat))
            else:
                p_hat = preds[i,:]
                entropy.append(self._calc_deterministic_entropy(p_hat))

        entropy = np.array(entropy)
        return entropy

    def get_acquisition_scores(self, features):
        # focal, bald, epistemic, entropy, max_std, random
        acquisition_strategy = globals.config['model']['acquisition']['strategy']

        acq_scores = np.zeros(features.shape[0])
        epistemic_unc = np.zeros(features.shape[0])
        aleatoric_unc = np.zeros(features.shape[0])
        ood_prob = np.zeros(features.shape[0])

        if acquisition_strategy == 'focal':
            aleatoric_unc, epistemic_unc = self.get_aleatoric_and_epistemic_uncertainty(features)
            ood_prob = self.get_ood_probabilities(features)

            aleatoric_factor = globals.config['model']['acquisition']['focal']['aleatoric_factor']
            ood_factor = globals.config['model']['acquisition']['focal']['ood_factor']

            acq_scores = epistemic_unc - aleatoric_factor*aleatoric_unc - ood_factor*ood_prob
        elif acquisition_strategy == 'bald':
            preds = self.get_predictions(features)
            acq_scores = self._calc_epistemic_unc(preds, type='entropy_based', focused=False)
        elif acquisition_strategy == 'epistemic':
            preds = self.get_predictions(features)
            acq_scores = self._calc_epistemic_unc(preds, type='variance_based', focused=False)
        elif acquisition_strategy == 'max_std':
            preds = self.get_predictions(features)
            acq_scores = np.mean(np.std(preds, axis=0), axis=-1)
        elif acquisition_strategy == 'entropy':
            preds = self.get_predictions(features)
            acq_scores = self.get_complete_entropy(preds, mode='probabilistic')
        elif acquisition_strategy == 'vr':
            preds = self.get_predictions(features)
            acq_scores = 1 - np.max(np.mean(preds, axis=0), axis=1)
        elif acquisition_strategy == 'det_entropy':
            preds = self.get_predictions(features)
            acq_scores = self.get_complete_entropy(preds, mode='deterministic')
        elif acquisition_strategy == 'random':
            acq_scores = np.random.uniform(0.0, 1.0, features.shape[0])

        self.uncertainty_logs['acq_scores_mean'] = np.mean(acq_scores)
        self.uncertainty_logs['acq_scores_min'] = np.min(acq_scores)
        self.uncertainty_logs['acq_scores_max'] = np.max(acq_scores)

        return acq_scores, epistemic_unc, aleatoric_unc, ood_prob
    def store_dataframes_for_logging(self, data_gen, acquisition_ids, acq_scores, epistemic_unc, aleatoric_unc, ood_unc):

        unc_names = ['acq_scores', 'epistemic_unc', 'aleatoric_unc', 'ood_unc']
        uncs = [acq_scores, epistemic_unc, aleatoric_unc, ood_unc]
        unlabeled_dataframe = data_gen.train_df.loc[data_gen.train_df['available_for_query']]

        # add uncertainties to dataframe
        for i in range(len(unc_names)):
            unlabeled_dataframe[unc_names[i]] = uncs[i]

        # store dataframes of acquisition and all uncertainties
        self.highest_uncertainty_dfs['acquisition'] = unlabeled_dataframe.iloc[acquisition_ids]
        for i in range(len(unc_names)):
            unc_name = unc_names[i]
            unc = uncs[i]
            n = 20
            sorted_ids = np.argsort(unc)[::-1][:n]
            self.highest_uncertainty_dfs[unc_name] = unlabeled_dataframe.iloc[sorted_ids]

    def _save_models(self, acquisition_step):
        save_dir = os.path.join(globals.config['logging']['experiment_folder'], str(acquisition_step))
        os.makedirs(save_dir, exist_ok=True)
        save_dir = os.path.join(save_dir, "models/")
        os.makedirs(save_dir, exist_ok=True)
        print('Save models in: ' + save_dir)
        os.makedirs(save_dir, exist_ok=True)
        fe_path = os.path.join(save_dir, "feature_extractor.h5")
        patch_head_path = os.path.join(save_dir, "patch_head.h5")
        self.patch_model.layers[0].save_weights(fe_path)
        self.patch_model.layers[1].save_weights(patch_head_path)
        if self.wsi_model is not None:
            wsi_head_path = os.path.join(save_dir, "wsi_head.h5")
            self.wsi_model.save_weights(wsi_head_path)

    def save_test_predictions(self, data_gen, test_feat, acquisition_step):
        selected_wsis = globals.config['logging']['test_pred_wsis']
        selected_df = data_gen.test_df[data_gen.test_df['wsi'].isin(selected_wsis)].copy()
        selected_features = test_feat[selected_df.index]
        preds = self.get_predictions(selected_features)
        if globals.config['model']['head']['type'] == 'bnn':
           preds = np.mean(preds, axis=0)
        class_preds = np.argmax(preds, axis=-1)
        acq_score, epistemic_unc, aleatoric_unc, ood_prob = self.get_acquisition_scores(selected_features)

        selected_df['prediction'] = class_preds
        selected_df['acq_score'] = acq_score
        selected_df['epistemic_unc'] = epistemic_unc
        selected_df['aleatoric_unc'] = aleatoric_unc
        selected_df['ood_prob'] = ood_prob

        out_dir = globals.config['logging']['experiment_folder']
        out_dir = os.path.join(out_dir, str(acquisition_step))
        os.makedirs(out_dir, exist_ok=True)

        selected_df.to_csv(os.path.join(out_dir, 'test_predictions.csv'))






