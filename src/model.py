import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
from mlflow_log import MLFlowCallback, format_metrics_for_mlflow
from model_architecture import create_model
from sklearn.utils import class_weight
from utils.mil_utils import combine_pseudo_labels_with_instance_labels, get_data_generator_with_targets, \
    get_data_generator_without_targets, get_one_hot_training_targets
from utils.save_utils import save_dataframe_with_output, save_metrics_artifacts
from metrics import MetricCalculator
from typing import Dict, Optional, Tuple
from data import DataGenerator


class Model:
    """
    Class that contains the classification model. Wraps keras model.
    """
    def __init__(self, config: Dict, n_training_points: int):
        self.n_training_points = n_training_points
        self.acquisition_step = 0
        self.batch_size = config["model"]["batch_size"]
        self.num_classes = config["data"]["num_classes"]
        self.config = config
        self.model = create_model(config, self.num_classes, n_training_points)
        if config["model"]["load_model"] != 'None':
            self._load_combined_model(config["model"]["load_model"])
        self._compile_model()

        print(self.model.layers[0].summary())
        print(self.model.layers[1].summary())

    def train(self, data_gen: DataGenerator):
        """
        Train the model with the parameters specified in the config. Log progress to mlflow (see README)
        :param data_gen: data generator object to provide the image data generators and dataframes
        """
        # initialize callback for training procedure: logging and metrics calculation at the end of each epoch
        metric_calculator = MetricCalculator(self.model, data_gen, self.config)
        mlflow_callback = MLFlowCallback(self.config, metric_calculator)
        callbacks = [mlflow_callback]

        # initialize generators with weak and strong augmentation
        class_weights = None

        # acquisition loop
        if self.config['data']['supervision'] == 'active_learning':
            total_acquisition_steps = int(self.config["data"]["active_learning"]["acquisition"]["total_steps"])
        else:
            total_acquisition_steps = 1

        for acquisition_step in range(total_acquisition_steps):
            self.acquisition_step = acquisition_step
            self.n_training_points = data_gen.get_number_of_training_points()
            mlflow_callback.data_acquisition_logging(acquisition_step, data_gen.get_labeling_statistics())
            # Optional: class-weighting based on groundtruth and estimated labels
            if self.config["model"]["class_weighted_loss"]:
                class_weights = self._calculate_class_weights(data_gen.train_df)

            steps = np.ceil(data_gen.get_number_of_training_points() / self.batch_size)
            if self.config["model"]['head']['type'] == 'gp':
                self._set_kl_weight(acquisition_step)

            for i in range(5):
                try:
                    mlflow_callback.model_converged = False
                    self.model.fit(
                        data_gen.train_generator_labeled,
                        epochs=self.config['model']['epochs'],
                        class_weight=class_weights,
                        steps_per_epoch=steps,
                        callbacks=callbacks,
                    )
                    success = True
                except Exception as e:
                    print('Exception: ', e)
                    print('Failure in training, try again')
                    self.model.set_weights(mlflow_callback.best_weights)
                    mlflow_callback.finished_epochs = mlflow_callback.best_result_epoch
                    success = False
                if success:
                    break
            if not mlflow_callback.model_converged:
                print('\nModel did not converge! Stopping..')
                # break
            if self.config['data']['supervision'] == 'active_learning':
                selected_wsis, train_indices = self.select_data_for_labeling(data_gen, mlflow_callback.best_metrics)
                data_gen.query_from_oracle(selected_wsis, train_indices)

    def test(self, data_gen: DataGenerator):
        """
        Test the model with the parameters specified in the config. Log progress to mlflow (see README)
        :param data_gen: data generator object to provide the image data generators and dataframes
        :return: dict of metrics from testing
        """
        metric_calculator = MetricCalculator(self.model, data_gen, self.config)
        metrics, artifacts = metric_calculator.calc_metrics()
        save_metrics_artifacts(artifacts, self.config['output_dir'])
        return metrics

    def predict(self, data_gen: DataGenerator):
        """
        Save predictions of a single random batch for demonstration purposes.
        :param data_gen:  data generator object to provide the image data generators and dataframes
        """
        image_batch = data_gen.test_generator.next()
        predictions = self.model.predict(image_batch[0], steps=1)
        self._save_predictions(image_batch, predictions, self.config['output_dir'])

    def predict_features(self, data_gen: DataGenerator):
        """
        Save feature output of model for potential bag-level classifier. Currently not used.
        :param data_gen:  data generator object to provide the image data generators and dataframes
        """
        output_dir = self.config['output_dir']
        feature_extractor = self.model.layers[0]
        generators = {'Train': data_gen.train_generator_labeled,
                      'Val': data_gen.validation_generator,
                      'Test': data_gen.test_generator}
        dataframes = {'Train': data_gen.train_df,
                      'Val': data_gen.val_df,
                      'Test': data_gen.test_df}
        for mode in generators.keys():
            generator = generators[mode]
            train_steps = np.ceil(generator.n / generator.batch_size)

            train_features = feature_extractor.predict(generator, steps=train_steps)
            train_predictions = self.model.predict(generator, steps=train_steps)
            save_dataframe_with_output(dataframes[mode], train_predictions, train_features, output_dir, mode)

    def predict_uncertainties(self, generator, val_metrics):
        print('Predict uncertainties..')
        if self.config['model']['head']['type'] == 'gp':
            number_of_samples = 10
            feature_extractor = self.model.layers[0]
            cnn_out = feature_extractor.predict(generator, verbose=True)
            vgp = self.model.layers[1].layers[0]
            uncertainties = np.array([])
            batch_size = 128
            steps = int(np.ceil(len(cnn_out) / 128))
            for step in range(steps):
                start = step * batch_size
                stop = (step + 1) * batch_size
                if stop > len(cnn_out):
                    stop = len(cnn_out)
                if stop-start > 1:
                    pred = tf.nn.softmax(vgp(cnn_out[start:stop]).sample(number_of_samples))
                else:
                    # An error occurs when vgp takes only one input. Artificially enlarge, then squash input.
                    tf.expand_dims(tf.nn.softmax(vgp(cnn_out[start-1:stop]).sample(number_of_samples))[:, 1, :], 1)
                uncertainty = self.probabilistic_uncertainty_calculation(pred, val_metrics)
                uncertainties = np.concatenate((uncertainties, uncertainty))
            uncertainties = np.array(uncertainties)
        elif self.config['model']['head']['type'] == 'deterministic':
            predictions = self.model.predict(generator, verbose=True)
            uncertainties = self.deterministic_uncertainty_calculation(predictions)

        assert uncertainties.size == generator.n
        return uncertainties

    def probabilistic_uncertainty_calculation(self, prediction_samples, val_metrics):
        acquisition_method = self.config['data']['active_learning']['acquisition']['strategy']
        if acquisition_method == 'max_var':
            uncertainty = np.mean(np.std(prediction_samples, axis=0), axis=-1)
        elif acquisition_method == 'max_class_var':
            class_id = self._calculate_most_uncertain_class(val_metrics)
            uncertainty = np.std(prediction_samples, axis=0)[...,class_id]
        elif acquisition_method == 'entropy':
            mean = np.mean(prediction_samples, axis=0)
            uncertainty = - np.sum(np.multiply(mean, np.log(mean)), axis=1)
        elif acquisition_method == 'bald':
            mean = np.mean(prediction_samples, axis=0)
            entropy = np.sum(- np.multiply(mean, np.log(mean)), axis=1)
            uncertainty = entropy + np.mean(
                np.sum(np.multiply(prediction_samples, np.log(prediction_samples)), axis=-1), axis=0)
        elif acquisition_method == 'var_ratio':
            mean = np.mean(prediction_samples, axis=0)
            uncertainty = 1 - np.max(mean, axis=1)
        return uncertainty

    def deterministic_uncertainty_calculation(self, predictions):
        acquisition_method = self.config['data']['active_learning']['acquisition']['strategy']
        if acquisition_method == 'max_var':
            raise Exception('Acquisition method "max_var" not supported for deterministic models.')
        elif acquisition_method == 'entropy':
            uncertainty = - np.sum(np.multiply(predictions, np.log(predictions)), axis=1)
        elif acquisition_method == 'bald':
            entropy = np.sum(- np.multiply(predictions, np.log(predictions)), axis=1)
            uncertainty = entropy + np.sum(np.multiply(predictions, np.log(predictions)), axis=-1)
        elif acquisition_method == 'var_ratio':
            uncertainty = 1 - np.max(predictions, axis=1)
        return uncertainty

    def select_data_for_labeling(self, data_gen: DataGenerator, val_metrics: Dict):
        print('Select data to be labeled..')
        dataframe = data_gen.train_df.loc[np.logical_not(data_gen.train_df['labeled'])]
        wsi_dataframe = data_gen.wsi_df

        strategy = self.config['data']['active_learning']['acquisition']['strategy']
        wsi_selection = self.config['data']['active_learning']['acquisition']['wsi_selection']
        wsis_per_acquisition = self.config['data']['active_learning']['acquisition']['wsis']
        labels_per_wsi = self.config['data']['active_learning']['acquisition']['labels_per_wsi']

        if strategy != 'random' and wsi_selection != 'gradual_learning':
            uncertainties_of_unlabeled = self.predict_uncertainties(data_gen.train_generator_unlabeled, val_metrics)
            sorted_rows = np.argsort(uncertainties_of_unlabeled)[::-1]
        elif wsi_selection == 'gradual_learning':
            uncertainties_of_unlabeled = self.predict_uncertainties(data_gen.train_generator_unlabeled, val_metrics)
            mean_unc = np.mean(uncertainties_of_unlabeled)
            max_unc = np.max(uncertainties_of_unlabeled)
            factor = np.clip(self.acquisition_step / 20, a_min=0.0, a_max=1.0)
            fix_uncertainty = mean_unc + (max_unc - mean_unc) * factor
        else:
            if wsi_selection != 'random':
                raise Exception('Please set wsi_selection to random when using strategy = random')


        if wsi_selection == 'uncertainty_max':
            already_labeled_wsi = wsi_dataframe['slide_id'].loc[wsi_dataframe['labeled']]

            selected_wsis = []
            # get the WSIs with the highest uncertainties
            for row in sorted_rows:
                wsi_name = dataframe['wsi'].iloc[[row]].values[0]
                if wsi_name not in selected_wsis and not np.any(already_labeled_wsi.str.contains(wsi_name)):
                    selected_wsis.append(wsi_name)
                if len(selected_wsis) >= wsis_per_acquisition:
                    break
            selected_wsis = np.array(selected_wsis)
        elif wsi_selection == 'uncertainty_mean':
            unlabeled_wsis = np.array(wsi_dataframe['slide_id'].loc[np.logical_and(np.logical_not(wsi_dataframe['labeled']),
                                                                          wsi_dataframe['Partition'] == 'train')])
            mean_uncertainties= np.zeros_like(unlabeled_wsis)
            for i in range(len(unlabeled_wsis)):
                rows = dataframe['wsi'] == unlabeled_wsis[i]
                mean_uncertainties[i] = np.mean(uncertainties_of_unlabeled[rows])
            sorted_wsi_rows = np.argsort(mean_uncertainties)[::-1]
            selected_wsis = unlabeled_wsis[sorted_wsi_rows[0:wsis_per_acquisition]]
        elif wsi_selection == 'gradual_learning':
            unlabeled_wsis = np.array(wsi_dataframe['slide_id'].loc[np.logical_and(np.logical_not(wsi_dataframe['labeled']),
                                                                          wsi_dataframe['Partition'] == 'train')])
            mean_uncertainties= np.zeros_like(unlabeled_wsis)
            for i in range(len(unlabeled_wsis)):
                rows = dataframe['wsi'] == unlabeled_wsis[i]
                mean_uncertainties[i] = np.mean(uncertainties_of_unlabeled[rows])
            # sorted_wsi_rows = np.argsort(mean_uncertainties)
            wsi_diff = np.abs(mean_uncertainties - fix_uncertainty)
            sorted_wsi_rows = np.argsort(wsi_diff)
            selected_wsis = unlabeled_wsis[sorted_wsi_rows[0:wsis_per_acquisition]]

            patch_diff = np.abs(uncertainties_of_unlabeled - fix_uncertainty)
            sorted_rows = np.argsort(patch_diff)
        else:
            unlabeled_wsis = wsi_dataframe['slide_id'].loc[np.logical_and(np.logical_not(wsi_dataframe['labeled']),
                                                                          wsi_dataframe['Partition'] == 'train')]
            selected_wsis = np.random.choice(unlabeled_wsis, size=wsis_per_acquisition, replace=False)

        # get the highest uncertainties of the selected WSIs
        ids = np.array([])
        for wsi in selected_wsis:
            wsi_rows = np.array([])
            if strategy != 'random':
                for row in sorted_rows:
                    if dataframe['wsi'].iloc[row] == wsi:
                        wsi_rows = np.concatenate([row, wsi_rows], axis=None)
                    if wsi_rows.size >= labels_per_wsi:
                        break
            else:
                candidates = np.squeeze(np.argwhere(np.array(dataframe['wsi']==wsi)))
                wsi_rows = np.random.choice(candidates, size=labels_per_wsi, replace=False)
            wsi_ids = dataframe['index'].iloc[wsi_rows].values[:]
            ids = np.concatenate([ids, wsi_ids], axis=None)
        if ids.size != wsis_per_acquisition*labels_per_wsi:
            print('Expected labels: ', wsis_per_acquisition*labels_per_wsi)
            print('Requested labels: ', ids.size)
            print('Not enough labels obtained!')
        return selected_wsis, ids

    def _calculate_class_weights(self, train_df: pd.DataFrame):
        """
        Calculate class weights based on gt, pseudo and soft labels.
        :param training_targets: gt, pseudo and soft labels (fused)
        :return: class weight dict
        """
        labels = np.array(train_df['class'].loc[train_df['labeled']], dtype=int)
        classes = np.arange(0,self.num_classes)
        class_weights_array = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=labels)
        class_weights = {}
        for class_id in classes:
            class_weights[class_id] = class_weights_array[class_id]
        return class_weights

    def _compile_model(self):
        """
        Compile keras model.
        """
        input_shape = (self.batch_size, self.config["data"]["image_target_size"][0],
                       self.config["data"]["image_target_size"][1], 3)
        self.model.build(input_shape)

        if self.config['model']['optimizer'] == 'sgd':
            optimizer = tf.optimizers.SGD(learning_rate=self.config["model"]["learning_rate"])
        else:
            optimizer = tf.optimizers.Adam(learning_rate=self.config["model"]["learning_rate"])

        if self.config['model']['loss_function'] == 'focal_loss':
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

    def _set_kl_weight(self, acquisition_step):
        kl_weights = self.config['model']['head']['gp']['kl_weights']
        if len(kl_weights) >= acquisition_step:
            kl_weight = kl_weights[-1]
        else:
            kl_weight = kl_weights[acquisition_step]
        if self.config['model']['head']['gp']['kl_decrease']:
            weight = tf.cast(kl_weight/ self.n_training_points, tf.float32)
        else:
            weight = tf.cast(kl_weight, tf.float32)

        self.model.layers[1].variables[7].assign(weight)


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

