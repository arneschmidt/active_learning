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
        metric_calculator = MetricCalculator(self.model, data_gen, self.config, mode='val')
        mlflow_callback = MLFlowCallback(self.config, metric_calculator)
        patience = int(self.config['data']['active_learning']['acquisition']['labels_per_wsi'])
        callbacks = [mlflow_callback]

        # initialize generators with weak and strong augmentation
        class_weights = None

        # acquisition loop
        total_acquisition_steps = int(self.config["data"]["active_learning"]["acquisition"]["total_steps"])
        for acquisition_step in range(total_acquisition_steps):
            self.n_training_points = data_gen.get_number_of_training_points()
            mlflow_callback.data_acquisition_logging(total_acquisition_steps, data_gen.get_labeling_statistics())
            # Optional: class-weighting based on groundtruth and estimated labels
            if self.config["model"]["class_weighted_loss"]:
                class_weights = self._calculate_class_weights(data_gen.train_df)

            steps = np.ceil(self.n_training_points / self.batch_size)
            self._set_kl_weight(acquisition_step)
            weights = self.model.get_weights()
            for i in range(5):
                try:
                    self.model.fit(
                        data_gen.train_generator_labeled,
                        epochs=200,
                        class_weight=class_weights,
                        steps_per_epoch=steps,
                        callbacks=callbacks,
                    )
                    success = True
                except:
                    print('Failure in training, try again')
                    self.model.set_weights(weights)
                    success = False
                if success:
                    break

            uncertainties_of_unlabeled = self.predict_uncertainties(data_gen.train_generator_unlabeled)
            train_indices = self.select_data_for_labeling(uncertainties_of_unlabeled, data_gen)
            data_gen.query_from_oracle(train_indices)

    def test(self, data_gen: DataGenerator):
        """
        Test the model with the parameters specified in the config. Log progress to mlflow (see README)
        :param data_gen: data generator object to provide the image data generators and dataframes
        :return: dict of metrics from testing
        """
        metric_calculator = MetricCalculator(self.model, data_gen, self.config, mode='test')
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

    def predict_uncertainties(self, generator):
        print('Predict uncertainties..')
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
            pred = tf.nn.softmax(vgp(cnn_out[start:stop]).sample(number_of_samples))
            uncertainty = self.uncertainty_calculation(pred)
            uncertainties = np.concatenate((uncertainties, uncertainty))
        uncertainties = np.array(uncertainties)
        assert uncertainties.size == generator.n
        return uncertainties

    def uncertainty_calculation(self, prediction_samples):
        acquisition_method = self.config['data']['active_learning']['acquisition']['strategy']
        if acquisition_method == 'max_var':
            uncertainty = np.mean(np.std(prediction_samples, axis=0), axis=-1)
        elif acquisition_method == 'entropy':
            mean = np.mean(prediction_samples, axis=0)
            uncertainty = np.sum(- np.multiply(mean, np.log(mean)), axis=1)
        elif acquisition_method == 'bald':
            mean = np.mean(prediction_samples, axis=0)
            entropy = np.sum(- np.multiply(mean, np.log(mean)), axis=1)
            uncertainty = entropy + np.mean(
                np.sum(np.multiply(prediction_samples, np.log(prediction_samples)), axis=-1), axis=0)
        return uncertainty

    def select_data_for_labeling(self, uncertainties: np.array, data_gen: DataGenerator):
        print('Select data to be labeled..')
        dataframe = data_gen.train_df.loc[np.logical_not(data_gen.train_df['labeled'])]
        wsi_dataframe = data_gen.wsi_df

        wsis_per_acquisition = self.config['data']['active_learning']['acquisition']['wsis']
        labels_per_wsi = self.config['data']['active_learning']['acquisition']['labels_per_wsi']
        sorted_rows = np.argsort(uncertainties)[::-1]
        selected_wsis = []
        already_labeled_wsi = wsi_dataframe['slide_id'].loc[wsi_dataframe['labeled']]
        # get the WSIs with the highest uncertainties
        for row in sorted_rows:
            wsi_name = dataframe['wsi'].iloc[[row]].values[0]
            if wsi_name not in selected_wsis and not np.any(already_labeled_wsi.str.contains(wsi_name)):
                selected_wsis.append(wsi_name)
            if len(selected_wsis) >= wsis_per_acquisition:
                break
        selected_wsis = np.array(selected_wsis)

        # get the highest uncertainties of the selected WSIs
        ids = np.array([])
        for wsi in selected_wsis:
            wsi_rows = np.array([])
            for row in sorted_rows:
                if dataframe['wsi'].iloc[row] == wsi:
                    wsi_rows = np.concatenate([row, wsi_rows], axis=None)
                if wsi_rows.size >= labels_per_wsi:
                    break
            wsi_ids = dataframe['index'].iloc[wsi_rows].values[:]
            ids = np.concatenate([ids, wsi_ids], axis=None)
        if ids.size != wsis_per_acquisition*labels_per_wsi:
            print('Expected labels: ', wsis_per_acquisition*labels_per_wsi)
            print('Requested labels: ', ids.size)
            raise Warning('Not enough labels obtained!')
        return ids

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
            loss = 'categorical_crossentropy'


        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=['accuracy',
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.Recall(),
                                    # tfa.metrics.F1Score(num_classes=self.num_classes),
                                    # tfa.metrics.CohenKappa(num_classes=self.num_classes, weightage='quadratic')
                                    ])

    def _set_kl_weight(self, acquisition_step):
        kl_weights = self.config['model']['head']['gp']['kl_weights']
        if len(kl_weights) <= acquisition_step:
            kl_weight = kl_weights[-1]
        else:
            kl_weight = kl_weights[acquisition_step]
        weight = tf.cast(kl_weight * self.batch_size / self.n_training_points, tf.float32)

        self.model.layers[1].variables[7].assign(weight)


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

