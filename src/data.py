import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.data_utils import extract_df_info, extract_wsi_df_info, get_start_label_ids
from typing import Dict, Optional, Tuple


class DataGenerator():
    """
    Object to obtain the patches and labels.
    """
    def __init__(self, config: Dict):
        """
        Initialize data generator object
        :param config: dict containing config
        """
        np.random.seed(config['data']['random_seed'])
        self.config = config
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.train_generator_labeled = None
        self.train_generator_unlabeled = None
        self.validation_generator = None
        self.test_generator = None
        self.wsi_df = None
        self._load_dataframes()
        self._initialize_data_generators()

    def get_number_of_training_points(self):
        return int(np.sum(self.train_df['labeled']))

    def query_from_oracle(self, train_indices):
        self.train_df['labeled'].loc[train_indices] = True
        self.train_generator_labeled = self.data_generator_from_dataframe(self.train_df.loc[self.train_df['labeled']])
        self.train_generator_unlabeled = self.data_generator_from_dataframe(self.train_df.loc[np.logical_not(self.train_df['labeled'])])

    def _load_dataframes(self):
        wsi_df = pd.read_csv(os.path.join(self.config['data']["dir"], "wsi_labels.csv"))
        wsi_df = extract_wsi_df_info(wsi_df)
        self.wsi_df = wsi_df
        if self.config['model']['mode'] == 'train':
            train_df_raw = pd.read_csv(os.path.join(self.config['data']["data_split_dir"], "train_patches.csv"))
            self.train_df = extract_df_info(train_df_raw, self.wsi_df, self.config['data'], split='train')
            val_df_raw = pd.read_csv(os.path.join(self.config['data']["data_split_dir"], "val_patches.csv"))
            self.val_df = extract_df_info(val_df_raw, self.wsi_df, self.config['data'], split='val')
        else:
            val_df_raw = pd.read_csv(os.path.join(self.config['data']["data_split_dir"], "val_patches.csv"))
            self.val_df = extract_df_info(val_df_raw, self.wsi_df, self.config['data'], split='val')
            test_df_raw = pd.read_csv(os.path.join(self.config['data']["data_split_dir"], "test_patches.csv"))
            self.test_df = extract_df_info(test_df_raw, self.wsi_df, self.config['data'], split='test')


    def _initialize_data_generators(self):
        # init some labeled patches for active learning
        if self.config['model']['mode'] == 'train' and self.config['data']['supervision'] == 'active_learning':
            self.train_df['labeled'] = False
            ids = get_start_label_ids(self.train_df, self.wsi_df, self.config['data'])
            self.train_df['labeled'][ids] = True
        self.train_generator_labeled = self.data_generator_from_dataframe(self.train_df.loc[self.train_df['labeled']],
                                                                          shuffle=True)
        self.train_generator_unlabeled = self.data_generator_from_dataframe(self.train_df.loc[np.logical_not(self.train_df['labeled'])])
        self.validation_generator = self.data_generator_from_dataframe(self.val_df, image_augmentation=False, shuffle=False)
        self.test_generator = self.data_generator_from_dataframe(self.test_df, image_augmentation=False, shuffle=False)


    def data_generator_from_dataframe(self, dataframe: pd.DataFrame, image_augmentation: bool = True,
                                      shuffle: bool = False, target_mode: str = 'class'):
        """
        Wrapper around 'flow_from_dataframe'-method. Uses loaded dataframes to load images and labels.

        :param dataframe: dataframe containing patch paths and labels
        :param image_augmentation: 'strong','weak' or 'None' indicating the level of augmentation
        :param shuffle: bool to shuffle the data after each epoch
        :param target_mode: 'class': loads patch classes, 'index': loads indices instead, or 'None' only loads images
        :return: data generator loading patches and labels (or indices)
        """
        if dataframe is None:
            return None

        if image_augmentation:
            datagen = ImageDataGenerator(
                brightness_range=self.config['data']['augmentation']['brightness_range'],
                channel_shift_range=self.config['data']['augmentation']["channel_shift"],
                rotation_range=360,
                fill_mode='reflect',
                horizontal_flip=True,
                vertical_flip=True)
        else:
            datagen = ImageDataGenerator()

        if target_mode == 'class':
            y_col = 'class'
            class_mode = 'categorical'
            classes = [str(i) for i in range(self.config['data']["num_classes"])]
        else:
            y_col = 'index'
            class_mode = None
            classes = None

        generator = datagen.flow_from_dataframe(
            dataframe=dataframe,
            directory=self.config['data']["dir"],
            x_col="image_path",
            y_col=y_col,
            target_size=self.config['data']["image_target_size"],
            batch_size=self.config['model']["batch_size"],
            shuffle=shuffle,
            classes=classes,
            class_mode=class_mode,
            # save_to_dir=self.config['data']['artifact_dir'] + '/' + image_augmentation,
            # save_format='jpeg'
            )

        return generator

    def get_train_data_statistics(self):
        """
        Calculate the number of labeled patches, classes and WSIs for statistics.
        :return: dict of statistics
        """
        train_df = self.train_df
        wsi_df = self.wsi_df
        wsi_names = np.unique(np.array(train_df['wsi']))
        out_dict = {}
        out_dict['number_of_wsis'] = len(wsi_names)
        out_dict['number_of_patches'] = len(train_df)
        if self.config['data']["dataset_type"] == "prostate_cancer":
            out_dict['number_of_patches_NC'] = np.sum(train_df['class'] == '0')
            out_dict['number_of_patches_GG3'] = np.sum(train_df['class'] == '1')
            out_dict['number_of_patches_GG4'] = np.sum(train_df['class'] == '2')
            out_dict['number_of_patches_GG5'] = np.sum(train_df['class'] == '3')

        return out_dict

    def get_labeling_statistics(self):
        out_dict = {}
        out_dict['labeled_wsis'] = np.sum(self.wsi_df['labeled'])
        out_dict['labeled_patches'] = np.sum(self.train_df['labeled'])
        return out_dict
