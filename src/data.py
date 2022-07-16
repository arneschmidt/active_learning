import os
import globals
import mxnet as mx
import pandas as pd
import numpy as np
import tensorflow as tf
from skimage.filters import gaussian
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.data_utils import extract_df_info, extract_wsi_df_info, get_start_label_ids
from utils.feature_data_generator import FeatureDataGenerator



class DataGenerator():
    """
    Object to obtain the patches and labels.
    """
    def __init__(self):
        """
        Initialize data generator object
        :param config: dict containing config
        """
        np.random.seed(globals.config['random_seed'])
        self.num_classes = globals.config["data"]["num_classes"]
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.train_generator_labeled = None
        self.train_generator_unlabeled = None
        self.validation_generator = None
        self.test_generator = None
        self.wsi_df = None
        self.train_feat_gen = None
        self.val_feat_gen = None
        self.test_feat_gen = None
        self._load_dataframes()
        self._initialize_data_generators()

    def get_number_of_training_points(self):
        return int(np.sum(self.train_df['labeled']))

    def query_from_oracle(self, selected_wsis, train_indices):
        self.wsi_df['labeled'].loc[self.wsi_df['slide_id'].isin(selected_wsis)] = True
        for wsi in selected_wsis:
            self.train_df['available_for_query'].loc[self.train_df['wsi'] == wsi] = False

        self.train_df['labeled'].loc[train_indices] = True
        self.train_generator_labeled = self.data_generator_from_dataframe(self.train_df.loc[self.train_df['labeled']], shuffle=True)
        self.train_generator_unlabeled = self.data_generator_from_dataframe(self.train_df.loc[self.train_df['available_for_query']],
                                                                            image_augmentation=False)

    def _load_dataframes(self):
        wsi_df = pd.read_csv(os.path.join(globals.config['data']["data_split_dir"], "wsi_labels.csv"))
        wsi_df = extract_wsi_df_info(wsi_df)
        self.wsi_df = wsi_df
        train_df_raw = pd.read_csv(os.path.join(globals.config['data']["data_split_dir"], "train_patches.csv"))
        self.train_df = extract_df_info(train_df_raw, self.wsi_df, globals.config['data'], split='train')
        val_df_raw = pd.read_csv(os.path.join(globals.config['data']["data_split_dir"], "val_patches.csv"))
        self.val_df = extract_df_info(val_df_raw, self.wsi_df, globals.config['data'], split='val')
        if globals.config['logging']['test_on_the_fly']:
            test_df_raw = pd.read_csv(os.path.join(globals.config['data']["data_split_dir"], "test_patches.csv"))
            self.test_df = extract_df_info(test_df_raw, self.wsi_df, globals.config['data'], split='test')


    def _initialize_data_generators(self):
        # init some labeled patches for active learning
        if globals.config['data']['supervision'] == 'active_learning':
            self.train_df['labeled'] = False
            self.train_df['available_for_query'] = True
            ids, selected_wsis = get_start_label_ids(self.train_df, self.wsi_df, globals.config['data'])
            for wsi in selected_wsis:
                self.train_df['available_for_query'].loc[self.train_df['wsi'] == wsi] = False
            self.train_df['labeled'].loc[ids] = True
            self.train_generator_unlabeled = self.data_generator_from_dataframe(
                self.train_df.loc[self.train_df['available_for_query']], image_augmentation=False)
        else:
            self.train_df['labeled'] = True
            self.train_df['available_for_query'] = False

        self.train_generator_labeled = self.data_generator_from_dataframe(self.train_df.loc[self.train_df['labeled']],
                                                                          shuffle=True)
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

        def hue_jitter(img):
            if globals.config['data']['augmentation']["hue"] > 0.0:
                aug = mx.image.HueJitterAug(hue=globals.config['data']['augmentation']["hue"])
                img = aug(img)
            return img

        def saturation_jitter(img):
            if globals.config['data']['augmentation']["saturation"] > 0.0:
                aug = mx.image.SaturationJitterAug(saturation=globals.config['data']['augmentation']["saturation"])
                img = aug(img)
            return img

        def contrast_jitter(img):
            if globals.config['data']['augmentation']["contrast"] > 0.0:
                aug = mx.image.ContrastJitterAug(contrast=globals.config['data']['augmentation']["contrast"])
                img = aug(img)
            return img

        def brightness_jitter(img):
            if globals.config['data']['augmentation']["contrast"] > 0.0:
                aug = mx.image.BrightnessJitterAug(brightness=globals.config['data']['augmentation']["brightness"])
                img = aug(img)
            return img

        def gaussian_blurr(img):
            if globals.config['data']['augmentation']["blur"] > 0.0:
                sigma = np.random.uniform(0, globals.config['data']['augmentation']["blur"], 1)
                img = gaussian(img, sigma=sigma[0], multichannel=True)
            return img

        def custom_augmentation(img):
            img = mx.nd.array(img)
            img = hue_jitter(img)
            img = saturation_jitter(img)
            img = contrast_jitter(img)
            img = brightness_jitter(img)
            img = img.asnumpy()
            img = gaussian_blurr(img)
            return img

        if image_augmentation:
            datagen = ImageDataGenerator(
                width_shift_range=globals.config['data']['augmentation']["width_shift_range"],
                height_shift_range=globals.config['data']['augmentation']["height_shift_range"],
                channel_shift_range=globals.config['data']['augmentation']["channel_shift_range"],
                zoom_range=globals.config['data']['augmentation']["zoom_range"],
                rotation_range=360,
                fill_mode='constant',
                horizontal_flip=True,
                vertical_flip=True,
                preprocessing_function=custom_augmentation)
        else:
            datagen = ImageDataGenerator()

        if target_mode == 'class':
            y_col = 'class'
            class_mode = 'categorical'
            classes = [str(i) for i in range(globals.config['data']["num_classes"])]
        else:
            y_col = 'index'
            class_mode = None
            classes = None

        generator = datagen.flow_from_dataframe(
            dataframe=dataframe,
            directory=globals.config['data']["dir"],
            x_col="image_path",
            y_col=y_col,
            target_size=globals.config['data']["image_target_size"],
            batch_size=globals.config['model']["batch_size"],
            shuffle=shuffle,
            classes=classes,
            class_mode=class_mode,
            # save_to_dir=globals.config['data']['artifact_dir'] + "image_augmentation",
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
        out_dict['number_of_wsis_train'] = len(wsi_names)
        out_dict['number_of_patches_train'] = len(train_df)
        if globals.config['data']["dataset_type"] == "prostate_cancer":
            out_dict['number_of_patches_NC_train'] = np.sum(train_df['class'] == '0')
            out_dict['number_of_patches_GG3_train'] = np.sum(train_df['class'] == '1')
            out_dict['number_of_patches_GG4_train'] = np.sum(train_df['class'] == '2')
            out_dict['number_of_patches_GG5_train'] = np.sum(train_df['class'] == '3')

        out_dict['number_of_wsis_val'] = len(np.unique(np.array(self.val_df['wsi'])))
        out_dict['number_of_patches_val'] = len(self.val_df)
        out_dict['number_of_wsis_test'] = len(np.unique(np.array(self.test_df['wsi'])))
        out_dict['number_of_patches_test'] = len(self.test_df)

        return out_dict

    def get_labeling_statistics(self):
        out_dict = {}
        out_dict['labeled_wsis'] = np.sum(self.wsi_df['labeled'])
        labeled_df = self.train_df[self.train_df['labeled']]
        out_dict['labeled_patches'] = len(labeled_df)
        out_dict['labeled_patches_NC'] = np.sum(labeled_df['class'] == '0')
        out_dict['labeled_patches_GG3'] = np.sum(labeled_df['class'] == '1')
        out_dict['labeled_patches_GG4'] = np.sum(labeled_df['class'] == '2')
        out_dict['labeled_patches_GG5'] = np.sum(labeled_df['class'] == '3')
        return out_dict
    
    def calculate_class_weights(self):
        """
        Calculate class weights based on gt, pseudo and soft labels.
        :param training_targets: gt, pseudo and soft labels (fused)
        :return: class weight dict
        """
        labels = np.array(self.train_df['class'].loc[self.train_df['labeled']], dtype=int)
        classes = np.arange(0,self.num_classes)
        class_weights_array = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=labels)
        class_weights = {}
        for class_id in classes:
            class_weights[class_id] = class_weights_array[class_id]
        return class_weights

    def create_wsi_level_dataset(self, train_feat, val_feat, test_feat):
        self.train_feat_gen = self.create_wsi_level_split_gen('train', train_feat, self.train_df.loc[np.logical_not(self.train_df['available_for_query'])])
        self.val_feat_gen = self.create_wsi_level_split_gen('val', val_feat, self.val_df)
        self.test_feat_gen = self.create_wsi_level_split_gen('test', test_feat, self.test_df)

    def create_wsi_level_split_gen(self, split: str, features: np.array, df: pd.DataFrame):
        """
        create data generator for train, validation or test split ('train', 'val' or 'test')
        """
        if split == 'train':
            wsi_df_split = self.wsi_df[np.logical_and(self.wsi_df['Partition'] == 'train', self.wsi_df['labeled'])]
            shuffle = True
        elif split == 'val':
            wsi_df_split = self.wsi_df[self.wsi_df['Partition'] == 'val']
            shuffle = False
        else:
            wsi_df_split = self.wsi_df[self.wsi_df['Partition'] == 'test']
            shuffle = False

        one_hot_labels = np.expand_dims(tf.keras.utils.to_categorical(np.array(wsi_df_split['isup_grade'])), axis=1)
        images, labels = self.prepare_bags(features, np.array(wsi_df_split['slide_id']),
                                           one_hot_labels, np.array(df['wsi']))

        data_gen = FeatureDataGenerator(images, labels, shuffle)
        return data_gen

    def prepare_bags(self, features, bag_names, bag_labels, bag_names_per_instance):
        """
        Create MIL bags.
        """
        bag_names = bag_names
        images = []
        labels = []

        for i in range(len(bag_names)):
            bag_name = bag_names[i]
            id_bool = (bag_name == bag_names_per_instance)
            bag_features = features[id_bool]

            images.append(bag_features)
            labels.append(bag_labels[i])

        return images, labels