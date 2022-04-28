# TODO
# adjust mlflow_log.py
# save acquired patches for each step

import argparse
import os
import collections
import yaml
import tensorflow as tf
from typing import Dict, Optional, Tuple
import globals
from data import DataGenerator
from model_handler import ModelHandler
from mlflow_log import MLFlowLogger


def main():
    # Only necessary if certain GPUs are used (like nvidia 2060rtx)
    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)

    config = globals.config

    # Init logging with mlflow (see README)
    logger = MLFlowLogger()
    logger.config_logging()

    print("Create data generators..")
    data_gen = DataGenerator()

    print("Load classification model")
    model = ModelHandler(data_gen.get_number_of_training_points())

    print("Train")
    logger.data_logging(data_gen.get_train_data_statistics())
    model.train(data_gen)

    if config['logging']['log_artifacts']:
        logger.log_artifacts()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cancer Classification")
    parser.add_argument("--default_config", "-dc", type=str, default="./config.yaml",
                        help="Config path (yaml file expected) to default config.")
    parser.add_argument("--experiment_folder", "-ef", type=str, default="None",
                        help="Config path to experiment folder. Parameters will override defaults. Optional.")
    args = parser.parse_args()
    globals.init_global_config(args)
    main()