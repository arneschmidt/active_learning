# TODO
# fix bug of logging _1, _2, _3 metrics
# repatch panda
# take best validation model
# print acquisition score and entropy


import argparse
import os
import collections
import yaml
import tensorflow as tf
from typing import Dict, Optional, Tuple
import globals
from data import DataGenerator
from model_handler import ModelHandler
from mlflow_log import start_logging, data_logging


def main():
    # Only necessary if certain GPUs are used (like nvidia 2060rtx)
    # devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(devices[0], True)

    config = globals.config

    # Init logging with mlflow (see README)
    start_logging()


    print("Create data generators..")
    data_gen = DataGenerator()

    print("Load classification model")
    model = ModelHandler(data_gen.get_number_of_training_points())

    print("Train")
    data_logging(data_gen.get_train_data_statistics())
    model.train(data_gen)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cancer Classification")
    parser.add_argument("--default_config", "-dc", type=str, default="./config.yaml",
                        help="Config path (yaml file expected) to default config.")
    parser.add_argument("--experiment_folder", "-ef", type=str, default="None",
                        help="Config path to experiment folder. Parameters will override defaults. Optional.")
    args = parser.parse_args()
    globals.init_global_config(args)
    main()