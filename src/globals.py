import collections
import yaml
import os
import warnings
import shutil
import tensorflow as tf
import numpy as np
import random

config = {}

def config_update(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, collections.Mapping):
            tmp = config_update(orig_dict.get(key, { }), val)
            orig_dict[key] = tmp
        elif isinstance(val, list):
            orig_dict[key] = val
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict

def init_global_config(args):
    global config

    # load default config
    with open(args.default_config) as file:
        config = yaml.full_load(file)
    with open(config["data"]["dataset_config"]) as file:
        config_data_dependent = yaml.full_load(file)

    config = config_update(config, config_data_dependent)

    # load experiment config, overwrite parameters if double
    if args.experiment_folder != 'None':
        experiment_config = os.path.join(args.experiment_folder, 'exp_config.yaml')
        if os.path.exists(experiment_config):
            with open(experiment_config) as file:
                exp_config = yaml.full_load(file)
            config = config_update(config, exp_config)
        config['logging']['experiment_folder'] = args.experiment_folder
        config['logging']['run_name'] = os.path.basename(args.experiment_folder)
    else:
        out_dir = './output/'
        os.makedirs(out_dir, exist_ok=True)
        warnings.warn("No experiment folder was given. Use ./output folder to store experiment results.")
        config['logging']['experiment_folder'] = out_dir
        config['logging']['run_name'] = 'default'

    for f in os.listdir(config['logging']['experiment_folder']):
        path = os.path.join(config['logging']['experiment_folder'], f)

        if os.path.isfile(path) and f != 'exp_config.yaml':
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(os.path.join(config['logging']['experiment_folder'], f))

    set_global_determinism(seed=config['random_seed'])


def set_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=0):
    set_seeds(seed=seed)

    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


