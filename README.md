# Active Learning with BNNs for prostate cancer

## Installation and Usage
To make this code run on your linux machine you need to:
* Install miniconda (or anaconda): https://docs.anaconda.com/anaconda/install/linux/ 
* Set up a conda environment and activate it:
    * `conda env create --file environment.yaml`
    * `conda activate tensorlfow_2_3`
* Download dataset
* Edit the configuration:
    * `./config.yaml` for general settings
    * `./dataset_dependent/panda/config.yaml` for dataset dependent settings
* Run the program:
    * `python ./src/main.py`
    
## Configuration
The experiment parameters are set by using three different configuration files. All of those files are in the yaml format.
1. The default configuration. The path of this file is specified by the '-dc /path/to/default/config.yaml' argument.
2. The dataset configurtaion. The path of the dataset configuration file is specified in the default configuration under 'data:
  dataset_config: /path/to/dataset/config.yaml'
3. The experiment config. The path of this file is specified by the '-ef /path/to/experiment/folder/' argument. 
If a file 'exp_config.yaml' is found in this folder, any parameter of the default or dataset config can be overwritten.
The same folder will be used to store all experiment artifacts like images.