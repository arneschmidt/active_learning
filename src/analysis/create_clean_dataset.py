import os
import pandas as pd
import numpy as np
import shutil

dataset_path = '/home/arne/data/Panda/Panda_patches_center/'
patch_df_file = 'train_patches.csv'
image_dir = 'images/'
output_dir = 'subselection3/'
patch_df = pd.read_csv(dataset_path + patch_df_file)

n = len(patch_df)
total_patches = 800
list_selected = []

os.makedirs(dataset_path + output_dir)

chosen_patch_names = np.random.choice(patch_df['image_name'], 800)

for p in chosen_patch_names:
    shutil.copy(dataset_path + image_dir + p, dataset_path + output_dir)



