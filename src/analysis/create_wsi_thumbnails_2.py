import os
import cv2
import pandas as pd
import numpy as np
import skimage.io
import skimage.transform
from scipy.interpolate import griddata

wsi_dir = '/home/arne/datasets/Panda/train_images/'
masks_dir = '/home/arne/datasets/Panda/train_label_masks/'
wsi_df_path = '/home/arne/projects/panda_scripts/artifacts/train_active.csv'
out_dir = '/home/arne/projects/active_learning/experiment_output/visualization/thumbnails_2/'
resize_factor = 0.2
wsi_df = pd.read_csv(wsi_df_path)

wsi_df_test = wsi_df[wsi_df['Partition'] == 'test']
wsi_df_test_cancerous = wsi_df_test[wsi_df_test['isup_grade'].astype(int) > 0]

wsi_list = np.array(wsi_df_test_cancerous['image_id'])

os.makedirs(out_dir, exist_ok=True)

color_NC = [96, 210, 128]
color_GG3 = [255, 224, 32]
color_GG4 = [255, 130, 0]
color_GG5 = [255, 255, -255]


for i in range(len(wsi_list)):
    print('Processing ' + wsi_list[i])
    wsi_path = os.path.join(wsi_dir, wsi_list[i] + '.tiff')
    wsi = skimage.io.MultiImage(wsi_path)
    if len(wsi) > 0:
        wsi = wsi[0]
    else:
        print('No image data! WSI ' + wsi_list[i])
        continue

    width = int(wsi.shape[1] * resize_factor)
    height = int(wsi.shape[0] * resize_factor)
    dim = (width, height)  # for resizing
    wsi = cv2.resize(wsi, dim, interpolation=cv2.INTER_CUBIC)

    mask_path = os.path.join(masks_dir, wsi_list[i] + '_mask.tiff')
    class_mask = skimage.io.MultiImage(mask_path)
    if len(class_mask) > 0:
        class_mask = class_mask[0]
    else:
        print('No image data! WSI ' + wsi_list[i])
        continue

    wsi_grey = cv2.cvtColor(wsi, cv2.COLOR_BGR2GRAY)
    wsi_grey = 255 - np.expand_dims(wsi_grey, axis=2)

    class_mask = cv2.resize(class_mask, dim, interpolation=cv2.INTER_CUBIC)

    mask = np.ones_like(class_mask, dtype=np.uint8)*255
    class_mask = np.expand_dims(class_mask[:, :, 0], axis=2)
    class_mask_wsi = np.where(wsi_grey > 20, 1, np.zeros_like(class_mask, dtype=np.uint8)).astype(np.uint8)
    class_mask = np.where(class_mask > 2, class_mask, class_mask_wsi).astype(np.uint8)
    # use only one channel
    mask_ones = np.ones_like(mask, dtype=np.uint8)
    mask = np.where(class_mask == 1, color_NC * mask_ones, mask).astype(np.uint8)
    # mask = np.where(class_mask == 2, color_NC * mask_ones, mask).astype(np.uint8)
    mask = np.where(class_mask == 3, color_GG3 * mask_ones, mask).astype(np.uint8)
    mask = np.where(class_mask == 4, color_GG4 * mask_ones, mask).astype(np.uint8)
    mask = np.where(class_mask == 5, color_GG5 * mask_ones, mask).astype(np.uint8)

    # wsi_masked = np.array(np.clip((255 + wsi_grey*color_NC) + mask, a_min=0, a_max=255), dtype=np.uint8)
    wsi_masked = np.array(np.clip(mask, a_min=0, a_max=255), dtype=np.uint8)

    path = os.path.join(out_dir, wsi_list[i] + '.png')
    skimage.io.imsave(path, wsi_masked)
    skimage.io.imsave(os.path.join(out_dir, wsi_list[i] + '_original.png'), wsi)

