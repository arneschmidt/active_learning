import os
import cv2
import pandas as pd
import numpy as np
import skimage.io
import skimage.transform
from scipy.interpolate import griddata

# best WSI:
# 24ecf26ce811ea7f0116d7ea5388bc4a
# 8d9bf04e714c959d4c571030c51ee9f5

csv_path = '/home/arne/projects/active_learning/experiment_output/a31d621ea058403b82a4a2ad755fc8fc_acq09/test_predictions.csv'
wsi_dir = '/home/arne/datasets/Panda/train_images/'
masks_dir = '/home/arne/datasets/Panda/train_label_masks/'
output_dir = '/home/arne/projects/active_learning/experiment_output/a31d621ea058403b82a4a2ad755fc8fc_acq09/visualization/'
wsi_list = []
wsi_cut = []
resize_factor = 0.2
output_types = ['mask', 'class', 'prediction', 'epistemic_unc', 'aleatoric_unc', 'ood_prob', 'acq_score']
patch_stride = 256
prepatching_factor = 0.5
uncertainty_normalization = False

patch_stride_resized = patch_stride * resize_factor * (1/prepatching_factor)


def create_grid(x, y, values, dim):
    # OpenCV and skimage expect the coordinates in the order (h,w) and (y,x)
    x_n_patches = int(np.floor(dim[0] / patch_stride_resized))
    y_n_patches = int(np.floor(dim[1] / patch_stride_resized))

    x = np.array(x, dtype=np.uint8)
    y = np.array(y, dtype=np.uint8)
    patch_grid = np.zeros(shape=[y_n_patches, x_n_patches])
    for i in range(len(x)):
        patch_grid[int(y[i]), int(x[i])] = values[i]

    patch_coordinate_grid = np.mgrid[0:y_n_patches, 0:x_n_patches] * patch_stride_resized + patch_stride_resized
    patch_coordinate_grid = np.moveaxis(patch_coordinate_grid, 0, 2)
    pixel_grid = np.mgrid[0:dim[1], 0:dim[0]]  # order: h,w (y,x)

    patch_coordinate_grid = np.reshape(patch_coordinate_grid, (-1, 2))
    patch_grid = np.reshape(patch_grid, (-1))

    # expected dims: patch_coordinate_grid [n_patches, 2]; patch_grid [n_patches,]; pixel_grid [2,wsi_height, wsi_width]
    pixel_grid_out = griddata(patch_coordinate_grid, patch_grid, (pixel_grid[0], pixel_grid[1]),
                              method='cubic', fill_value=0.0)
    pixel_grid_out = np.expand_dims(pixel_grid_out, axis=-1)
    return pixel_grid_out


def cut_and_save_image(wsi_masked, cut, wsi_name, output_type):
    if len(cut) == 4:
        wsi_cut = wsi_masked[cut[0], cut[1], cut[2], cut[3]]
    else:
        wsi_cut = wsi_masked
    wsi_out_dir = os.path.join(output_dir, wsi_name)
    os.makedirs(wsi_out_dir, exist_ok=True)
    path = os.path.join(wsi_out_dir, output_type + '.png')
    skimage.io.imsave(path, wsi_cut)


def generate_wsi_with_mask(wsi, wsi_name, cut, wsi_patch_df, dim, output_type):
    color_GG3 = [-255, -255, 255]
    color_GG4 = [-255, 255, 255]
    color_GG5 = [-255, 255, -255]
    color_uncertainty = [255, 255, -255]

    if output_type == 'mask':
        mask_path = os.path.join(masks_dir, wsi_list[i] + '_mask.tiff')
        class_mask = skimage.io.MultiImage(mask_path)[0]
        class_mask = cv2.resize(class_mask, dim, interpolation=cv2.INTER_CUBIC)

        mask = np.zeros_like(class_mask, dtype=np.uint8)
        class_mask = np.expand_dims(class_mask[:, :, 0], axis=2)
         # use only one channel
        mask_ones = np.ones_like(mask, dtype=np.uint8)
        mask = np.where(class_mask==3, color_GG3*mask_ones, mask).astype(np.uint8)
        mask = np.where(class_mask==4, color_GG4*mask_ones, mask).astype(np.uint8)
        mask = np.where(class_mask==5, color_GG5*mask_ones, mask).astype(np.uint8)
        mask_factor = 0.4
    elif output_type == 'class' or output_type == 'prediction':
        classes = wsi_patch_df[output_type]
        grid_gg3 = create_grid(wsi_patch_df['x'], wsi_patch_df['y'], np.array((classes == 1), dtype=np.float16), dim)
        grid_gg4 = create_grid(wsi_patch_df['x'], wsi_patch_df['y'], np.array((classes == 2), dtype=np.float16), dim)
        grid_gg5 = create_grid(wsi_patch_df['x'], wsi_patch_df['y'], np.array((classes == 3), dtype=np.float16), dim)
        mask = np.clip(grid_gg3*color_GG3 + grid_gg4*color_GG4 + grid_gg5*color_GG5, a_min=-255, a_max=255).astype(np.int32)
        mask_factor = 0.2
    else:
        values = np.array(wsi_patch_df[output_type])
        if output_type == 'ood_prob':
            values = (values - 0.1)
        if output_type == 'acq_score':
            values = (values + 0.1) * 2
        if uncertainty_normalization:
            values = (values - np.min(values))/(np.max(values) - np.min(values))
        else:
            values = np.clip(values, a_min=0.0, a_max=1.0)
        grid = create_grid(wsi_patch_df['x'], wsi_patch_df['y'], values, dim)
        mask = np.clip(grid*color_uncertainty, a_min=-255, a_max=255).astype(np.int32)
        mask_factor = 0.5

    wsi_masked = np.clip(wsi + mask_factor*mask, a_min=0, a_max=255).astype(np.uint8)
    cut_and_save_image(wsi_masked, cut, wsi_name, output_type)


if __name__ == "__main__":
    # In this whole script we use the following coordinate order:
    # x - width - horizontal
    # y - height - vertical
    test_patch_df = pd.read_csv(csv_path)
    test_patch_df['y'] = test_patch_df['image_path'].str.split('_|.jpg', expand=True).iloc[:, 1].astype('int')
    test_patch_df['x'] = test_patch_df['image_path'].str.split('_|.jpg', expand=True).iloc[:, 2].astype('int')

    if len(wsi_list) == 0:
        wsi_list = np.unique(test_patch_df['wsi'])
        wsi_cut = []

    for i in range(len(wsi_list)):
        print('Processing ' + wsi_list[i])
        wsi_path = os.path.join(wsi_dir, wsi_list[i] + '.tiff')
        wsi = skimage.io.MultiImage(wsi_path)
        if len(wsi) > 0:
            wsi = wsi[0]
        else:
            print('No image data! WSI ' + wsi_list[i])
            continue
        wsi_patch_df = test_patch_df[test_patch_df['wsi'] == wsi_list[i]]

        width = int(wsi.shape[1] * resize_factor)
        height = int(wsi.shape[0] * resize_factor)

        dim = (width, height) # for resizing
        wsi = cv2.resize(wsi, dim, interpolation=cv2.INTER_CUBIC)

        cut_resized = []
        if len(wsi_cut)>0:
            if len(wsi_cut[i]) > 0:
                cut_resized = np.array(wsi_cut[i] * resize_factor).astype(np.int)

        for j in range(len(output_types)):
            print('Visualizing ' + output_types[j])
            generate_wsi_with_mask(wsi, wsi_list[i], cut_resized, wsi_patch_df, dim, output_types[j])










