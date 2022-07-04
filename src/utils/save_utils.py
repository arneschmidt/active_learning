import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import globals


def save_dataframe_with_output(dataframe: pd.DataFrame, predictions: np.array, features: np.array, output_dir: str,
                               save_name: str):
    out_df = pd.DataFrame()
    out_df['cnn_prediction'] = np.argmax(predictions, axis=1)
    for feature_id in range(np.shape(features)[1]):
        feature_name = 'feature_' + str(feature_id)
        out_df[feature_name] = features[:, feature_id]
    out_df['instance_name'] = dataframe['image_path']
    out_df['instance_label'] = dataframe['class']
    out_df['bag_name'] = dataframe['wsi']
    out_df['bag_label'] = dataframe['wsi_label']
    output_dir = os.path.join(output_dir, 'feature_predictions')
    os.makedirs(output_dir, exist_ok=True)
    save_path = output_dir + '/' + save_name + '.csv'
    print('Saving dataframe with output: ' + save_path)
    out_df.to_csv(save_path, index=False)


def save_metrics_artifacts(artifacts, output_dir):
    if 'confusion_matrices' in artifacts:
        save_confusion_matrices(artifacts['confusion_matrics'], output_dir)
    if 'roc' in artifacts:
        save_roc(artifacts['roc'], output_dir)


def save_confusion_matrices(confusion_matrices, output_dir):
    output_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(output_dir, exist_ok=True)
    for name, matrix in confusion_matrices.items():
        save_path = output_dir + '/' + name + '.csv'
        np.savetxt(save_path, matrix, '%10i',  delimiter=',')


def save_roc(roc, output_dir):
    output_dir = os.path.join(output_dir, 'roc')
    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    lw = 2
    plt.plot(roc['fpr'], roc['tpr'], color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic ')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(os.path.join(output_dir, 'roc.jpg'))
    np.savetxt(os.path.join(output_dir, 'fpr.csv'), roc['fpr'], delimiter=",")
    np.savetxt(os.path.join(output_dir, 'tpr.csv'), roc['tpr'], delimiter=",")
    np.savetxt(os.path.join(output_dir, 'thresholds.csv'), roc['thresholds'], delimiter=",")


def save_acquired_images(data_gen, highest_unc_indices, highest_unc_values, train_indices, acquisition_step):
    data_dir = globals.config['data']["dir"]
    out_dir = globals.config['logging']['experiment_folder']
    out_dir = os.path.join(out_dir, str(acquisition_step))
    os.makedirs(out_dir, exist_ok=True)
    data_gen.wsi_df.to_csv(os.path.join(out_dir, 'wsi_df.csv'))
    out_dir_acq = os.path.join(out_dir, 'acquisition')
    out_dir_acq_images = os.path.join(out_dir_acq, 'images')
    out_dir_acq_masks = os.path.join(out_dir_acq, 'masks')
    os.makedirs(out_dir_acq_images, exist_ok=True)
    os.makedirs(out_dir_acq_masks, exist_ok=True)

    os.makedirs(out_dir_acq, exist_ok=True)
    for i in range(train_indices.shape[0]):

        file = data_gen.train_df['image_path'].loc[train_indices[i]]
        path = os.path.join(data_dir, file)
        shutil.copy(path, out_dir_acq_images)
        mask_path = path.replace('/images/', '/masks/')
        shutil.copy(mask_path, out_dir_acq_masks)

    for unc in highest_unc_indices.keys():
        out_dir_unc = os.path.join(out_dir, unc)
        out_dir_unc_images = os.path.join(out_dir_unc, 'images')
        out_dir_unc_masks = os.path.join(out_dir_unc, 'masks')
        os.makedirs(out_dir_unc_images, exist_ok=True)
        os.makedirs(out_dir_unc_masks, exist_ok=True)

        top_unc = highest_unc_indices[unc]
        images = []
        uncertainties = []
        for i in range(top_unc.shape[0]):
            file = data_gen.train_df['image_path'].loc[top_unc[i]]
            images.append(str(file))
            uncertainties.append(highest_unc_values[unc][i])
            path = os.path.join(data_dir, file)
            shutil.copy(path, out_dir_unc_images)

            mask_path = path.replace('/images/', '/masks/')
            shutil.copy(mask_path, out_dir_unc_masks)

        out_df = pd.DataFrame()
        out_df['image'] = np.array(images)
        out_df['unc'] = np.array(uncertainties)
        out_df.to_csv(os.path.join(out_dir_unc, 'uncertainties.csv'))


