import os
import random
import pandas as pd
import numpy as np


def extract_df_info(dataframe_raw, wsi_df, data_config, split='train'):
    print('Preparing data split '+split)
    # Notice: class 0 = NC, class 1 = G3, class 2 = G4, class 3 = G5
    dataframe = pd.DataFrame()
    dataframe["image_path"] = 'images/' + dataframe_raw["image_name"]
    wsis = dataframe_raw["image_name"].str.split('_').str[0]
    dataframe["wsi"] = wsis
    # dataframe_raw["wsi"]  = wsis

    dataframe = get_instance_classes(dataframe, dataframe_raw, wsi_df, data_config, split)

    dataframe = dataframe.sort_values(by=['image_path'], ignore_index=True)
    dataframe = dataframe.reset_index(inplace=False)
    dataframe['index'] = dataframe.index
    # return dataframe with some instance labels
    return dataframe

def clean_wsi_df(wsi_df, train_df, val_df, test_df):
    used_wsis = np.concatenate([np.unique(train_df['wsi']), np.unique(val_df['wsi'])], axis=0)
    if test_df is not None:
        used_wsis = np.concatenate([used_wsis, np.unique(test_df['wsi'])], axis=0)
    new_wsi_df = wsi_df.loc[wsi_df['slide_id'].isin(used_wsis)]
    return new_wsi_df


def get_instance_classes(dataframe, dataframe_raw, wsi_df, data_config, split):
    dataframe = set_wsi_labels_pc(dataframe, wsi_df)
    class_columns = [dataframe_raw["NC"], dataframe_raw["G3"], dataframe_raw["G4"], dataframe_raw["G5"]]
    dataframe["class"] = np.argmax(class_columns, axis=0).astype(str)
    if data_config['supervision'] == 'active_learning' and split == 'train':
        dataframe["labeled"] = False
    else:
        dataframe["labeled"] = True

    return dataframe

def get_start_label_ids(dataframe, wsi_dataframe, data_config):
    class_ids =  np.unique(dataframe['class'])
    number_wsis = data_config['active_learning']['start']['wsis_per_class']
    number_labels = data_config['active_learning']['start']['labels_per_class_and_wsi']
    sampled_indices = np.array([])
    selected_wsis = []
    for class_id in class_ids:
        for iter_wsi in range(number_wsis):
            for attempt in range(10):
                wsi_candidates = (wsi_dataframe['class_primary'] == int(class_id)) & (wsi_dataframe['Partition'] == 'train')
                wsi_selection = np.random.choice(wsi_dataframe['slide_id'].loc[wsi_candidates], size=1)[0]
                df_candidates = (dataframe['wsi'] == wsi_selection) & (dataframe['class'] == class_id)
                if np.sum(df_candidates) >= number_labels and not (wsi_selection in selected_wsis):
                    break
                elif attempt == 9:
                    raise Exception('Not enough start samples found. Choose a smaller number of labels per WSI.')
            wsi_dataframe['labeled'].loc[wsi_dataframe['slide_id'] == wsi_selection] = True
            selected_wsis.append(wsi_selection)
            if number_labels != -1:
                df_selection = np.random.choice(dataframe['index'].loc[df_candidates],
                                                size=number_labels,
                                                replace=False)
            else:
                df_selection = dataframe['index'].loc[dataframe['wsi'] == wsi_selection]
            sampled_indices = np.concatenate([sampled_indices, df_selection])
    return sampled_indices, selected_wsis
#
#
# def hide_instance_labels_pc(dataframe, wsi_dataframe, num_instance_samples):
#     rows_of_visible_instance_labels = get_rows_of_visible_instances_pc(dataframe, wsi_dataframe, num_instance_samples)
#     dataframe["instance_label"] = 4  # class_id 4: unlabeled
#     dataframe["instance_label"][rows_of_visible_instance_labels] = dataframe['class']
#     dataframe['class'] = dataframe['instance_label'].astype(str)
#     return dataframe
#
# def hide_instance_labels_cb(dataframe, wsi_dataframe, num_instance_samples):
#     rows_of_visible_instance_labels = get_rows_of_visible_instances_cb(dataframe, wsi_dataframe, num_instance_samples)
#     dataframe["instance_label"] = 2  # class_id 2: unlabeled
#     dataframe["instance_label"][rows_of_visible_instance_labels] = dataframe['class']
#     dataframe['class'] = dataframe['instance_label'].astype(str)
#     return dataframe

def set_wsi_labels_pc(dataframe, wsi_dataframe):
    dataframe['wsi_index'] = -1
    dataframe["wsi_primary_label"] = -1
    dataframe["wsi_secondary_label"] = -1
    # dataframe['wsi_contains_unlabeled'] = True
    for row in range(len(wsi_dataframe)):
        id_bool = dataframe['wsi']==wsi_dataframe['slide_id'][row]
        if np.any(id_bool) == False:
            continue
        dataframe['wsi_index'].iloc[id_bool] = row
        dataframe['wsi_primary_label'].iloc[id_bool] = np.max([int(wsi_dataframe['Gleason_primary'][row]) - 2, 0])
        dataframe['wsi_secondary_label'].iloc[id_bool] = np.max([int(wsi_dataframe['Gleason_secondary'][row]) - 2, 0])
    assert(np.all(dataframe['wsi_index'] != -1))
    assert(np.all(dataframe['wsi_primary_label'] != -1))
    assert(np.all(dataframe['wsi_secondary_label'] != -1))
    return dataframe

def extract_wsi_df_info(wsi_df):
    wsi_df['Gleason_primary'] = wsi_df['gleason_score'].str.split('+').str[0].astype(int)
    wsi_df['Gleason_secondary'] = wsi_df['gleason_score'].str.split('+').str[1].astype(int)
    wsi_df['class_primary'] = np.clip(wsi_df['Gleason_primary'] - 2, a_min=0, a_max=3)
    wsi_df['class_secondary'] = np.clip(wsi_df['Gleason_secondary'] - 2, a_min=0, a_max=3)
    wsi_df['labeled'] = False
    wsi_df.rename(columns={"image_id": "slide_id"}, inplace=True)

    return wsi_df

#
# def set_wsi_labels_cb(dataframe, wsi_dataframe):
#     dataframe['wsi_index'] = -1
#     dataframe["wsi_label"] = -1
#     dataframe['wsi_contains_unlabeled'] = True
#     for row in range(len(wsi_dataframe)):
#         id_bool = dataframe['wsi']==wsi_dataframe['slide_id'][row]
#         if np.any(id_bool) == False:
#             continue
#         dataframe['wsi_index'][id_bool] = row
#         dataframe['wsi_label'][id_bool] = wsi_dataframe['class'][row].astype(int)
#     assert(np.all(dataframe['wsi_index'] != -1))
#     assert(np.all(dataframe['wsi_label'] != -1))
#     return dataframe

# TODO: adapt to binary
# def check_if_wsi_contains_unlabeled(dataframe, wsi_dataframe, dataset_type):
#     if dataset_type == 'prostate_cancer':
#         wsi_label_col = 'Gleason_primary'
#         unlabeled_index = str(4)
#     else:
#         wsi_label_col = 'class'
#         unlabeled_index = str(2)
#     for row in range(len(wsi_dataframe)):
#         id_bool = dataframe['wsi']==wsi_dataframe['slide_id'][row]
#         if np.any(id_bool) == False:
#             continue
#         if wsi_dataframe[wsi_label_col][row] == '0':
#             wsi_contains_unlabeled = False
#         elif np.all(dataframe['class'][id_bool] != unlabeled_index):
#             wsi_contains_unlabeled = False
#         else:
#             wsi_contains_unlabeled = True
#         dataframe['wsi_contains_unlabeled'][id_bool] = wsi_contains_unlabeled
#     return dataframe
#
#
# def get_rows_of_visible_instances_pc(dataframe, wsi_dataframe, num_instance_samples):
#     rows_of_visible_instance_labels = []
#     for wsi_df_row in range(len(wsi_dataframe["Gleason_primary"])):
#         if wsi_dataframe['Gleason_primary'][wsi_df_row] == wsi_dataframe['Gleason_secondary'][wsi_df_row] == 0:
#             negative_bag = True
#         else:
#             negative_bag = False
#
#         primary_gleason_grade_rows = []
#         secondary_gleason_grade_rows = []
#         for instance_df_row in range(len(dataframe["image_path"])):
#             if wsi_dataframe['slide_id'][wsi_df_row] == dataframe["wsi"][instance_df_row]:
#                 if negative_bag:
#                     rows_of_visible_instance_labels.append(instance_df_row)
#                 elif wsi_dataframe['Gleason_primary'][wsi_df_row] - 2 == int(dataframe["class"][instance_df_row]):
#                     primary_gleason_grade_rows.append(instance_df_row)
#                 elif wsi_dataframe['Gleason_secondary'][wsi_df_row] - 2 == int(dataframe["class"][instance_df_row]):
#                     secondary_gleason_grade_rows.append(instance_df_row)
#         rows_of_visible_instance_labels += sample_or_complete_list(primary_gleason_grade_rows, num_instance_samples)
#         rows_of_visible_instance_labels += sample_or_complete_list(secondary_gleason_grade_rows, num_instance_samples)
#     return rows_of_visible_instance_labels
#
# def get_rows_of_visible_instances_cb(dataframe, wsi_dataframe, num_instance_samples):
#     rows_of_visible_instance_labels = []
#     for wsi_df_row in range(len(wsi_dataframe)):
#         if wsi_dataframe['class'][wsi_df_row] == 0:
#             negative_bag = True
#         else:
#             negative_bag = False
#
#         positive_rows = []
#         for instance_df_row in range(len(dataframe["image_path"])):
#             if wsi_dataframe['slide_id'][wsi_df_row] == dataframe['wsi'][instance_df_row]:
#                 if negative_bag:
#                     rows_of_visible_instance_labels.append(instance_df_row)
#                 elif dataframe['class'][instance_df_row] == '1':
#                     positive_rows.append(instance_df_row)
#
#         rows_of_visible_instance_labels += sample_or_complete_list(positive_rows, num_instance_samples)
#     return rows_of_visible_instance_labels
#
#
# def sample_or_complete_list(list, num_samples):
#     random.seed(42)
#     if num_samples >= len(list):
#         return list
#     else:
#         return random.sample(list, num_samples)