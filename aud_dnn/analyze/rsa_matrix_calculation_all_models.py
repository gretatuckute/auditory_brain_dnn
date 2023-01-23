"""
Computes pca on the activations to the 165 sound set for all models and computes the effective dimensionality. 

Assumes that the statistics are already normalized, if normalization should be performed. # TODO: change this to match normalization for predictions? 
"""
import numpy as np
import torch as ch
from pathlib import Path
import pickle
import matplotlib.pylab as plt
import argparse
import h5py
import os
from plotting_specs import d_layer_reindex, d_sound_category_colors, sound_category_order, d_model_colors, d_model_names
from sklearn.preprocessing import StandardScaler
import sys # TODO: this is a bit hacky TODO: remove
sys.path.append('../../') # TODO: remove
sys.path.append('../')
from aud_dnn.utils import get_source_features
from scipy import stats
import pandas as pd 
from scipy.io import loadmat

from scipy.stats import wilcoxon

try:
    import pingouin as pg
except:
    pass

# So that we can edit the text in illustrator
import matplotlib
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

DATADIR = (Path(os.getcwd()) / '..' / 'data').resolve()
RESULTDIR = (Path(os.getcwd()) / '..' / 'results').resolve()

# TODO: move this into utils? 
## Stimuli (original indexing, activations are extracted in this order) ##
sound_meta = np.load(os.path.join(DATADIR, f'neural/NH2015/neural_stim_meta.npy'))

# Extract wav names in order (this is how the neural data is ordered --> order all activations in the same way)
stimuli_IDs = []
stimuli_categories = []

for i in sound_meta:
    stimuli_IDs.append(i[0][:-4].decode("utf-8")) # remove .wav
    stimuli_categories.append(i[4])

correlation_plot_order = []
category_color_order = []
for category in sound_category_order:
    correlation_plot_order+=[i for i, x in enumerate(stimuli_categories) if x == category]
    category_color_order+=[x for i, x in enumerate(stimuli_categories) if x == category]

def run_correlation_on_feature_matrix(feature_matrix, corr_type='pearson'):
    if corr_type=='pearson':
        return np.corrcoef(feature_matrix)
    elif corr_type=='spearman':
        r, p = stats.spearmanr(feature_matrix)
        return r

def correlate_two_matrices_rsa(matrix_1, matrix_2, distance_measure='pearson'):
    upper_tri = np.triu_indices_from(matrix_1, k=1)

    if distance_measure=='pearson':
        r, p = stats.pearsonr(matrix_1[upper_tri], matrix_2[upper_tri]) 
        return r # TODO: return p-value? 
    elif distance_measure=='spearman':
        r, p = stats.spearmanr(matrix_1[upper_tri], matrix_2[upper_tri])
        return r # TODO: return p-value?

# TODO: Is there a "correct" noise correlation to use for the RSA analysis?  
def noise_correct_from_scan_splits(true_r, split_r, data_len):
    if data_len==82: # TODO: are thse the right floors for this analysis? # If we are using spearman, are these values the same????? 
        correlation_floor = 0.182
    elif data_len==83:
        correlation_floor= 0.183
    else:
        raise ValueError('Not Implemented For this data len')

    rv_med = np.median(split_r)
    rv = np.max([3 * rv_med / (1 + 2 * rv_med), correlation_floor])

    # TODO: What should this be? 
    return true_r / rv

# TODO: Is there a "correct" noise correlation to use for the RSA analysis?
def noise_correct_two_roi_from_scan_splits(true_r, split_r_roi_1, split_r_roi_2, data_len):
    if data_len==82: # TODO: are thse the right floors for this analysis? # If we are using spearman, are these values the same?????
        correlation_floor = 0.182
    elif data_len==83:
        correlation_floor= 0.183
    elif data_len==165:
        correlation_floor = 0
    else:
        raise ValueError('Not Implemented For this data len')

    rv_med_roi_1 = np.median(split_r_roi_1)
    rv_roi_1 = np.max([3 * rv_med_roi_1 / (1 + 2 * rv_med_roi_1), correlation_floor])

    rv_med_roi_2 = np.median(split_r_roi_2)
    rv_roi_2 = np.max([3 * rv_med_roi_2 / (1 + 2 * rv_med_roi_2), correlation_floor])

    # TODO: What should this be? 
    return true_r / np.sqrt(rv_roi_1 * rv_roi_2)
   
def get_neural_data_matrix(roi_name=None, data_by_participant=False, reorder_sounds=True, noise_correction=False, target='NH2015'):
    # TODO: update for Dana's data too. 
    DATADIR='/om/user/gretatu/aud-dnn/data'

    if target=='NH2015':
        voxel_meta_with_roi = pd.read_pickle(os.path.join(DATADIR, f'neural/{target}/df_roi_meta.pkl'))
        voxel_data_all = np.load(os.path.join(DATADIR, f'neural/{target}/voxel_features_array.npy'))
        voxel_meta_all = np.load(os.path.join(DATADIR, f'neural/{target}/voxel_features_meta.npy'))
        voxel_idx_list = ([np.where(voxel_meta_all['voxel_id']==k)[0][0] for k in voxel_meta_with_roi['voxel_id']])
        voxel_data = voxel_data_all[:,voxel_idx_list,:]
        voxel_meta = voxel_meta_with_roi
    elif target == 'B2021':
        voxel_meta_with_roi = pd.read_pickle(os.path.join(DATADIR, f'neural/{target}/df_roi_meta.pkl'))
        sound_meta_mat = loadmat(os.path.join(DATADIR, f'neural/{target}/stim_info_v4'))['stim_info']
        stimuli_IDs_B2021 = []
        for i in sound_meta_mat['stim_names'][0][0]:
            stimuli_IDs_B2021.append((i[0][0].astype(str)))
        assert(stimuli_IDs == stimuli_IDs_B2021[:165]) # same order as NH2021
        voxel_data = np.load(os.path.join(DATADIR, f'neural/{target}/voxel_features_array.npy'))
        # Truncate to only run the first 165 sound set
        voxel_data = voxel_data[:len(stimuli_IDs), :, :]
        voxel_meta = voxel_meta_with_roi
    else:
        raise LookupError(f'Target dataset of interest {target} not found!')

    if reorder_sounds: # Group sounds by categories.  
        voxel_data = voxel_data[correlation_plot_order,:,:]

    if (roi_name is not None) and (roi_name!="All"):
        is_in_roi = voxel_meta['roi_label_general']==roi_name 
        voxel_meta, voxel_data = voxel_meta[is_in_roi], voxel_data[:, is_in_roi, :]
        print('%s voxels in ROI %s'%(voxel_data.shape, roi_name))

    if data_by_participant:
        all_participant_data = {}
        for participant in np.unique(voxel_meta['subj_idx']):
            is_in_participant = voxel_meta['subj_idx'] == participant
            all_participant_data['participant_%d'%participant] = {'voxel_data': voxel_data[:, is_in_participant, :], 
                                                                  'voxel_meta': voxel_meta[is_in_participant]}
        return all_participant_data
    else:
        return {'voxel_data': voxel_data, 'voxel_meta': voxel_meta}

def get_all_participants_correlation_matrices(participant_neural_data_dict, all_splits_train_test_bool, mean_subtract, with_std, corr_type_matrix, distance_measure, noise_correction, n_CV_splits):
    all_participants_neural_rsa_matrix = {}
    for participant_id, participant_info in participant_neural_data_dict.items():
        all_participants_neural_rsa_matrix[participant_id] = {'train_corr_matrix':[], 'test_corr_matrix':[], 'r_scan_splits':[]}
        # First make a matrix on all of the data for visualization purposes. # TODO: should this be on splits? Or is it fine to use all of it for visualization?
        if mean_subtract:
            scalar_transform = StandardScaler(with_std=with_std).fit(participant_info['voxel_data'].mean(2))
            all_voxel_activations = scalar_transform.transform(participant_info['voxel_data'].mean(2))
        else:
            all_voxel_activations = participant_info['voxel_data'].mean(2)

        # TODO: should these be noise corrected? Not sure.
        all_participants_neural_rsa_matrix[participant_id]['all_data_corr_mat'] = run_correlation_on_feature_matrix(all_voxel_activations, corr_type=corr_type_matrix)

        if noise_correction:
            raise NotImplementedError('THIS NOISE CORRECTION FOR RSA ANALYSIS MAY NOT BE THE CORRECT IMPLEMENTATION FOR WHAT YOU NEED. DID NOT IMPLEMENT FOR FULL MATRIX YET')

        for split_idx in range(n_CV_splits):
            is_train_data = all_splits_train_test_bool[split_idx]['train']
            is_test_data = all_splits_train_test_bool[split_idx]['test']

            split_train_data = participant_info['voxel_data'][is_train_data, :, :]
            split_test_data = participant_info['voxel_data'][is_test_data, :, :]

            if mean_subtract:
                scaler_train_transform = StandardScaler(with_std=with_std).fit(split_train_data.mean(2)) # fit demeaning transform on train data only
                split_train_data_avg = scaler_train_transform.transform(split_train_data.mean(2)) # demeans column-wise
                if split_test_data.shape[0]!=0:
                    split_test_data_avg = scaler_train_transform.transform(split_test_data.mean(2))
            else:
                split_train_data_avg = split_train_data.mean(2)
                split_test_data_avg = split_test_data.mean(2)

            # Average across three runs for the correlation matrix.
            train_split_corr_mat = run_correlation_on_feature_matrix(split_train_data_avg, corr_type=corr_type_matrix)
            all_participants_neural_rsa_matrix[participant_id]['train_corr_matrix'].append(train_split_corr_mat)
            if split_test_data.shape[0]!=0:
                test_split_corr_mat = run_correlation_on_feature_matrix(split_test_data_avg, corr_type=corr_type_matrix)
                all_participants_neural_rsa_matrix[participant_id]['test_corr_matrix'].append(test_split_corr_mat)
            else:
                all_participants_neural_rsa_matrix[participant_id]['test_corr_matrix'].append(np.nan)

            if noise_correction:
                print('**WARNING -- THIS NOISE CORRECTION FOR RSA ANALYSIS MAY NOT BE THE CORRECT IMPLEMENTATION FOR WHAT YOU NEED**')
                r_scan_splits = {'train':[], 'test':[]}
                for scan_split_idx, scan_split in enumerate([[0, 1], [1, 2], [0, 2]]):
                    scan_split_data_1_train = split_train_data[:, :, scan_split[0]]
                    scan_split_data_2_train = split_train_data[:, :, scan_split[1]]
    
                    if split_test_data.shape[0]!=0:
                        scan_split_data_1_test = split_test_data[:, :, scan_split[0]]
                        scan_split_data_2_test = split_test_data[:, :, scan_split[1]]
    
                    if mean_subtract:
                        scaler_train_transform_1 = StandardScaler(with_std=with_std).fit(scan_split_data_1_train) # fit demeaning transform on train data only
                        scan_split_data_1_train = scaler_train_transform_1.transform(scan_split_data_1_train) # demeans column-wise
                        if split_test_data.shape[0]!=0:
                            scan_split_data_1_test = scaler_train_transform_1.transform(scan_split_data_1_test)
    
                        scaler_train_transform_2 = StandardScaler(with_std=with_std).fit(scan_split_data_2_train) # fit demeaning transform on train data only
                        scan_split_data_2_train = scaler_train_transform_2.transform(scan_split_data_2_train) # demeans column-wise
                        if split_test_data.shape[0]!=0:
                            scan_split_data_2_test = scaler_train_transform_2.transform(scan_split_data_2_test)
    
                    scan_split_corr_mat_train_1 = run_correlation_on_feature_matrix(split_train_data[:, :, scan_split[0]], corr_type=corr_type_matrix)
                    scan_split_corr_mat_train_2 = run_correlation_on_feature_matrix(split_train_data[:, :, scan_split[1]], corr_type=corr_type_matrix)
    
                    if split_test_data.shape[0]!=0:
                        scan_split_corr_mat_test_1 = run_correlation_on_feature_matrix(split_test_data[:, :, scan_split[0]])
                        scan_split_corr_mat_test_2 = run_correlation_on_feature_matrix(split_test_data[:, :, scan_split[1]])
    
                    split_correction_train = correlate_two_matrices_rsa(scan_split_corr_mat_train_1, scan_split_corr_mat_train_2, distance_measure=distance_measure)
    
                    if split_test_data.shape[0]!=0:
                        split_correction_test = correlate_two_matrices_rsa(scan_split_corr_mat_test_1, scan_split_corr_mat_test_2, distance_measure=distance_measure)
                    else:
                        split_correction_test = np.nan
    
                    r_scan_splits['train'].append(split_correction_train)
                    r_scan_splits['test'].append(split_correction_test)
    
                all_participants_neural_rsa_matrix[participant_id]['r_scan_splits'].append(r_scan_splits)

    # Only do this once.
    if n_CV_splits>1:
        for participant_id in participant_neural_data_dict.keys():
            all_participants_neural_rsa_matrix[participant_id]['leave_one_out_neural_r_cv_test_splits'] = []
            for split_idx in range(n_CV_splits):
                this_participant = all_participants_neural_rsa_matrix[participant_id]['test_corr_matrix'][split_idx]
                other_participants = np.array([all_participants_neural_rsa_matrix[p]['test_corr_matrix'][split_idx] for p in participant_neural_data_dict.keys() if p!=participant_id])
                other_participants_avg = other_participants.mean(0)
                all_participants_neural_rsa_matrix[participant_id]['leave_one_out_neural_r_cv_test_splits'].append(correlate_two_matrices_rsa(this_participant,
                                                                     other_participants_avg,
                                                                     ))
    else:
        for participant_id in participant_neural_data_dict.keys():
            all_participants_neural_rsa_matrix[participant_id]['leave_one_out_neural_r_cv_test_splits'] = np.nan
    
    return all_participants_neural_rsa_matrix

def get_train_test_splits_bool(n_CV_splits, n_for_train, n_for_test, n_stim):
    # Randomly pick train/test indices. Reuse for model layers and participants.
    all_splits_train_test_bool = []
    for split_idx in range(n_CV_splits):
        train_data_idxs = np.random.choice(n_stim, size=n_for_train, replace=False)
        set_of_possible_test_idxs = set(np.arange(n_stim)) - set(train_data_idxs)
        if n_for_test!= 0:
            test_data_idxs = np.random.choice(list(set_of_possible_test_idxs), size=n_for_test, replace=False)
        else:
            test_data_idxs = []
        is_train_data, is_test_data = np.zeros((n_stim), dtype=bool), np.zeros((n_stim), dtype=bool)
        is_train_data[train_data_idxs], is_test_data[test_data_idxs] = True, True
        all_splits_train_test_bool.append({'train':is_train_data.copy(),
                                           'test': is_test_data.copy()})
    return all_splits_train_test_bool

def rsa_comparison_neural_roi(roi_names=['Primary', 'Lateral', 'Posterior', 'Anterior'], save_name_base=None, mean_subtract=False, with_std=False, reorder_sounds=True, corr_type_matrix='pearson', distance_measure='spearman', extra_title_str='', noise_correction=False, target='NH2015', climits=[0,1.5], with_tick_labels=False):
    if save_name_base is not None:
        analysis_folder_name = os.path.join(save_name_base, target, "%s_corr-%s_distance-%s_meansub-%s_withstd-%s"%('fmri-data-comparison', corr_type_matrix, distance_measure, mean_subtract, with_std))
        os.makedirs(analysis_folder_name, exist_ok=True)

    ## Setup splits ##
    np.random.seed(0)
    n_CV_splits = 1
    n_for_train = 165 # Here we don't need to choose a "best" layer so we can use all of the sounds for constructing the RDM. 
    n_for_test = 0
    n_stim = 165

    all_splits_train_test_bool = get_train_test_splits_bool(n_CV_splits, n_for_train, n_for_test, n_stim)

    all_participants_neural_rsa_matrix = {}
    for roi_name in roi_names:
        # Measure the train and test correlation matrix for each participant for each split
        participant_neural_data_dict = get_neural_data_matrix(roi_name=roi_name, data_by_participant=True, reorder_sounds=reorder_sounds, target=target)
        all_participants_neural_rsa_matrix[roi_name] = get_all_participants_correlation_matrices(participant_neural_data_dict,
                                                                                                 all_splits_train_test_bool,
                                                                                                 mean_subtract,
                                                                                                 with_std,
                                                                                                 corr_type_matrix,
                                                                                                 distance_measure,
                                                                                                 noise_correction,
                                                                                                 n_CV_splits)

    save_all_participant_info = {}

    for participant_idx in all_participants_neural_rsa_matrix[roi_names[0]].keys():
        save_all_participant_info[participant_idx] = {'all_roi_correlations_training':[], 'all_roi_correlations_training_corrected':[]}
        for split_idx in range(n_CV_splits):
            # TODO: how to handle singular matrix?
            all_roi_training_correlations = {roi_name_2:{roi_name_1:correlate_two_matrices_rsa(all_participants_neural_rsa_matrix[roi_name_1][participant_idx]['train_corr_matrix'][split_idx],
                                                                         all_participants_neural_rsa_matrix[roi_name_2][participant_idx]['train_corr_matrix'][split_idx],
                                                                          distance_measure=distance_measure)
                                                       for roi_name_1 in roi_names} for roi_name_2 in roi_names}
            if noise_correction:
                all_roi_training_corrected = [[noise_correct_two_roi_from_scan_splits(true_r, 
                                                                                      all_participants_neural_rsa_matrix[roi_name_1][participant_idx]['r_scan_splits'][split_idx]['train'],
                                                                                      all_participants_neural_rsa_matrix[roi_name_2][participant_idx]['r_scan_splits'][split_idx]['train'],
                                                                                      n_for_train) for roi_name_1, true_r in x.items()] for roi_name_2, x in all_roi_training_correlations.items()]
                save_all_participant_info[participant_idx]['all_roi_correlations_training_corrected'].append(all_roi_training_corrected)
    
            all_roi_training_correlations = [[true_r for roi_name_1, true_r in x.items()] for roi_name_2, x in all_roi_training_correlations.items()]

            save_all_participant_info[participant_idx]['all_roi_correlations_training'].append(all_roi_training_correlations)

    # Make a figure with the correlation values between each ROI
    if len(roi_names)>1:
        plt.figure()
        if noise_correction:
            average_participants_roi_correlation = np.concatenate([np.array(p['all_roi_correlations_training_corrected']) for p_idx, p in save_all_participant_info.items()]).mean(0)
            plot_title_str = extra_title_str + '\n Noise Corrected ROI Correlations'
        else:
            average_participants_roi_correlation = np.concatenate([np.array(p['all_roi_correlations_training']) for p_idx, p in save_all_participant_info.items()]).mean(0)
            plot_title_str = extra_title_str + '\n ROI Correlations'
        plt.imshow(average_participants_roi_correlation, vmin=0, vmax=1)
        plt.xticks(np.arange(len(roi_names)), labels=roi_names)
        plt.yticks(np.arange(len(roi_names)), labels=roi_names)
        plt.colorbar()
        plt.title(plot_title_str)
        if save_name_base is not None:
            plt.savefig(os.path.join(analysis_folder_name, 'roi_correlations.pdf'))
    
        plt.show()

    plt.figure(figsize=(6,6*len(roi_names)))
    for roi_idx, roi_name_1 in enumerate(roi_names):
        plt.subplot(len(roi_names),1,roi_idx+1)
        concat_roi_mat = np.concatenate([np.array(all_participants_neural_rsa_matrix[roi_name_1][participant_idx]['train_corr_matrix']) for participant_idx in save_all_participant_info.keys()])
        avg_corr_matrix = concat_roi_mat.mean(0)
        plot_correlation_matrix_with_color_categories(1-avg_corr_matrix, climits=climits, with_tick_labels=with_tick_labels)
        plt.title(extra_title_str + '\n Neural RDM (Average Across Participants) \n ROI=%s'%roi_name_1)

    if save_name_base is not None:
        if with_tick_labels:
            plt.savefig(os.path.join(analysis_folder_name, '-'.join(roi_names) + '_with_sound_names_individual_roi_neural_rdms.pdf'))
        else:
            plt.savefig(os.path.join(analysis_folder_name, '-'.join(roi_names) + '_individual_roi_neural_rdms.pdf'))

def rsa_cross_validated_all_models(randnetw='False', save_name_base=None, roi_name=None, mean_subtract=False, with_std=False, noise_correction=False, target='NH2015', ignore_layer_keys=None):
    rsa_analysis_dict = {}
    # TODO: handle all zero rows better. 
    for model, layers in d_layer_reindex.items():
        if (model in ['Kell2018init', 'ResNet50init', 'wav2vecpower']) and randnetw=='True': # These models don't have random activations saved.
            continue
        elif (model in ['wav2vecpower', 'ResNet50init', 'Kell2018init']):
            continue
        print('Analyzing model %s'%model)
        if model == 'spectemp': # No random model for spectemmp
            rsa_analysis_dict[model] = run_cross_validated_rsa_to_choose_best_layer(model, randnetw='False', save_name_base=save_name_base, roi_name=roi_name, mean_subtract=mean_subtract, with_std=with_std, corr_type_matrix='pearson', distance_measure='spearman', noise_correction=noise_correction, target=target, ignore_layer_keys=ignore_layer_keys) # TODO: add other options
        else:
            rsa_analysis_dict[model] = run_cross_validated_rsa_to_choose_best_layer(model, randnetw=randnetw, save_name_base=save_name_base, roi_name=roi_name, mean_subtract=mean_subtract, with_std=with_std, corr_type_matrix='pearson', distance_measure='spearman', noise_correction=noise_correction, target=target, ignore_layer_keys=ignore_layer_keys) # TODO: add other options
     
    return rsa_analysis_dict

def check_num_sounds_with_all_same_activations(model_name, randnetw='False', mean_subtract=False, with_std=False):
    all_model_layers_count_sound_all_same = {}
    for model_layer in d_layer_reindex[model_name]:
        all_model_features=get_source_features(model_name, model_layer, randnetw=randnetw, stimuli_IDs=stimuli_IDs)

        # Get a matrix using all of the data.
        if mean_subtract:
            scalar_all_data = StandardScaler(with_std=with_std).fit(all_model_features)
            all_network_activation = scalar_all_data.transform(all_model_features)
        else:
            all_network_activation = all_model_features
        all_model_layers_count_sound_all_same[model_layer] = sum(all_network_activation.std(1)==0)

    return all_model_layers_count_sound_all_same

def count_dead_units_all_models(randnetw='False', mean_subtract=False, with_std=False):
    num_sounds_with_all_same_activations = {}
    for model, layers in d_layer_reindex.items():
        if (model in ['Kell2018init', 'ResNet50init', 'wav2vecpower','spectemp']) and randnetw=='True': # These models don't have random activations saved.
            continue
        if (model in ['wav2vecpower', 'ResNet50init', 'Kell2018init']):
            continue
        print('Analyzing model %s'%model)
        num_sounds_with_all_same_activations[model] = check_num_sounds_with_all_same_activations(model, randnetw=randnetw, mean_subtract=mean_subtract, with_std=with_std)
    return num_sounds_with_all_same_activations

def run_cross_validated_rsa_to_choose_best_layer(model_name, randnetw='False', save_name_base=None, roi_name=None, corr_type_matrix='pearson', distance_measure='spearman', mean_subtract=False, with_std=False, reorder_sounds=True, noise_correction=False, target='NH2015', ignore_layer_keys=None):

    if save_name_base is not None:
        analysis_folder_name = os.path.join(save_name_base, target, "%s_rand-%s_roi-%s_corr-%s_distance-%s_meansub-%s_withstd-%s-noisecorrect-%s"%(model_name, randnetw, roi_name, corr_type_matrix, distance_measure, mean_subtract, with_std, noise_correction))
        os.makedirs(analysis_folder_name, exist_ok=True)

    ## Setup splits ##
    np.random.seed(0)
    n_CV_splits = 10
    n_for_train = 83
    n_for_test = 82
    n_stim = 165

    # Randomly pick train/test indices. Reuse for model layers and participants.
    all_splits_train_test_bool = get_train_test_splits_bool(n_CV_splits, n_for_train, n_for_test, n_stim)

    print('Measuring the correlation matrix for each participant')
    # Measure the train and test correlation matrix for each participant for each split
    participant_neural_data_dict = get_neural_data_matrix(roi_name=roi_name, data_by_participant=True, reorder_sounds=reorder_sounds, target=target)
    all_participants_neural_rsa_matrix = get_all_participants_correlation_matrices(participant_neural_data_dict, all_splits_train_test_bool, mean_subtract, with_std, corr_type_matrix, distance_measure, noise_correction, n_CV_splits)

    print('Measuring the correlation matrix for each model and each layer')
    # For each layer of the model, measure the correlation matrix for the train and test splits.
    all_model_layers_rsa_matrix = {}
    activations_all_model_layer_concat = []
    activations_layers_order = []
    for model_layer_idx, model_layer in enumerate(d_layer_reindex[model_name]):
        if ignore_layer_keys is not None:
            if model_layer in ignore_layer_keys: # skip layers if requested. 
                continue

        activations_layers_order.append(model_layer)

        all_model_features=get_source_features(model_name, model_layer, randnetw=randnetw, stimuli_IDs=stimuli_IDs)
        if reorder_sounds:
            all_model_features = all_model_features[correlation_plot_order,:]

        activations_all_model_layer_concat.append(all_model_features)
        if model_layer_idx==(len(d_layer_reindex[model_name])-1):
            activations_all_model_layer = np.concatenate(activations_all_model_layer_concat, 1)
            print(activations_all_model_layer.shape)
            if mean_subtract:
                scalar_all_data_all_layer = StandardScaler(with_std=with_std).fit(activations_all_model_layer)
                all_network_activation_all_layer = scalar_all_data_all_layer.transform(activations_all_model_layer)
            else:
                all_network_activation_all_layer = activations_all_model_layer
            corr_all_data_all_layer = run_correlation_on_feature_matrix(all_network_activation_all_layer, corr_type=corr_type_matrix)
            all_model_layers_rsa_matrix['all_features_concat'] = {'train_corr_matrix':[], 'test_corr_matrix':[]}
            all_model_layers_rsa_matrix['all_features_concat']['all_data_corr_mat'] = corr_all_data_all_layer

        all_model_layers_rsa_matrix[model_layer] = {'train_corr_matrix':[], 'test_corr_matrix':[]}

        # Get a matrix using all of the data. 
        if mean_subtract:
            scalar_all_data = StandardScaler(with_std=with_std).fit(all_model_features)
            all_network_activation = scalar_all_data.transform(all_model_features)
        else:
            all_network_activation = all_model_features
        corr_all_data = run_correlation_on_feature_matrix(all_network_activation, corr_type=corr_type_matrix)
        all_model_layers_rsa_matrix[model_layer]['all_data_corr_mat'] = corr_all_data

        for split_idx in range(n_CV_splits):
            is_train_data = all_splits_train_test_bool[split_idx]['train']
            is_test_data= all_splits_train_test_bool[split_idx]['test']

            split_train_data = all_model_features[is_train_data, :]
            split_test_data = all_model_features[is_test_data, :]
            if mean_subtract:
                scaler_train_transform = StandardScaler(with_std=with_std).fit(split_train_data) # fit demeaning transform on train data only
                split_train_data = scaler_train_transform.transform(split_train_data) # demeans column-wise
                split_test_data = scaler_train_transform.transform(split_test_data)

            split_model_train = run_correlation_on_feature_matrix(split_train_data, corr_type=corr_type_matrix)
            split_model_test = run_correlation_on_feature_matrix(split_test_data, corr_type=corr_type_matrix)
            all_model_layers_rsa_matrix[model_layer]['train_corr_matrix'].append(split_model_train)
            all_model_layers_rsa_matrix[model_layer]['test_corr_matrix'].append(split_model_test)

            if model_layer_idx==(len(d_layer_reindex[model_name])-1):
                split_train_data_all = activations_all_model_layer[is_train_data, :]
                split_test_data_all = activations_all_model_layer[is_test_data, :]
                if mean_subtract:
                    scaler_train_transform_all = StandardScaler(with_std=with_std).fit(split_train_data_all) # fit demeaning transform on train data only
                    split_train_data_all = scaler_train_transform_all.transform(split_train_data_all) # demeans column-wise
                    split_test_data_all = scaler_train_transform_all.transform(split_test_data_all)
    
                split_model_train_all = run_correlation_on_feature_matrix(split_train_data_all, corr_type=corr_type_matrix)
                split_model_test_all = run_correlation_on_feature_matrix(split_test_data_all, corr_type=corr_type_matrix)
                all_model_layers_rsa_matrix['all_features_concat']['train_corr_matrix'].append(split_model_train_all)
                all_model_layers_rsa_matrix['all_features_concat']['test_corr_matrix'].append(split_model_test_all)


    model_layers = activations_layers_order # d_layer_reindex[model_name]

#     if ignore_layer_keys is not None:
#         for l in ignore_layer_keys:
#             try:
#                 model_layers.remove(l)
#             except ValueError:
#                 pass

    print('Measuring the distance between the participant and model correlation matrix, and choosing the best layer')
    # Now, choose the best layer based on the training split of the data and save the R^2 value for the test matrix. 
    # Do this for each split. 
    save_all_participant_info = {}
    for participant_idx, participant_correlations in all_participants_neural_rsa_matrix.items():
        save_all_participant_info[participant_idx] = {}
        # Run using the whole correlation matrix -- this is used for the argmax analysis (because we aren't comparing the values)
        

        for split_idx in range(n_CV_splits):
            # TODO: how to handle singular matrix? 
            all_layer_training_correlations = [correlate_two_matrices_rsa(all_model_layers_rsa_matrix[model_layer]['train_corr_matrix'][split_idx],
                                                                          participant_correlations['train_corr_matrix'][split_idx],
                                                                          distance_measure=distance_measure) 
                                                       for model_layer in model_layers]

            if noise_correction:
                training_split_r  = participant_correlations['r_scan_splits'][split_idx]['train']
                all_layer_training_correlations_corrected = [noise_correct_from_scan_splits(true_r, training_split_r, n_for_train) for true_r in all_layer_training_correlations]

            all_layer_test_correlations= [correlate_two_matrices_rsa(all_model_layers_rsa_matrix[model_layer]['test_corr_matrix'][split_idx],
                                                                          participant_correlations['test_corr_matrix'][split_idx],
                                                                          distance_measure=distance_measure)
                                                       for model_layer in model_layers]
            if noise_correction:
                test_split_r  = participant_correlations['r_scan_splits'][split_idx]['test']
                all_layer_test_correlations_corrected = [noise_correct_from_scan_splits(true_r, test_split_r, n_for_test) for true_r in all_layer_test_correlations]

            if noise_correction:
                best_layer_idx = np.nanargmax(all_layer_training_correlations_corrected)
                best_layer_test_value = all_layer_test_correlations_corrected[best_layer_idx]
            else:
                best_layer_idx = np.nanargmax(all_layer_training_correlations)
                best_layer_test_value = all_layer_test_correlations[best_layer_idx]

            save_all_participant_info[participant_idx]['split_%d'%split_idx] = {'best_layer_test_value':best_layer_test_value,
                                                                     'best_layer_idx': best_layer_idx, 
                                                                     'best_layer_name': model_layers[best_layer_idx],
                                                                     'all_layer_test_correlations':all_layer_test_correlations,
                                                                     }

        participant_rsa_value = np.median([save_all_participant_info[participant_idx]['split_%d'%split_idx]['best_layer_test_value'] for split_idx in range(n_CV_splits)])
        save_all_participant_info[participant_idx]['best_layer_rsa_median_across_splits'] = participant_rsa_value
        save_all_participant_info[participant_idx]['splits_median_distance_all_layers'] = np.median(np.array([save_all_participant_info[participant_idx]['split_%d'%split_idx]['all_layer_test_correlations'] for split_idx in range(n_CV_splits)]),0)
        save_all_participant_info[participant_idx]['splits_median_best_layer_idx'] = np.median([save_all_participant_info[participant_idx]['split_%d'%split_idx]['best_layer_idx'] for split_idx in range(n_CV_splits)])
        save_all_participant_info[participant_idx]['all_data_corr_mat_fmri'] = all_participants_neural_rsa_matrix[participant_idx]['all_data_corr_mat'] 
        save_all_participant_info[participant_idx]['leave_one_out_neural_r_cv_test_splits'] = all_participants_neural_rsa_matrix[participant_idx]['leave_one_out_neural_r_cv_test_splits']

        # Get correlations from all 165 sounds -- used for the argmax where we don't need to cross-validate to choose best layer (because we are not reporting the values)
        save_all_participant_info[participant_idx]['all_data_rsa_for_best_layer'] = [correlate_two_matrices_rsa(all_model_layers_rsa_matrix[model_layer]['all_data_corr_mat'],
                                                                                                               all_participants_neural_rsa_matrix[participant_idx]['all_data_corr_mat'],
                                                                                                               distance_measure=distance_measure) for model_layer in model_layers]

        # Get correlation with the all-layer activation data. 
        all_model_features_rsa = correlate_two_matrices_rsa(all_model_layers_rsa_matrix['all_features_concat']['all_data_corr_mat'], 
                                                            all_participants_neural_rsa_matrix[participant_idx]['all_data_corr_mat'], 
                                                            distance_measure=distance_measure)
        if noise_correction:
            raise NotImplementedError
        save_all_participant_info[participant_idx]['all_model_features_rsa'] = all_model_features_rsa

        save_all_participant_info[participant_idx]['model_layers'] = model_layers

    clims_rdm = [None, None]
    # Make visuals of the correlation matrix
    if save_name_base is not None:
        num_model_layers = len(model_layers)
        num_subplot_columns = max(2, num_model_layers)
        plt.figure(figsize=(3*num_subplot_columns, 9)) # human in first row, then a row for each matrix of the model. 
        # Average the correlation matrix across participants for visualization? 
        plt.subplot(3, num_subplot_columns, 1)
        neural_correlation_matrix = np.array([all_participants_neural_rsa_matrix[participant_idx]['all_data_corr_mat'] for participant_idx in all_participants_neural_rsa_matrix.keys()]).mean(0) # average across participants. 
        plot_correlation_matrix_with_color_categories(1-neural_correlation_matrix, climits=clims_rdm)
        plt.title('fMRI RDM averaged across participants')
    
        average_scores_layers = np.array([save_all_participant_info[participant_idx]['splits_median_distance_all_layers'] for participant_idx in all_participants_neural_rsa_matrix.keys()]).mean(0)
        median_best_layer = np.median([save_all_participant_info[participant_idx]['splits_median_best_layer_idx'] for participant_idx in all_participants_neural_rsa_matrix.keys()])
        best_layer_plots = []
        # Show correlation matrix for all layers of the model. 
        for layer_idx, layer in enumerate(model_layers):
            plt.subplot(3, num_subplot_columns, num_subplot_columns + layer_idx + 1)
            plot_correlation_matrix_with_color_categories(1-all_model_layers_rsa_matrix[layer]['all_data_corr_mat'], climits=clims_rdm)
            layer_plot_title = "Layer: %s \n Average Score %f"%(layer, average_scores_layers[layer_idx])
            if (np.floor(median_best_layer) == (layer_idx)) or (np.ceil(median_best_layer) == (layer_idx)):
                plt.title('**Median Best Layer** \n' + layer_plot_title)
                best_layer_plots.append(layer)
            else:
                plt.title(layer_plot_title)

        # Add a plot that is the cross-validated R^2 for each layer
        plt.subplot(3, num_subplot_columns, num_subplot_columns*2 + 1)
        across_layer_scores = np.array([save_all_participant_info[participant_idx]['splits_median_distance_all_layers'] for participant_idx in all_participants_neural_rsa_matrix.keys()])
        across_layer_mean = across_layer_scores.mean(0)
        across_layer_mean_sub_participant = across_layer_scores - np.mean(across_layer_scores, 1, keepdims=True)
        across_layer_sem = np.std(across_layer_mean_sub_participant, 0) / np.sqrt(across_layer_mean_sub_participant.shape[0]-1)
        plt.errorbar(np.arange(len(across_layer_mean)), across_layer_mean, yerr=across_layer_sem)
        plt.xlabel('Layer')
        plt.xticks(np.arange(len(across_layer_mean)), labels=model_layers, rotation=90)
        plt.ylabel('Human-Model Similarity (RSA)')
        plt.title(model_name)

        # Add a plot that is the full-data R^2 for each layer (since we aren't propagating the splits anyway). 
        plt.subplot(3, num_subplot_columns, num_subplot_columns*2 + 2)
        across_layer_scores = np.array([save_all_participant_info[participant_idx]['all_data_rsa_for_best_layer'] for participant_idx in all_participants_neural_rsa_matrix.keys()])
        across_layer_mean = across_layer_scores.mean(0)
        across_layer_mean_sub_participant = across_layer_scores - np.mean(across_layer_scores, 1, keepdims=True)
        across_layer_sem = np.std(across_layer_mean_sub_participant, 0) / np.sqrt(across_layer_mean_sub_participant.shape[0]-1)
        plt.errorbar(np.arange(len(across_layer_mean)), across_layer_mean, yerr=across_layer_sem)
        plt.xlabel('Layer')
        plt.xticks(np.arange(len(across_layer_mean)), labels=model_layers, rotation=90)
        plt.ylabel('Human-Model Similarity (RSA) \n All 165 Sounds In Correlation')
        plt.title(model_name)
  
        plt.savefig(os.path.join(analysis_folder_name, 'full_data_correlation_matrix.pdf'))
        plt.show()

        plt.figure(figsize=(10*len(best_layer_plots), 10)) # larger so that we can list the names of the sounds
        plt.subplot(1,1+len(best_layer_plots),1)
        plot_correlation_matrix_with_color_categories(1-neural_correlation_matrix, with_tick_labels=True, climits=clims_rdm)
        plt.title('Full RDM fMRI Data')

        for layer_idx, layer in enumerate(best_layer_plots):
            plt.subplot(1,1+len(best_layer_plots),2+layer_idx)
            plot_correlation_matrix_with_color_categories(1-all_model_layers_rsa_matrix[layer]['all_data_corr_mat'], with_tick_labels=True, climits=clims_rdm)
            plt.title('Full RDM Layer %s, Model %s'%(layer, model_name))

        plt.savefig(os.path.join(analysis_folder_name, 'best_layer_correlation_matrix_with_labels.pdf'))
        plt.show()

    return save_all_participant_info

def plot_correlation_matrix_with_color_categories(correlation_matrix, with_tick_labels=False, linewidth=25, climits=[None,None]):
    stimuli_id_ticks = [stimuli_IDs[s_idx] for s_idx in correlation_plot_order]

#     plt.imshow(correlation_matrix,cmap='Greys', origin='lower')
    for category in sound_category_order:
        sound_idx_in_category = [c_idx for c_idx, c in enumerate(category_color_order) if c==category]
#         plt.plot([-3, -3], [sound_idx_in_category[0]-0.5, sound_idx_in_category[-1]+0.5], color=d_sound_category_colors[category], linewidth=linewidth, solid_capstyle="butt", zorder=0)
#         plt.plot([sound_idx_in_category[0]-0.5, sound_idx_in_category[-1]+0.5], [len(category_color_order) + 2, len(category_color_order) + 2], color=d_sound_category_colors[category], linewidth=linewidth, solid_capstyle="butt", zorder=0)

        plt.plot([-6, -6], [sound_idx_in_category[0], sound_idx_in_category[-1]+1], color=d_sound_category_colors[category], linewidth=linewidth, solid_capstyle="butt", zorder=0)
        plt.plot([sound_idx_in_category[0], sound_idx_in_category[-1]+1], [len(category_color_order) + 5, len(category_color_order) + 5], color=d_sound_category_colors[category], linewidth=linewidth, solid_capstyle="butt", zorder=0)

#     plt.imshow(correlation_matrix,cmap='Greys', origin='lower', zorder=100, vmin=climits[0], vmax=climits[1], interpolation='none', rasterized=True)
    plt.pcolormesh(correlation_matrix,cmap='Greys', zorder=100, vmin=climits[0], vmax=climits[1])
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.gca().set_aspect(1)
    plt.gca().invert_yaxis()

#     plt.xlim([-3.5, correlation_matrix.shape[0]-0.5])
#     plt.ylim([correlation_matrix.shape[0]+2.5, -0.5])
    plt.xlim([-4, correlation_matrix.shape[0]])
    plt.ylim([correlation_matrix.shape[0]+4, 0])
    if with_tick_labels:
        plt.xticks(np.arange(correlation_matrix.shape[0])+0.5, stimuli_id_ticks, rotation=90, fontsize=2)
        plt.yticks(np.arange(correlation_matrix.shape[1])+0.5, stimuli_id_ticks, fontsize=2)
    else:
        plt.axis('off')

def plot_ordered_cross_val_RSA(rsa_analysis_dict, model_ordering=None, alpha=1, extra_title_str='', save_fig_path=None, use_165_sounds_for_fMRI_ceiling=True):
    participant_ids = np.sort(list(rsa_analysis_dict['Kell2018music'].keys()))
    leave_one_out_neural_r = None
    all_model_names = []
    model_participant_matrix = []

    for model_name, model_info in rsa_analysis_dict.items():
        all_model_names.append(model_name)
        model_participant_matrix.append([model_info[p]['best_layer_rsa_median_across_splits'] for p in participant_ids])

        # Only do this once. 
        if (leave_one_out_neural_r is None) and use_165_sounds_for_fMRI_ceiling:
            leave_one_out_neural_r=[]
            for participant_id in participant_ids:
                this_participant = model_info[participant_id]['all_data_corr_mat_fmri']
                other_participants = np.array([model_info[p]['all_data_corr_mat_fmri'] for p in participant_ids if p!=participant_id])
                other_participants_avg = other_participants.mean(0)
                leave_one_out_neural_r.append(correlate_two_matrices_rsa(this_participant, 
                                                                         other_participants_avg, 
                                                                         ))
        elif (leave_one_out_neural_r is None) and (not use_165_sounds_for_fMRI_ceiling):
            participant_split_medians = np.median(np.array([model_info[p]['leave_one_out_neural_r_cv_test_splits'] for p in participant_ids]), 1)
            assert(len(participant_split_medians) == len(participant_ids))
            leave_one_out_neural_r = participant_split_medians

    model_participant_matrix = np.array(model_participant_matrix)

    mean_subtract_models = model_participant_matrix - np.nanmean(model_participant_matrix, 0)
    y_err_within_participant = np.nanstd(mean_subtract_models, 1) / np.sqrt(mean_subtract_models.shape[0] - 1)

    mean_values = np.mean(model_participant_matrix, 1)

    df_grouped = pd.DataFrame({'mean_values': mean_values,
                               'y_err': y_err_within_participant, 
                               'source_model': all_model_names},
                              )
    
    df_grouped_w_spectemp = df_grouped.copy(deep=True)

    # drop the spectemp row (we want to plot it separately)
    df_spectemp = df_grouped.loc[df_grouped['source_model'] == 'spectemp']
    df_grouped.drop(df_grouped.index[df_grouped['source_model'] == 'spectemp'], inplace=True)
    
    if model_ordering is None:
        df_grouped = df_grouped.sort_values('mean_values', ascending=False)
    else:
        df_grouped = df_grouped.set_index('source_model', inplace=False, drop=False)
        df_grouped = df_grouped.reindex(model_ordering)

    # plot specs
    bar_placement = np.arange(0, len(df_grouped) / 2, 0.5)
    color_order = [d_model_colors[x] for x in df_grouped.source_model]

    # plot specs
    bar_placement = np.arange(0, len(df_grouped) / 2, 0.5)
    color_order = [d_model_colors[x] for x in df_grouped.source_model]
    model_legend = [d_model_names[x] for x in df_grouped.source_model]

    title_str = extra_title_str + 'Human-Model RSA'

    # Obtain xmin and xmax for spectemp line
    xmin = np.unique((bar_placement[0] - np.diff(bar_placement) / 2))
    xmax = np.unique((bar_placement[-1] + np.diff(bar_placement) / 2))

    fig, ax = plt.subplots(figsize=(6, 7.5))
    ax.set_box_aspect(0.8)

    # Plot the neural data comparison
    ax.hlines(xmin=xmin, xmax=xmax, y=np.mean(leave_one_out_neural_r), color='black', linestyle='--',
              zorder=2)
    plt.fill_between(
        [(bar_placement[0] - np.diff(bar_placement) / 2)[0],
         (bar_placement[-1] + np.diff(bar_placement) / 2)[0]],
        np.mean(leave_one_out_neural_r) - np.std(leave_one_out_neural_r)/np.sqrt(len(leave_one_out_neural_r)-1),
        np.mean(leave_one_out_neural_r) + np.std(leave_one_out_neural_r)/np.sqrt(len(leave_one_out_neural_r)-1),
        color='grey')

    ax.hlines(xmin=xmin, xmax=xmax, y=df_spectemp['mean_values'].values, color='darkgrey',
              zorder=2)
    plt.fill_between(
        [(bar_placement[0] - np.diff(bar_placement) / 2)[0],
         (bar_placement[-1] + np.diff(bar_placement) / 2)[0]],
        df_spectemp['mean_values'].values - df_spectemp['y_err'].values,
        # plot yerr for spectemp too
        df_spectemp['mean_values'].values + df_spectemp['y_err'].values,
        color='gainsboro')
    ax.bar(bar_placement, df_grouped['mean_values'].values,
           yerr=df_grouped['y_err'].values,
           width=0.3, color=color_order, zorder=2, alpha=alpha)
    plt.xticks(bar_placement, model_legend, rotation=80, fontsize=13, ha='right', rotation_mode='anchor')
    plt.ylim([0, 1])
    plt.ylabel('Spearman R Human vs. Model RSA', fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(title_str)
    plt.tight_layout(pad=1)

    if save_fig_path is not None:
        plt.savefig(os.path.join(save_fig_path, title_str.replace(':','_').replace(' ','') + '.pdf'))
    if save_fig_path is not None:
        plt.savefig(os.path.join(save_fig_path, title_str.replace(':','_').replace(' ','') + '.svg'))
    fig.show()
    
    # Return the ordering
    return df_grouped.source_model.to_numpy()
        

def get_mean_and_sem_best_layer(model_participant_matrix,  all_model_names, roi_name):  
    mean_subtract_models = model_participant_matrix - np.nanmean(model_participant_matrix, 0)
    y_err_within_participant = np.nanstd(mean_subtract_models, 1) / np.sqrt(mean_subtract_models.shape[0] - 1)
    mean_values = np.mean(model_participant_matrix, 1)
    
    df_grouped = pd.DataFrame({'%s_mean_values'%roi_name: mean_values,
                               '%s_err'%roi_name: y_err_within_participant, 
                               'source_model': all_model_names},
                              )

    return df_grouped

def load_all_models_best_layer_df(pckl_path, target, roi_non_primary, permuted=True, use_all_165_sounds=True, save_fig_path=None):
    with open(pckl_path, 'rb') as f:
        best_layer_roi_data_dict = pickle.load(f)

    if permuted:
        rsa_analysis_dict_all_rois =  best_layer_roi_data_dict['rsa_analysis_dict_all_rois_permuted'][target]
    else:
        rsa_analysis_dict_all_rois = best_layer_roi_data_dict['rsa_analysis_dict_all_rois'][target]

    if use_all_165_sounds:
        rsa_key = 'all_data_rsa_for_best_layer'
    else:
        rsa_key = 'splits_median_distance_all_layers'

    combine_roi_model_dict = {}
    for roi_name, rsa_analysis_dict in rsa_analysis_dict_all_rois.items():
        combine_roi_model_dict[roi_name] = {}
        participant_ids = np.sort(list(rsa_analysis_dict['Kell2018music'].keys()))

        leave_one_out_neural_r = None
        all_model_names = []
        model_participant_matrix = []

        for model_name, model_info in rsa_analysis_dict.items():
            if model_name in ['DCASE2020', 'metricGAN', 'DS2', 'ZeroSpeech2020', 'spectemp']: # We exclude some of the models for this plot.
                continue
            all_model_names.append(model_name)
            # Check for layers that are exactly equal
            check_ties = ([len(set(model_info[p][rsa_key])) == len(model_info[p][rsa_key]) for p in participant_ids])
            if not all(check_ties):
                print('Model %s has a tie'%model_name)
            # Divide by num_layers + 1 so that the minimum layer = 0 and the max layer = 1
            best_layer_argmax_normalized = [np.nanargmax(np.array(model_info[p][rsa_key])) /
                                                         (len(model_info[p][rsa_key])-1)
                                                         for p in participant_ids]
            model_participant_matrix.append(best_layer_argmax_normalized)

        model_participant_matrix = np.stack(model_participant_matrix)
        combine_roi_model_dict[roi_name]['model_participant_matrix'] = model_participant_matrix
        combine_roi_model_dict[roi_name]['all_model_names'] = all_model_names

    df_primary = get_mean_and_sem_best_layer(combine_roi_model_dict['Primary']['model_participant_matrix'],
                                         combine_roi_model_dict['Primary']['all_model_names'],
                                            'Primary')

    # TODO: do we want to list through them? 
    for roi_idx, roi_name in enumerate([roi_non_primary]):
        df_roi = get_mean_and_sem_best_layer(combine_roi_model_dict[roi_name]['model_participant_matrix'],
                                             combine_roi_model_dict[roi_name]['all_model_names'],
                                            roi_name)
        two_roi_df = pd.merge(df_roi, df_primary, on=['source_model'])

    return two_roi_df


def plot_all_roi_rsa_scatter(rsa_analysis_dict_all_rois, extra_title_str='', use_all_165_sounds=True, save_fig_path=None):
    if use_all_165_sounds:
        rsa_key = 'all_data_rsa_for_best_layer'
    else:
        rsa_key = 'splits_median_distance_all_layers'

    combine_roi_model_dict = {}
    for roi_name, rsa_analysis_dict in rsa_analysis_dict_all_rois.items():
        combine_roi_model_dict[roi_name] = {}
        participant_ids = np.sort(list(rsa_analysis_dict['Kell2018music'].keys()))
    
        leave_one_out_neural_r = None
        all_model_names = []
        model_participant_matrix = []
    
        for model_name, model_info in rsa_analysis_dict.items():
            if model_name in ['DCASE2020', 'metricGAN', 'DS2', 'ZeroSpeech2020', 'spectemp']: # We exclude some of the models for this plot. 
                continue
            all_model_names.append(model_name)
            # Check for layers that are exactly equal
            check_ties = ([len(set(model_info[p][rsa_key])) == len(model_info[p][rsa_key]) for p in participant_ids])
            if not all(check_ties):
                print('Model %s has a tie'%model_name)
            # Divide by num_layers + 1 so that the minimum layer = 0 and the max layer = 1
            best_layer_argmax_normalized = [np.nanargmax(np.array(model_info[p][rsa_key])) / 
                                                         (len(model_info[p][rsa_key])-1) 
                                                         for p in participant_ids]
            # print('Model %s, ROI %s, %s'%(model_name, roi_name, best_layer_argmax_normalized))
            model_participant_matrix.append(best_layer_argmax_normalized)
            
        model_participant_matrix = np.stack(model_participant_matrix)
        combine_roi_model_dict[roi_name]['model_participant_matrix'] = model_participant_matrix
        combine_roi_model_dict[roi_name]['all_model_names'] = all_model_names
        
    
    df_primary = get_mean_and_sem_best_layer(combine_roi_model_dict['Primary']['model_participant_matrix'],
                                         combine_roi_model_dict['Primary']['all_model_names'],
                                            'Primary')

    plt.figure(figsize=(12,4))
    for roi_idx, roi_name in enumerate(['Anterior', 'Lateral', 'Posterior']):
        ax = plt.subplot(1,3,roi_idx+1)
        df_roi = get_mean_and_sem_best_layer(combine_roi_model_dict[roi_name]['model_participant_matrix'],
                                             combine_roi_model_dict[roi_name]['all_model_names'],
                                            roi_name)
        two_roi_df = pd.merge(df_roi, df_primary, on=['source_model'])
        for row_idx, row in two_roi_df.iterrows():
            plot_color = d_model_colors[row.source_model]
            plt.errorbar(row['Primary_mean_values'],
                         row['%s_mean_values'%roi_name],
                         xerr=row['Primary_err'],
                         yerr=row['%s_err'%roi_name],
                         color=plot_color,
                         marker='o')

        avg_best_primary = df_primary['Primary_mean_values'].mean()
        avg_best_roi = df_roi['%s_mean_values'%roi_name].mean()
        print(f'{extra_title_str} Avg across models best layer Primary: {avg_best_primary}, ROI: {avg_best_roi}')

        w, p = wilcoxon(df_primary['Primary_mean_values'], df_roi['%s_mean_values'%roi_name]) # This doesn't propagate the errors from the participants, but maybe thats fine? 
        print(f'{extra_title_str} Wilcoxon signed-rank test between Primary and {roi_name}: {w}, p-value: {p:.5f}')

        try:
            roi_w_stats = pg.wilcoxon(df_primary['Primary_mean_values'], df_roi['%s_mean_values'%roi_name], alternative='two-sided', correction=False)
            w_pg = roi_w_stats['W-val']
            p_pg = roi_w_stats['p-val']
            rbc_pg = roi_w_stats['RBC']
            cl_pg = roi_w_stats['CLES']
            print(f'EFFECT SIZE: {extra_title_str} Wilcoxon signed-rank test between Primary and {roi_name}: ')
            print(roi_w_stats)
            cohen_stats = pg.compute_effsize(df_primary['Primary_mean_values'], df_roi['%s_mean_values'%roi_name], paired=True, eftype='cohen')
            hedges_stats = pg.compute_effsize(df_primary['Primary_mean_values'], df_roi['%s_mean_values'%roi_name], paired=True, eftype='cohen')
            print(f"Cohen's d: {cohen_stats:.5f}, Hedges g: {hedges_stats:.5f}")

        except NameError:
            print('Not running effect size because pingouin is not installed')

        plt.xlabel('Primary')
        plt.ylabel(roi_name)
        plt.plot([0,1], [0,1], '--', color='grey', linewidth=0.5)
        plt.ylim([0,1])
        plt.xlim([0,1])
        plt.title(extra_title_str + '%s vs. Primary'%roi_name + f'\nWilcoxon Test {w}, p-value: {p:.5f}')
        ax.set_aspect('equal', adjustable='box')

    print('\n \n')
    if save_fig_path is not None:
        plt.savefig(os.path.join(save_fig_path, extra_title_str.replace(':','_').replace(' ','') + '_all_roi_scatter.pdf'))
        plt.savefig(os.path.join(save_fig_path, extra_title_str.replace(':','_').replace(' ','') + '_all_roi_scatter.svg'))
    plt.show()

def make_paper_plots(save_fig_path='rsa_plots'):
    # Figure 2
    make_all_voxel_rsa_bar_plots(save_fig_path=save_fig_path)

    # Figure 5 schematic (and supplement)
#     make_neural_roi_rdms(save_fig_path=save_fig_path)

    # Figure 5, and Supplement Fig
#     make_best_layer_roi_scatter_plots(save_fig_path)


def make_neural_roi_rdms(save_fig_path):
    for dataset in ['NH2015', 'B2021']:
        rsa_comparison_neural_roi(mean_subtract=True,
                                  with_std=True, 
                                  extra_title_str='Z-Score: ', 
                                  target=dataset,
                                  save_name_base=save_fig_path)

        rsa_comparison_neural_roi(mean_subtract=True,
                                  with_std=True,
                                  extra_title_str='Z-Score: ',
                                  with_tick_labels=True,
                                  target=dataset,
                                  save_name_base=save_fig_path)

        rsa_comparison_neural_roi(roi_names=["All"],
                                  mean_subtract=True,
                                  with_std=True,
                                  extra_title_str='Z-Score: ',
                                  with_tick_labels=True,
                                  target=dataset,
                                  save_name_base=save_fig_path)

def make_best_layer_roi_scatter_plots_from_pckl(pckl_path, save_fig_path):
    with open(pckl_path, 'rb') as f:
        best_layer_roi_data_dict = pickle.load(f)

    rsa_analysis_dict_all_rois = best_layer_roi_data_dict['rsa_analysis_dict_all_rois']
    rsa_analysis_dict_all_rois_permuted = best_layer_roi_data_dict['rsa_analysis_dict_all_rois_permuted']

    for dataset in ['NH2015','B2021']:
        plot_all_roi_rsa_scatter(rsa_analysis_dict_all_rois[dataset],
                                 extra_title_str = dataset + '_Trained: ',
                                 save_fig_path=save_fig_path)

        plot_all_roi_rsa_scatter(rsa_analysis_dict_all_rois_permuted[dataset],
                                 extra_title_str = dataset + '_Permuted: ',
                                 save_fig_path=save_fig_path)

def make_best_layer_roi_scatter_plots(save_fig_path):
    rsa_analysis_dict_all_rois = {}
    rsa_analysis_dict_all_rois_permuted = {}
    for dataset in ['NH2015','B2021']:
        rsa_analysis_dict_all_rois[dataset] = {}
        for ROI in ['Primary', 'Lateral', 'Posterior', 'Anterior']:
            rsa_analysis_dict = rsa_cross_validated_all_models(randnetw='False', 
                                                               roi_name=ROI,
                                                               mean_subtract=True, 
                                                               with_std=True,
                                                               target=dataset)
            rsa_analysis_dict_all_rois[dataset][ROI] = rsa_analysis_dict
            
        plot_all_roi_rsa_scatter(rsa_analysis_dict_all_rois[dataset], 
                                 extra_title_str = dataset + '_Trained: ',
                                 save_fig_path=save_fig_path)

#         plot_all_roi_rsa_scatter(rsa_analysis_dict_all_rois[dataset],
#                                  extra_title_str = dataset + '_Trained_10_SPLIT_MEDIAN: ',
#                                  use_all_165_sounds=False,
#                                  save_fig_path=save_fig_path)
    
        rsa_analysis_dict_all_rois_permuted[dataset] = {}
        for ROI in ['Primary', 'Lateral', 'Posterior', 'Anterior']:
            rsa_analysis_dict = rsa_cross_validated_all_models(randnetw='True', 
                                                               roi_name=ROI,
                                                               mean_subtract=True, 
                                                               with_std=True,
                                                               target=dataset,
                                                               ignore_layer_keys=['input_after_preproc'])
            rsa_analysis_dict_all_rois_permuted[dataset][ROI] = rsa_analysis_dict
    
        plot_all_roi_rsa_scatter(rsa_analysis_dict_all_rois_permuted[dataset], 
                                 extra_title_str = dataset + '_Permuted: ',
                                 save_fig_path=save_fig_path)
#         plot_all_roi_rsa_scatter(rsa_analysis_dict_all_rois_permuted[dataset],
#                                  extra_title_str = dataset + '_Permuted_10_SPLIT_MEDIAN: ',
#                                  use_all_165_sounds=False,
#                                  save_fig_path=save_fig_path)

    with open(os.path.join(save_fig_path, 'best_layer_rsa_analysis_dict.pckl'), 'wb') as f:
        pickle.dump({'rsa_analysis_dict_all_rois':rsa_analysis_dict_all_rois,
                     'rsa_analysis_dict_all_rois_permuted':rsa_analysis_dict_all_rois_permuted},
                    f)

def make_all_voxel_rsa_bar_plots_from_pckl(pckl_path, save_fig_path):
    with open(pckl_path, 'rb') as f:
        all_dataset_rsa_dict = pickle.load(f)

    ## Get the plots for the RSA across all models (Figure 2)
    for dataset in ['B2021','NH2015']:
        rsa_analysis_dict_trained = all_dataset_rsa_dict[dataset]['trained']
        model_ordering = plot_ordered_cross_val_RSA(rsa_analysis_dict_trained,
                                                    model_ordering=None,
                                                    use_165_sounds_for_fMRI_ceiling=False, # Plot the ceiling based on the splits
                                                    extra_title_str=dataset + '_Trained: ',
                                                    save_fig_path=save_fig_path)

        rsa_analysis_dict_permuted = all_dataset_rsa_dict[dataset]['permuted']
        _ = plot_ordered_cross_val_RSA(rsa_analysis_dict_permuted,
                                       model_ordering=model_ordering,
                                       use_165_sounds_for_fMRI_ceiling=False, # Plot the ceiling based on the splits
                                       extra_title_str=dataset + '_Permuted: ',
                                       save_fig_path=save_fig_path)


def make_all_voxel_rsa_bar_plots(save_fig_path):
    ## Get the plots for the RSA across all models (Figure 2)
    all_dataset_rsa_dict = {}
    for dataset in ['B2021','NH2015']:
        rsa_analysis_dict_trained = rsa_cross_validated_all_models(randnetw='False', 
                                                                   roi_name=None,
                                                                   mean_subtract=True, 
                                                                   with_std=True,
                                                                   target=dataset,
                                                                   save_name_base=save_fig_path)

        all_dataset_rsa_dict[dataset] = {'trained':rsa_analysis_dict_trained}
        model_ordering = plot_ordered_cross_val_RSA(rsa_analysis_dict_trained, 
                                                    model_ordering=None, 
                                                    use_165_sounds_for_fMRI_ceiling=False, # Plot the ceiling based on the splits
                                                    extra_title_str=dataset + '_Trained: ',
                                                    save_fig_path=save_fig_path)
   
        # Cochleagram is currently included in the permuted model analysis for Figure 2.  
        rsa_analysis_dict_permuted = rsa_cross_validated_all_models(randnetw='True', 
                                                                    roi_name=None,
                                                                    mean_subtract=True, 
                                                                    with_std=True,
                                                                    target=dataset,
                                                                    save_name_base=save_fig_path)
        
        _ = plot_ordered_cross_val_RSA(rsa_analysis_dict_permuted, 
                                       model_ordering=model_ordering, 
                                       use_165_sounds_for_fMRI_ceiling=False, # Plot the ceiling based on the splits
                                       extra_title_str=dataset + '_Permuted: ',
                                       save_fig_path=save_fig_path)
        
        all_dataset_rsa_dict[dataset]['permuted'] = rsa_analysis_dict_permuted

    with open(os.path.join(save_fig_path, 'all_dataset_rsa_dict.pckl'), 'wb') as f:
        pickle.dump(all_dataset_rsa_dict, f)

if __name__ == "__main__":
    make_paper_plots(save_fig_path=os.path.join(RESULTDIR, 'rsa_analysis'))
