from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from os.path import join
import operator
import copy
from tqdm import tqdm
from scipy.io import savemat
import random
import datetime
import scipy.io as sio
from scipy.io import loadmat
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon
import getpass
import xarray as xr
import pickle
import scipy.stats as stats
import statsmodels.stats.descriptivestats as ds
import sys
user = getpass.getuser()
# Go one directory up to import resources
sys.path.append(str(Path(os.getcwd()).parent))
from resources import *
now = datetime.datetime.now()
datetag = now.strftime("%Y%m%d-%H%M")

np.random.seed(0)
random.seed(0)

#### Paths ####
DATADIR = (Path(os.getcwd()) / '..' / '..' / 'data').resolve()
ROOT = (Path(os.getcwd()) / '..' / '..').resolve()
print(f'ROOT: {ROOT}')

# run_locally = True
# if user == 'gt':
#     if run_locally:
#         print(f' ------------- Running locally as {user} -----------')
#         ROOT = f'/Users/{user}/Documents/GitHub/auditory_brain_dnn/'
#     else:
#         print(f' ------------- Running as {user} using SFTP -----------')
#         ROOT = f'/Users/{user}/bur/'
# else:
#     print(f' ------------- Running on openmind as {user} -----------')
#     ROOT = f'/mindhive/mcdermott/u/{user}/auditory_brain_dnn/'

SAVEDIR_CENTRALIZED = f'{ROOT}/results/PLOTS_across-models/'
STATSDIR_CENTRALIZED = f'{ROOT}/results/STATS_across-models/'
RESULTDIR_ROOT = (Path(f'{ROOT}/results/')).resolve()


#### LOADING FUNCTIONS ####

def get_identifier(id_str):
    identifier_parts = id_str.split('_')
    mapping = identifier_parts[0]  # method
    target = identifier_parts[1].split('TARGET-')[1]
    source_model = identifier_parts[2].split('SOURCE-')[1]  # with layer
    source_model = source_model.split('-')[0]
    randnetw = id_str.split('RANDNETW-')[1].split('_')[0]
    randemb = id_str.split('RANDEMB-')[1].split('_')[0]
    return mapping, target, source_model, randnetw, randemb

def concat_dfs_modelwise(RESULTDIR,
                         mapping,
                         df_str,
                         source_model,
                         target,
                         truncate=False,
                         randnetw='False'):
    """Loads and concatenates dfs
    If target is B2021, compile outputs first.
    :param: truncate, int or False. If int will only load n number of files
    """
    
    layer_reindex = d_layer_reindex[source_model]
    df_str = df_str + '.pkl'
    # List all files in analysis type of interest
    files = []
    for file in os.listdir(RESULTDIR):
        # print(file)
        if file.startswith(mapping):
            files.append(os.path.join(file))
    
    # Get correct source model and target
    files_model_of_interest = []
    for f in files:
        s = f.split('SOURCE-')[-1]
        t = f.split('TARGET-')[1].split('_SOURCE')[0]
        if s.startswith(source_model):
            if t == target:
                files_model_of_interest.append(f)
    
    if truncate:  # for testing
        files_model_of_interest = files_model_of_interest[:truncate]

    # Get either randnetw = True or False (permuted network)
    files_after_rand = []
    for f in files_model_of_interest:
        if f.split('RANDNETW-')[1].split('_')[0] == randnetw:
            files_after_rand.append(f)
    
    # Only use files in layer reindex (not all layers)
    files_after_reindex = []
    intended_positions = {}
    for f in files_after_rand:  # find layer name, take care of dashes in the name..
        layer = f.split('SOURCE-')[1].split('_RAND')[0].split('-')[1:]
        if len(layer) == 1:
            layer = layer[0]
        else:
            layer = '-'.join(layer)
        if layer in layer_reindex: # This ensures that we get all the layers we want, but not necessarily in the right order
            files_after_reindex.append(f)
            # Get the intended position of the layer, i.e. the position it should have in the final dataframe which matches layer_reindex
            # Get the index of the layer in layer_reindex, 0 for first layer, 1 for second layer etc.
            numerical_index = layer_reindex.index(layer)
            intended_positions[numerical_index] = f

    print(f'Loading {df_str} results from {len(files_after_reindex)} folders')
    # Assert that intended positions and layer_reindex are the same
    assert len(intended_positions) == len(layer_reindex)
    assert set(intended_positions.values()) == set(files_after_reindex)

    # Now sort files_after_reindex according to intended_positions (does not matter for downstream code in which way the
    # original df_output is sorted, this is just a bit nicer.
    files_after_reindex_sorted = []
    for i in range(len(layer_reindex)):
        files_after_reindex_sorted.append(intended_positions[i])
    
    # get pkl files
    dfs_lst = []
    for i, f in tqdm(enumerate(files_after_reindex_sorted)):
            df = pd.read_pickle(RESULTDIR / f / df_str)
            
            dfs_lst.append(df)
    
    dfs_merge = pd.concat(dfs_lst)
    print(f'Len of final df: {len(dfs_merge)}')
    
    return dfs_merge, files_after_reindex_sorted


#### AGGREGATING AND PLOTTING SCORES FOR VOXELS FUNCTIONS ####
def get_vox_by_layer_pivot(output,
                           source_model,
                           val_of_interest='median_r2_test_c'):
    """Obtain a pivot of correctly reindex voxel (based on unique voxel_id) by source layer datafrane
    This does not aggregate (mean/median) across any voxels, it simply goes from a long format with source layers in one column,
     to having source layers as separate columns.
     
     Uses voxel_id as index.
    """
    assert (np.sum(np.isnan(output[val_of_interest].values)) == 0)

    # voxel_id is both index and a column, so we need to reset index to get a column. Copy the output df to avoid changing the original
    assert (output.index.values == output['voxel_id'].values).all()
    output2 = output.copy(deep=True).reset_index(drop=True)
    
    p = output2.pivot_table(index='voxel_id', columns='source_layer', values=val_of_interest)
    
    # make sure that columns are available for reindexing:
    l_layer_reindex = [i for i in d_layer_reindex[source_model] if
                       i in p.columns.values]  # get the available layers based on columns
    try:  # reindex columns
        p2 = p[d_layer_reindex[source_model]]  # if all cols match precisely
    except:
        print(
            f'\nNot all layers available for making a voxel by layer pivot table with value of interest: {val_of_interest}, using available ones:\n {l_layer_reindex}')
        p2 = p[l_layer_reindex]
    
    return p2


def voxwise_best_layer(output, source_model, val_of_interest='median_r2_train'):
    """Selects the best layer per voxel based on value of interest. Not in use."""
    
    p2 = get_vox_by_layer_pivot(output, source_model, val_of_interest=val_of_interest)
    
    assert (np.sum(np.isnan(p2)).values == 0).all()
    
    p_idxmax = pd.DataFrame(p2.idxmax(axis=1), columns=[f'best_layer_{val_of_interest}'])
    
    d_rel_layer_reindex = {}  # get idx of each layer position
    for idx, l in enumerate(d_layer_reindex[source_model]):
        d_rel_layer_reindex[l] = idx
    
    # convert from string layer number to int
    rel_pos = []
    for i in p_idxmax[f'best_layer_{val_of_interest}'].values:
        rel_pos.append(d_rel_layer_reindex[i])
    
    # the relative position, rel_pos in the model:
    p_idxmax['rel_pos'] = rel_pos
    
    return p_idxmax


def find_roi_vox_overlap(mask_roi_1, mask_roi_2, roi_1_name, roi_2_name):
    """

    :param mask_roi_1: array 0 denoting no ROI
    :param mask_roi_2: array 0 denoting no ROI
    :return:
    """
    i = np.intersect1d(np.argwhere(mask_roi_1 != 0).ravel(), np.argwhere(mask_roi_2 != 0).ravel())
    print(f'Total of {len(i)} overlaps between {roi_1_name} and {roi_2_name}.')
    return i


def inv_ROI_overlap(output):
    """Investigate overlap between the four ROIs of interest"""
    source_layers = output.source_layer.values
    
    # just use a single source layer
    output = output.loc[output.source_layer == source_layers[0]]
    
    print(f'Number of voxels in tonotopic ROI: {np.sum(output.tonotopic.values)}')
    print(f'Number of voxels in pitch ROI: {np.sum(output.pitch.values)}')
    print(f'Number of voxels in music ROI: {np.sum(output.music.values)}')
    print(f'Number of voxels in speech ROI: {np.sum(output.speech.values)}')
    
    ## overlaps
    find_roi_vox_overlap(output.tonotopic.values, output.pitch.values, 'tonotopic', 'pitch')
    find_roi_vox_overlap(output.tonotopic.values, output.music.values, 'tonotopic', 'music')
    find_roi_vox_overlap(output.tonotopic.values, output.speech.values, 'tonotopic', 'speech')
    find_roi_vox_overlap(output.pitch.values, output.music.values, 'pitch', 'music')
    find_roi_vox_overlap(output.pitch.values, output.speech.values, 'pitch', 'speech')
    find_roi_vox_overlap(output.music.values, output.speech.values, 'music', 'speech')
    
    # no overlaps of a voxel across all rois
    np.sum(np.logical_and.reduce(
        (output.tonotopic.values, output.pitch.values, output.music.values, output.speech.values)))
    
    # create col if voxel is in ANY roi
    # first, test by creating overlap of tonotopic and pitch voxels
    # should be 379+379-86=672 voxels
    # i.e. max ensures that if there is a 1, then that one is chosen
    np.sum(np.max(np.vstack([output.tonotopic.values, output.pitch.values]), axis=0))  # checks out
    
    any_roi_array = np.max(np.vstack([output.tonotopic.values, output.pitch.values,
                                      output.music.values, output.speech.values]), axis=0)
    
    print(f'{np.sum(any_roi_array)} voxels are in any ROI (one or more)')

def select_r2_test_CV_splits_nit(output_folders_paths,
                                 df_meta_roi,
                                 source_model,
                                 target,
                                 roi=None,
                                 value_of_interest='r2_test_c',
                                 randnetw='False',
                                 save=False,
                                 collapse_over_splits='median',
                                 nit=10,
                                 verbose=False,
                                 store_for_stats=True):
    """Select the best layer based on 5 CV splits, and take the r2 test value at that layer.
    As in all other analyses, make sure that we clip r2_test_c values at 1.
    Run the same procedure across n_it iterations (to not make the layer selection procedure dependent on a single split).
    Obtain the median across voxels per subject.

    We set the random seed to be the iteration split index. That means that if we run the different split iterations,
    we get different random splits (which of course is the intention).
    Within the split iteration loop, we load all the given layers of a model. Because we load the layers in the same order,
    we ensure that the random indices match across models with same architecture (or permuted networks).
    """
    print(f'\nFUNC: select_r2_test_CV_splits_nit\nMODEL: {source_model}, value_of_interest: {value_of_interest}, collapse_over_splits: {collapse_over_splits}, randnetw: {randnetw}\n')
    ## FOR LOGGING ##
    lst_df_best_layer_r_values = []

    if target == 'NH2015comp':
        index_val = df_meta_roi.comp_idx
    else:
        index_val = df_meta_roi.voxel_id
    
    # Loop over nit samples
    for i in tqdm(range(nit)):
        np.random.seed(i) # Use different random seed for each iteration, but still make it reproducible.

        lst_layer_selection = [] # For each iteration, store the values that are used for layer selection
        lst_r_value = [] # For each iteration, store the values that are for readout

        # Load the across-CV splits data for each layer
        for f in output_folders_paths:
            layer = f.split('SOURCE-')[1].split('_RAND')[0].split('-')[1:]
            if len(layer) == 1:
                layer = layer[0]
            else:
                layer = '-'.join(layer)
            ds = pd.read_pickle(join(f, 'ds.pkl'))
            ds_val = pd.DataFrame(ds[value_of_interest].values,
                                      index=index_val)  # index according to the comp/voxel id
            
            # Generate random indices for each voxel and layer
            m = np.zeros((ds_val.shape[0], ds_val.shape[1]))
            
            for n_vox in range(m.shape[0]):
                m[n_vox] = np.tile([0, 1], reps=5)  # 5 zeros, 5 ones
                np.random.shuffle(m[n_vox])
            
            # Use the 0's to obtain a median value for each voxel to select the best layer
            data_layer_selection = ds_val.copy(deep=True)
            data_r_value = ds_val.copy(deep=True)
            
            data_layer_selection[m == 0] = np.nan  # set indices vals of 0 to nan, only look at indices of 1
            data_r_value[m == 1] = np.nan  # set indices vals of 1 to nan, only look at indices of 0
            # assert that for each row (i.e. voxel), there are 5 nans and 5 non-nans
            assert np.all(np.sum(np.isnan(data_layer_selection), axis=1) == 5)
            assert np.all(np.sum(~np.isnan(data_layer_selection), axis=1) == 5)
            
            # Get the median/mean value for each voxel/component (across the five values that are held out)
            if collapse_over_splits == 'median': # Default, and used in paper
                data_layer_selection_median = np.nanmedian(data_layer_selection, axis=1)
                data_r_value_median = np.nanmedian(data_r_value, axis=1)
            elif collapse_over_splits == 'mean':
                data_layer_selection_median = np.nanmean(data_layer_selection, axis=1)
                data_r_value_median = np.nanmean(data_r_value, axis=1)
            else:
                raise ValueError('collapse_over_splits must be either median or mean')
            
            if value_of_interest == 'r2_test_c': # Clip value by 1 AFTER the median operation
                data_layer_selection_median = data_layer_selection_median.clip(max=1)
                data_r_value_median = data_r_value_median.clip(max=1)
            
            if verbose:
                print(
                    f'It: {i}: Correlation between two splits of the data using collapse over splits {collapse_over_splits} are: {np.corrcoef(data_layer_selection_median, data_r_value_median)[0][1]:.4}')
            
            # Create a dataframe with the median values for each voxel and column name as layer
            lst_layer_selection.append(pd.DataFrame(data_layer_selection_median, index=index_val, columns=[layer]))
            lst_r_value.append(pd.DataFrame(data_r_value_median, index=index_val, columns=[layer]))
        
        # prefix the value name with either the mean or the median, according to which operation happens over splits
        value_of_interest_post_collapse = collapse_over_splits + '_' + value_of_interest
        
        # Merge across layers and reindex
        df_layer_selection = pd.concat(lst_layer_selection, axis=1)[d_layer_reindex[source_model]]
        df_r_value = pd.concat(lst_r_value, axis=1)[d_layer_reindex[source_model]]
        
        # Get best layer for each voxel based on the df_layer_selection part of the data (same function as all the other layer argmax analyses)
        df_best_layer, _ = layer_position_argmax(p=df_layer_selection, source_model=source_model)
        
        # Now use the best layer per voxel to obtain the r2 test value for each voxel (independently selected)
        df_best_layer_r_values = pd.DataFrame(df_r_value.lookup(df_best_layer.index, df_best_layer.layer_pos.values),
                                              index=df_best_layer.index, columns=[value_of_interest_post_collapse])
        df_best_layer_r_values['pos'] = df_best_layer.pos
        df_best_layer_r_values['rel_pos'] = df_best_layer.rel_pos
        df_best_layer_r_values['layer_pos'] = df_best_layer.layer_pos
        df_best_layer_r_values['layer_legend'] = df_best_layer.layer_legend
        df_best_layer_r_values['nit'] = i
        
        assert (df_best_layer.index == df_best_layer_r_values.index).all()
        assert (df_best_layer_r_values.index == index_val).all()
        
        # Now obtain the subj_idx from df_roi_meta (if voxel values!) and merge with the r2 test values
        if target != 'NH2015comp':
            df_best_layer_r_values['subj_idx'] = df_meta_roi.subj_idx.values
            df_best_layer_r_values_grouped = df_best_layer_r_values.copy(deep=True).groupby('subj_idx').median()  # take median across voxels
        else:
            df_best_layer_r_values['comp'] = df_meta_roi.comp.values
            df_best_layer_r_values_grouped = df_best_layer_r_values.copy(deep=True)
        
        lst_df_best_layer_r_values.append(df_best_layer_r_values_grouped)
    
    # If store for stats, do not take mean over iterations
    if store_for_stats:
        df_best_layer_r_values_grouped_store = pd.concat(lst_df_best_layer_r_values, axis=0)
        df_best_layer_r_values_grouped_store['roi'] = roi
        df_best_layer_r_values_grouped_store['source_model'] = source_model
        df_best_layer_r_values_grouped_store['target'] = target
        df_best_layer_r_values_grouped_store['randnetw'] = randnetw
    
    # Now take the mean and std and sem across the iterations
    if target != 'NH2015comp':
        df_best_layer_r_values_grouped_mean = pd.concat(lst_df_best_layer_r_values).groupby('subj_idx').mean()
        df_best_layer_r_values_grouped_std = pd.concat(lst_df_best_layer_r_values).groupby('subj_idx').std(ddof=1)
        df_best_layer_r_values_grouped_sem = pd.concat(lst_df_best_layer_r_values).groupby('subj_idx').sem(ddof=1)
    else:
        df_best_layer_r_values_grouped_mean = pd.concat(lst_df_best_layer_r_values).groupby('comp_idx').mean()
        df_best_layer_r_values_grouped_std = pd.concat(lst_df_best_layer_r_values).groupby('comp_idx').std(ddof=1)
        df_best_layer_r_values_grouped_sem = pd.concat(lst_df_best_layer_r_values).groupby('comp_idx').sem(ddof=1)

    # rename std/sem over iterations
    df_best_layer_r_values_grouped_std.rename(columns={'pos': 'pos_std_over_it', 'rel_pos': 'rel_pos_std_over_it',
                                                       value_of_interest_post_collapse: f'{value_of_interest_post_collapse}_std_over_it'}, inplace=True)
    df_best_layer_r_values_grouped_sem.rename(columns={'pos': 'pos_sem_over_it', 'rel_pos': 'rel_pos_sem_over_it',
                                                       value_of_interest_post_collapse: f'{value_of_interest_post_collapse}_sem_over_it'}, inplace=True)
    
    df_best_layer_r_values_grouped_across_it = pd.concat([df_best_layer_r_values_grouped_mean,
                                                          df_best_layer_r_values_grouped_std,
                                                          df_best_layer_r_values_grouped_sem], axis=1)
    
    df_best_layer_r_values_grouped_across_it['roi'] = roi
    df_best_layer_r_values_grouped_across_it['source_model'] = source_model
    df_best_layer_r_values_grouped_across_it['target'] = target
    df_best_layer_r_values_grouped_across_it['randnetw'] = randnetw
    df_best_layer_r_values_grouped_across_it['datetag'] = datetag
    df_best_layer_r_values_grouped_across_it['nit'] = nit
    df_best_layer_r_values_grouped_across_it['method'] = 'regr'
    
    if save:
        if target != 'NH2015comp':
            save_str = f'best-layer_CV-splits-nit-{nit}_roi-{roi}_{source_model}{d_randnetw[randnetw]}_{target}_{value_of_interest_post_collapse}.csv'
        else:
            df_best_layer_r_values_grouped_across_it['comp_idx'] = df_best_layer_r_values_grouped_across_it.index
            df_best_layer_r_values_grouped_across_it.index = df_meta_roi.comp
            save_str = f'best-layer-CV-splits-nit-{nit}_per-comp_{source_model}{d_randnetw[randnetw]}_{target}_{value_of_interest_post_collapse}.csv'

        df_best_layer_r_values_grouped_across_it.to_csv(join(save, save_str))
        print(f'Saved {save}/{save_str}')
        if store_for_stats:
            df_best_layer_r_values_grouped_store.to_csv(join(save, save_str.replace('.csv', '_stats.csv')))
            print(f'Saved {save}/{save_str.replace(".csv", "_stats.csv")}')


def get_subject_pivot(output,
                      source_model,
                      roi=None,
                      value_of_interest='median_r2_test_c',
                      yerr_type='sem'):
    """
    Given a value of interest (e.g. median_r2_test_c, the median r2 value across CV splits), obtain a pivot table:
    subject by source layer. This pivot table is obtained by taking the median across voxels (i.e. using the median to aggregate within each subject's voxels).
    Obtain a subject-wise mean (one value per subject, independent of how many source layers there are: piv.mean(axis=1)
    The subject-wise mean is then subtracted from each subject-layer pair. The mean is used to aggregate the median-derived values from the subjects,
    i.e. when we have the subject-wise values we use the mean to get a single value for each layer, across subjects.
    
    :param output: output df
    :param source_model: str
    :param roi: None or str
    :param value_of_interest: str
    :param yerr_type: 'sem' or 'std' (within-subject error bar)
    :return: piv, df, the pivot table with subjects as rows, and layers as columns
            yerr, array of sem/std values, within-subject error bar
    """
    
    layer_reindex = d_layer_reindex[source_model]
    
    if roi:
        # extract roi voxels
        output = output.loc[output[roi] == 1]
    
    # check nans
    assert (output[value_of_interest].isna().sum() == 0)
    
    # manually compute pivots..
    # output.query('`source_layer` == "MaxPool2d_2" & `subj_idx` == 1').median_r2_test_c.median()
    # output.query('`source_layer` == "MaxPool2d_3" & `subj_idx` == 2')
    # output.query('`source_layer` == "MaxPool2d_3" & `subj_idx` == 1')
    # np.sum(np.isnan(output.query('`source_layer` == "MaxPool2d_3" & `subj_idx` == 2').median_r2_test_c.values))
    # output.loc[output.source_layer == 'MaxPool2d_3'].subj_idx.unique()
    
    # for each cond (layer), subtract the mean across cond. then, each subject will have a demeaned array
    piv = output.pivot_table(index='subj_idx', columns='source_layer', values=value_of_interest, aggfunc='median')
    try:
        piv = piv[layer_reindex]
    except:  # for random networks, pure nan columns exist, add 0 for plotting
        raise ValueError('There should be no NaNs in the pivot array!')
        zero_layers = np.setdiff1d(layer_reindex, piv.columns.values)
        print(f'The following layers have pure NaN values: {zero_layers}, setting to nan for plotting')
        c = {}
        for i in zero_layers:
            c[i] = np.nan
        
        piv = piv.assign(**c)
        piv = piv[layer_reindex]
    
    demeaned = piv.subtract(piv.mean(axis=1).values,
                            axis=0)  # get a mean value for each subject across all conds, and subtract it from individual layers
    if yerr_type == 'sem':
        yerr = np.std(demeaned.values.T + piv.mean().mean(), axis=1) / np.sqrt(
            piv.shape[0]-1)  # the first mean operation (piv.mean()) yields a value per layer across subjects
    # last part is the grand avg (where we are taking mean across voxels)
    # piv.shape[0] is the number of subjects (rows)
    if yerr_type == 'std':
        yerr = np.std(demeaned.values.T + piv.mean().mean(), axis=1)
    
    # Assert that plotted results have no nans
    assert(piv.isna().sum().sum() == 0)
    return piv, yerr

def obtain_spectemp_val(roi, target, value_of_interest='median_r2_test_c', yerr='sem'):
    """
    Obtain value from the spectemp model by taking median of voxels for each subject.
    No best layer needs to be selected.
    
    :param roi:
    :param target:
    :param value_of_interest:
    :param yerr:
    :return: subject x layer (even though there is just one) pivot
    """
    
    # Load spectemp output!
    output, _ = concat_dfs_modelwise((Path(f'/Users/gt/om2/results/AUD/20210915_median_across_splits_correction/spectemp')).resolve(),
                                              mapping='AUD-MAPPING-Ridge', df_str='df_output',
                                                  source_model='spectemp', target=target,
                                                  truncate=None, randemb='False', randnetw='False')
    
    # checks
    # np.sum(output[value_of_interest] > 1)
    
    if value_of_interest.endswith('_c'): # Clip value of interest by 1
        output[value_of_interest] = output[value_of_interest].clip(upper=1)
        
    assert (output[value_of_interest].isna().sum() == 0)

    if roi: # extract roi voxels
        if roi == 'any':
            output = output.loc[output['any_roi'] == 1]
        else: # specific ROIs
            output = output.loc[output[roi] == 1]
    else: # no roi, e.g. for NH2015comp
        output = output
    
    piv = output.pivot_table(index='subj_idx', columns='source_layer', values=value_of_interest, aggfunc='median')
    piv.rename(columns={'avgpool':value_of_interest}, inplace=True)
    piv['subj_idx'] = piv.index
    piv['layer'] = 'avgpool'
    piv['source_model'] = 'spectemp'
    piv['roi'] = roi
    
    return piv.reset_index(drop=True)


def obtain_spectemp_val_CV_splits_nit(roi,
                                      target,
                                      df_meta_roi,
                                      value_of_interest='median_r2_test_c',
                                      nit=10,
                                      collapse_over_splits='median',
                                      save=True,
                                      mapping='Ridge',
                                      alphalimit='50',
                                      store_for_stats=True,
                                      randnetw='False'):
    """
    Obtain value from the spectemp model by taking median of voxels for each subject.
    No best layer needs to be selected.
    Mirrors approach in select_r2_test_CV_splits_nit(): obtain value of interst from the ds array
    which has values for each CV split. Take the median across 5 randomly selected splits.
    Repeat nit times and take the mean across iterations.
    
    Collapse over splits refers to the operation that is taken over the randomly selected CV splits (NOT iteration splits)
    
    Edit 20220210: Add save and store for stats option. Stored these on 20220210.
    Can take randnetw arg, but will always return randnetw False. For compatability with the rest of the code.
    
    :param roi:
    :param target:
    :param value_of_interest:
    :param yerr:
    :return: subject x layer (even though there is just one) pivot
    """
    
    # Load spectemp output!
    ds = pd.read_pickle(f'{RESULTDIR_ROOT}/spectemp/AUD-MAPPING-{mapping}_TARGET-{target}_SOURCE-spectemp-' \
                                    f'avgpool_RANDNETW-False_ALPHALIMIT-{alphalimit}/ds.pkl')
    
    # Get value of interest without the 'median'/'mean' appended to it
    value_of_interest_suffix = '_'.join(value_of_interest.split('_')[1:])

    assert (np.sum(np.isnan(ds[value_of_interest_suffix]) == 0))
    
    if target != 'NH2015comp':
        index_val = df_meta_roi.voxel_id
    else:
        index_val = df_meta_roi.comp_idx
    
    ds_val = pd.DataFrame(ds[value_of_interest_suffix].values,index=index_val)  # index according to the voxel id or comp idx
    
    lst_df_best_layer_r_values = []

    # Loop over nit samples
    for i in range(nit):
        np.random.seed(i) # To match the select_r2_test_CV_splits_nit() function and for reproducibility
        # Generate random indices for each voxel
        m = np.zeros((ds_val.shape[0], ds_val.shape[1]))

        for n_vox in range(m.shape[0]):
            m[n_vox] = np.tile([0, 1], reps=5)  # 5 zeros, 5 ones
            np.random.shuffle(m[n_vox])

        # Use the 0's to obtain a median value for each voxel to take the median of
        data_r_value = ds_val.copy(deep=True) # Simply 5 randomly selected values, no selection
        data_r_value[m == 1] = np.nan

        # Get the median/mean value for each voxel across splits (collapse over splits)
        if collapse_over_splits == 'median':
            data_r_value_median = np.nanmedian(data_r_value, axis=1)
        elif collapse_over_splits == 'mean':
            data_r_value_median = np.nanmean(data_r_value, axis=1)
        else:
            raise ValueError('collapse_over_splits must be either median or mean')

        if value_of_interest_suffix.endswith('_c'):  # Clip value by 1 AFTER the median operation
            data_r_value_median = data_r_value_median.clip(max=1)
            
        # Take median across voxels for each subject, but first append the df_meta_roi information
        df_r_value_median = pd.DataFrame(data_r_value_median, index=index_val, columns=[value_of_interest])
        df_r_value_median['nit'] = i
        if target == 'NH2015comp':
            df_r_value_median['comp'] = df_meta_roi.comp
        
        if target != 'NH2015comp':
            df_r_value_median['subj_idx'] = df_meta_roi.subj_idx.values
            # Now take the median across voxels for each subject
            df_r_value_median_grouped = df_r_value_median.copy(deep=True).groupby('subj_idx').median()
        else:
            df_r_value_median_grouped = df_r_value_median.copy(deep=True)
            
        
        # Store so I can take the mean over iterations later
        lst_df_best_layer_r_values.append(df_r_value_median_grouped)
    
    # If store for stats, do not take mean over iterations
    if store_for_stats:
        df_best_layer_r_values_grouped_store = pd.concat(lst_df_best_layer_r_values, axis=0)
        df_best_layer_r_values_grouped_store['roi'] = roi
        df_best_layer_r_values_grouped_store['source_model'] = 'spectemp'
        df_best_layer_r_values_grouped_store['target'] = target
        df_best_layer_r_values_grouped_store['randnetw'] = 'False'
    
    # Now take the mean and std and sem across the iterations
    if target != 'NH2015comp':
        df_best_layer_r_values_grouped_mean = pd.concat(lst_df_best_layer_r_values).groupby('subj_idx').mean()
        df_best_layer_r_values_grouped_std = pd.concat(lst_df_best_layer_r_values).groupby('subj_idx').std()
        df_best_layer_r_values_grouped_sem = pd.concat(lst_df_best_layer_r_values).groupby('subj_idx').sem()
    else:
        df_best_layer_r_values_grouped_mean = pd.concat(lst_df_best_layer_r_values).groupby('comp_idx').mean()
        df_best_layer_r_values_grouped_std = pd.concat(lst_df_best_layer_r_values).groupby('comp_idx').std()
        df_best_layer_r_values_grouped_sem = pd.concat(lst_df_best_layer_r_values).groupby('comp_idx').sem()

    # rename std and sem over iterations
    df_best_layer_r_values_grouped_std.rename(columns={value_of_interest: f'{value_of_interest}_std_over_it'},
                                              inplace=True)
    df_best_layer_r_values_grouped_sem.rename(columns={value_of_interest: f'{value_of_interest}_sem_over_it'},
                                              inplace=True)

    df_best_layer_r_values_grouped_across_it = pd.concat(
        [df_best_layer_r_values_grouped_mean, df_best_layer_r_values_grouped_std, df_best_layer_r_values_grouped_sem], axis=1)

    df_best_layer_r_values_grouped_across_it['roi'] = roi
    df_best_layer_r_values_grouped_across_it['source_model'] = 'spectemp'
    df_best_layer_r_values_grouped_across_it['target'] = target
    df_best_layer_r_values_grouped_across_it['randnetw'] = 'False'
    df_best_layer_r_values_grouped_across_it['subj_idx'] = df_best_layer_r_values_grouped_across_it.index
    df_best_layer_r_values_grouped_across_it['pos'] = 1 # dummy
    df_best_layer_r_values_grouped_across_it['rel_pos'] = 1 # dummy
    df_best_layer_r_values_grouped_across_it['roi'] = roi
    df_best_layer_r_values_grouped_across_it['datetag'] = datetag
    df_best_layer_r_values_grouped_across_it['nit'] = nit
    df_best_layer_r_values_grouped_across_it['method'] = 'regr'
    
    if target == 'NH2015comp':
        df_best_layer_r_values_grouped_across_it['comp_idx'] = df_best_layer_r_values_grouped_across_it.index
        df_best_layer_r_values_grouped_across_it['comp'] = df_meta_roi.comp
        df_best_layer_r_values_grouped_across_it.index = df_meta_roi.comp

    # prefix the value name with either the mean or the median, according to which operation happens over splits
    value_of_interest_post_collapse = collapse_over_splits + '_' + value_of_interest_suffix
    
    if save:
        PLOTDIR = (
            Path(f'{ROOT}/results/spectemp/outputs')).resolve() # create path to the output spectemp dir
        if target != 'NH2015comp':
            save_str = f'best-layer_CV-splits-nit-{nit}_roi-{roi}_spectemp_{target}{d_randnetw[randnetw]}_{value_of_interest_post_collapse}.csv'
        else:
            df_best_layer_r_values_grouped_across_it['comp_idx'] = df_best_layer_r_values_grouped_across_it.index
            df_best_layer_r_values_grouped_across_it.index = df_meta_roi.comp
            save_str = f'best-layer-CV-splits-nit-{nit}_per-comp_spectemp_{target}{d_randnetw[randnetw]}_{value_of_interest_post_collapse}.csv'

        df_best_layer_r_values_grouped_across_it.to_csv(join(PLOTDIR, save_str))
        if store_for_stats:
            df_best_layer_r_values_grouped_store.to_csv(join(PLOTDIR, save_str.replace('.csv', '_stats.csv')))
    
    return df_best_layer_r_values_grouped_across_it


def obtain_NH2015comp_spectemp_val(target,
                                   value_of_interest='median_r2_test'):
    """
    Obtain value from the spectemp model as predicted by NH2015comp.
    If the value is median_r2_test, then yerr will be fetched as std_r2_test and divided by (10-1) to obtain SEM.

    :param roi:
    :param target:
    :param value_of_interest: which r2 value to take
    :return:
    """
    
    # Load spectemp output!
    output, _ = concat_dfs_modelwise(
        (Path(f'{RESULTDIR_ROOT}/spectemp')).resolve(),
        mapping='AUD-MAPPING-Ridge',
        df_str='df_output',
        source_model='spectemp',
        target=target,
        truncate=None,
        randnetw='False')
    
    value_of_interest_suffix = '_'.join(value_of_interest.split('_')[1:])
    
    # output = output.set_index('comp')
    output_vals_of_interest = output[[value_of_interest, f'std_{value_of_interest_suffix}', 'comp', 'source_model', 'source_layer']]
    output_vals_of_interest[f'std_{value_of_interest_suffix}'] = output[f'std_{value_of_interest_suffix}']/(10-1) # divide by n-1 to obtain SEM. 10 splits
    
    # mock meta cols
    output_vals_of_interest['layer_legend'] = 'AvgPool'
    output_vals_of_interest['layer_pos'] = 'avgpool'
    output_vals_of_interest['rel_pos'] = 1
    output_vals_of_interest['pos'] = 1
    
    output_vals_of_interest.rename(columns={f'std_{value_of_interest_suffix}':f'sem_{value_of_interest_suffix}',}, inplace=True)
    
    return output_vals_of_interest

def plot_score_across_layers(output,
                             output_randnetw=None,
                             source_model='',
                             target='',
                             roi=None,
                             ylim=[0, 1],
                             save=False,
                             alpha=1,
                             alpha_randnetw=0.3,
                             label_rotation=45,
                             value_of_interest='median_r2_test_c',):
    """
    Plot median variance explained across layers.
    Default error bar is within-subject SEM.
    
    :param output: pd.DataFrame with columns: ['source_model', 'source_layer', 'target', 'roi', '{value_of_interest}']
    :param output_randnetw: same type of df as output, but for permuted network.
    :param source_model: str
    :param roi: None or str. If str, only plot subsets of ROIs. If None, plot all voxels available.
        Str options are: "roi_label_general" (anatomical ROIs)
    :param ylim: list of 2 elements
    :param save: bool
    :param alpha: float, for plotting
    :param alpha_randnetw: float, for plotting
    :param label_rotation: int, for plotting
    :param value_of_interest: str, which value to plot. E.g., 'median_r2_test_c', 'median_r2_test'
    :return:
    """
    
    layer_reindex = d_layer_reindex[source_model]
    layer_legend = [d_layer_names[source_model][layer] for layer in layer_reindex]

    if roi == 'roi_label_general': # Primary, Anterior, Lateral, Posterior
        # extract roi voxels
        piv_primary, yerr_primary = get_subject_pivot(output=output, source_model=source_model,
                                                      roi='Primary', value_of_interest=value_of_interest, yerr_type='sem')
        piv_anterior, yerr_anterior = get_subject_pivot(output=output, source_model=source_model,
                                                        roi='Anterior', value_of_interest=value_of_interest, yerr_type='sem')
        piv_lateral, yerr_lateral = get_subject_pivot(output=output, source_model=source_model,
                                                        roi='Lateral', value_of_interest=value_of_interest, yerr_type='sem')
        piv_posterior, yerr_posterior = get_subject_pivot(output=output, source_model=source_model,
                                                        roi='Posterior', value_of_interest=value_of_interest, yerr_type='sem')

        title_str = f'{d_model_names[source_model]}, {target}, {roi}'
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_box_aspect(0.6)
        plt.errorbar(np.arange(len(layer_reindex)), piv_primary.mean().values, yerr=yerr_primary, alpha=alpha, lw=2,
                     label='Primary', color=d_roi_colors['Primary'],)
        plt.errorbar(np.arange(len(layer_reindex)), piv_anterior.mean().values, yerr=yerr_anterior, alpha=alpha, lw=2,
                     label='Anterior', color=d_roi_colors['Anterior'],)
        plt.errorbar(np.arange(len(layer_reindex)), piv_lateral.mean().values, yerr=yerr_lateral, alpha=alpha, lw=2,
                     label='Lateral', color=d_roi_colors['Lateral'],)
        plt.errorbar(np.arange(len(layer_reindex)), piv_posterior.mean().values, yerr=yerr_posterior, alpha=alpha, lw=2,
                        label='Posterior', color=d_roi_colors['Posterior'],)
        plt.xticks(np.arange(len(layer_legend)), layer_legend, rotation=label_rotation,
                   fontsize=14, ha='right', rotation_mode='anchor')
        plt.yticks(fontsize=16)
        plt.ylabel(d_value_of_interest[value_of_interest], fontsize=16)
        plt.title(title_str, fontsize=16)
        plt.tight_layout(h_pad=1.2)
        plt.ylim(ylim)
        plt.legend(frameon=False)
        if save:
            plt.savefig(
                join(SAVEDIR_CENTRALIZED, f'across-layers_roi-{roi}_{source_model}_{target}_{value_of_interest}.svg'),
                dpi=180)

            # save csv for each pivot
            # Iterate over the piv dfs and yerrs
            lst_pivs = [piv_primary, piv_anterior, piv_lateral, piv_posterior]
            lst_yerrs = [yerr_primary, yerr_anterior, yerr_lateral, yerr_posterior]
            lst_roi_labels = ['Primary', 'Anterior', 'Lateral', 'Posterior']
            for piv, yerr, roi_label in zip(lst_pivs, lst_yerrs, lst_roi_labels):
                piv_save = piv.copy(deep=True) # copy the pivot table # rename columns to include roi label
                df_yerr = pd.DataFrame([yerr], columns=piv_save.columns, index=['yerr'])  # append yerr to the pivot table that is plotted
                piv_save = piv_save.append(df_yerr)

                piv_save.to_csv(join(save, f'across-layers_roi-{roi_label}_{source_model}_{target}_{value_of_interest}.csv'))

        plt.show()
    
    else:  # no ROI, all voxels
        piv, yerr = get_subject_pivot(output=output, source_model=source_model, roi=roi,
                                      value_of_interest=value_of_interest, yerr_type='sem')
        if output_randnetw is not None:
            piv_randnetw, yerr_randnetw = get_subject_pivot(output=output_randnetw, source_model=source_model, roi=roi,
                                                            value_of_interest=value_of_interest, yerr_type='sem')
        # Plot specs
        title_str = f'{d_model_names[source_model]}, {target}, all voxels'

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_box_aspect(0.6)
        plt.errorbar(np.arange(len(layer_reindex)), piv.mean().values, yerr=yerr, alpha=alpha, lw=2,
                     color=d_roi_colors['none'])
        if output_randnetw is not None:
            plt.errorbar(np.arange(len(layer_reindex)), piv_randnetw.mean().values, yerr=yerr_randnetw,
                         alpha=alpha_randnetw, lw=2, color=d_roi_colors['none'],
                         label='Permuted network')
        plt.xticks(np.arange(len(layer_legend)), layer_legend, rotation=label_rotation,
                   fontsize=14, ha='right', rotation_mode='anchor')
        plt.yticks(fontsize=16)
        plt.ylabel(d_value_of_interest[value_of_interest], fontsize=16)
        plt.ylim(ylim)
        plt.title(title_str, fontsize=16)
        plt.tight_layout(h_pad=1.2)
        plt.legend(frameon=False)
        if save:
            plt.savefig(
                join(SAVEDIR_CENTRALIZED, f'across-layers_roi-{roi}_{source_model}_{target}_{value_of_interest}.svg'), dpi=180)

            # save csv
            piv_save = piv.copy(deep=True)
            df_yerr = pd.DataFrame([yerr], columns=piv_save.columns, index=['yerr'])  # append yerr to the pivot table that is plotted
            piv_save = piv_save.append(df_yerr)
            piv_save.to_csv(join(save, f'across-layers_roi-{roi}_{source_model}_{target}_{value_of_interest}.csv'))

            if output_randnetw is not None:
                # save csv randnetw
                piv_save = piv_randnetw.copy(deep=True)
                df_yerr = pd.DataFrame([yerr_randnetw], columns=piv_save.columns, index=['yerr'])
                piv_save = piv_save.append(df_yerr)
                piv_save.to_csv(
                    join(save, f'across-layers_roi-{roi}_{source_model}_randnetw_{target}_{value_of_interest}.csv'))

        plt.show()

def load_score_across_layers_across_models(source_models,
                                         target='NH2015',
                                         roi=None,
                                         value_of_interest='median_r2_test_c',
                                         randnetw='False',
                                         agg_method='mean',
                                         save=False,
                                         RESULTDIR_ROOT='/Users/gt/bur/results',
                                         add_savestr=''):
    """
    Loads the score across layers for a list of source models, for a given target, ROI, and randnetw specification
    (the data stored using plot_score_across_layers()).
    
    Loop across models and compile a dataframe for each model that aggregates (default: mean) across subjects and stores the
    dataframe in a dictionary with the key = source model.
    
    Args:
        source_models (list): list of source models to load
        target (str): target to load
        roi (str or None): ROI to load
        value_of_interest (str): value of interest to load
        randnetw (str): whether to load randnetw or not
        agg_method (str): method to aggregate the data (i.e. which aggregation to perform across subjects)
        save (bool): whether to save the data or not
        RESULTROOT (str): path to the results root
        add_savestr (str): string to add to the save path
        
    Returns:
        dict: dictionary with the dataframes for each source model (key = source model, value = df with rows as layers and
            columns with the meaned (agg_method) across subjects) and the yerror as obtained from the layer-wise within-subject err
        d_df_across_models: dict of dataframes for each source model. Key = source_model. Value = df with index as the subj idx
         and columns are median_r2_test_c (value_of_interest) for each layer

    """
    
    d_across_models = {}
    d_df_across_models = {}
    for source_model in source_models:
        RESULTDIR_MODEL = f'{RESULTDIR_ROOT}/{source_model}/outputs/'
        fname = f'across-layers_roi-{roi}_{source_model}{d_randnetw[randnetw]}_{target}_{value_of_interest}.csv'
        try:
            df_model = pd.read_csv(join(RESULTDIR_MODEL, fname)).copy(deep=True).set_index('Unnamed: 0')
        except FileNotFoundError:
            print(f'Source model {source_model} does not have a file {fname}')
            continue
        
        # Check whether layers are correctly ordered
        layer_reindex = d_layer_reindex[source_model]
        assert layer_reindex == list(df_model.columns)
        
        # Obtain mean of all rows besides the "yerr" row, but store it so we can re-append later
        yerr = df_model.loc['yerr', :]
        df_model_no_yerr = df_model.drop('yerr', axis=0)
        d_df_across_models[source_model] = df_model_no_yerr # store the subject-wise scores too
        
        if agg_method == 'mean': # means across subjects just like in the remaining analyses
            df_model_mean = df_model_no_yerr.mean(axis=0)
        elif agg_method == 'median':
            df_model_mean = df_model_no_yerr.median(axis=0)
        else:
            raise ValueError(f'{agg_method} is not a valid aggregation method')
        
        # Package the dataframe along with the previously computed within-subject y error
        df_model_mean_yerr = pd.DataFrame(df_model_mean.copy(deep=True)).rename(columns={0: f'{value_of_interest}_agg-{agg_method}{d_randnetw[randnetw]}'})
        df_model_mean_yerr[f'yerr_{value_of_interest}_agg-{agg_method}{d_randnetw[randnetw]}'] = yerr # the y error here is the previously computed within-subject y error
        
        # Log
        df_model_mean_yerr['source_model'] = source_model
        df_model_mean_yerr['target'] = target
        df_model_mean_yerr['roi'] = roi
        df_model_mean_yerr['value_of_interest'] = value_of_interest
        df_model_mean_yerr['randnetw'] = randnetw
        df_model_mean_yerr['agg_method'] = agg_method
        df_model_mean_yerr['datetag'] = datetag
        
        if save: # store each model's dataframe to a csv for logging purposes
            df_model_mean_yerr.to_csv(join(RESULTDIR_MODEL, f'across-layers-aggregated-{agg_method}_'
                                                            f'roi-{roi}_{source_model}{d_randnetw[randnetw]}_'
                                                            f'{target}_{value_of_interest}{add_savestr}.csv'))
        
        d_across_models[source_model] = df_model_mean_yerr
        
    return d_across_models, d_df_across_models
        

#### PLOTTING COMPONENT RESPONSES ####
def plot_comp_across_layers(output,
                            source_model,
                            output_randnetw=None,
                            target='',
                            value_of_interest='median_r2_test',
                            ylim=[0, 1],
                            save=False,
                            alpha=1,
                            alpha_randnetw=0.7,
                            label_rotation=45, ):
    """
    Plot component responses across layers and plot SEM across splits (std across r2 test splits / np.sqrt(num splits-1))
    
    :param output:
    :param source_model:
    :param output_randnetw:
    :param value_of_interest:
    :param ylim:
    :param save:
    :param alpha:
    :param alpha_randnetw:
    :param label_rotation:
    :return:
    """
    
    layer_reindex = d_layer_reindex[source_model]
    layer_legend = [d_layer_names[source_model][l] for l in layer_reindex]
    value_of_interest_suffix = '_'.join(value_of_interest.split('_')[1:]) # remove 'median_'
    
    # Get layer position info
    pos = np.arange(len(layer_legend)) + 1
    num_layers = len(layer_reindex)
    print(f'{source_model} has {num_layers} layers')
    min_possible_layer = 1  # for MATLAB, and to align with "pos"
    max_possible_layer = num_layers  # for MATLAB and to align with "pos"
    rel_pos = np.divide((np.subtract(pos, min_possible_layer)),(max_possible_layer - min_possible_layer))
    
    output = output.set_index('source_layer')
    
    title_str = f'NH2015 components: {source_model}'
    title_str_randnetw = f'NH2015 components: {source_model}, randnetw'

    d = {}  # store component name as key, and value as the across layer scores of interest
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_box_aspect(0.6)
    for i in np.unique(output.comp_idx):
        comp_name = np.unique(output.loc[output.comp_idx == i].comp)[0]
        comp_vals_across_layers = output.loc[output.comp_idx == i].reindex(layer_reindex)[value_of_interest].values
        comp_sem_across_layers = output.loc[output.comp_idx == i].reindex(layer_reindex)[
                                     f'std_{value_of_interest_suffix}'].values / np.sqrt(10 - 1)
        plt.errorbar(np.arange(len(layer_reindex)), comp_vals_across_layers,
                     yerr=comp_sem_across_layers,
                     color=d_roi_colors[comp_name], label=comp_name, alpha=alpha, lw=2)
        d[f'{comp_name}_{value_of_interest}'] = comp_vals_across_layers
        d[f'{comp_name}_sem_{value_of_interest_suffix}'] = comp_sem_across_layers
    plt.ylim(ylim)
    plt.xticks(np.arange(len(layer_legend)), layer_legend, rotation=label_rotation)
    plt.ylabel(d_value_of_interest[value_of_interest])
    plt.title(title_str)
    # plt.legend(frameon=False)
    plt.tight_layout(h_pad=1.2)
    if save:
        plt.savefig(
            join(SAVEDIR_CENTRALIZED, f'across-layers_{source_model}_{target}_{value_of_interest}_nolegend.png'), dpi=180)
        plt.savefig(
            join(SAVEDIR_CENTRALIZED, f'across-layers_{source_model}_{target}_{value_of_interest}_nolegend.svg'), dpi=180)
        # compile csv
        df_save = pd.DataFrame(d, index=layer_reindex)
        df_save['layer_pos'] = layer_reindex
        df_save['pos'] = pos
        df_save['rel_pos'] = rel_pos
        df_save['layer_legend'] = layer_legend

        df_save.to_csv(join(save, f'across-layers_{source_model}_{target}_{value_of_interest}.csv'))
    plt.show()
    
    if output_randnetw is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_box_aspect(0.6)
        output_randnetw = output_randnetw.set_index('source_layer')
        d_randnetw = {}
        for i in np.unique(output_randnetw.comp_idx):
            comp_name_randnetw = np.unique(output_randnetw.loc[output_randnetw.comp_idx == i].comp)[0]
            comp_vals_across_layers_randnetw = \
            output_randnetw.loc[output_randnetw.comp_idx == i].reindex(layer_reindex)[value_of_interest].values
            comp_sem_across_layers_randnetw = output_randnetw.loc[output_randnetw.comp_idx == i].reindex(layer_reindex)[
                                                  f'std_{value_of_interest_suffix}'].values / np.sqrt(10-1)
            plt.errorbar(np.arange(len(layer_reindex)),
                         comp_vals_across_layers_randnetw,
                         yerr=comp_sem_across_layers_randnetw,
                         color=d_roi_colors[comp_name_randnetw], label=comp_name_randnetw, alpha=alpha_randnetw, lw=2)
            d_randnetw[f'{comp_name_randnetw}_{value_of_interest}'] = comp_vals_across_layers_randnetw
            d_randnetw[f'{comp_name_randnetw}_sem_{value_of_interest_suffix}'] = comp_sem_across_layers_randnetw
        plt.ylim(ylim)
        plt.xticks(np.arange(len(layer_legend)), layer_legend, rotation=label_rotation)
        plt.ylabel(d_value_of_interest[value_of_interest])
        plt.title(title_str_randnetw)
        # plt.legend(frameon=False)
        plt.tight_layout(h_pad=1.2)
        if save:
            plt.savefig(join(SAVEDIR_CENTRALIZED, f'across-layers_{source_model}_randnetw_{target}_{value_of_interest}_nolegend.svg'), dpi=180)
            plt.savefig(join(SAVEDIR_CENTRALIZED, f'across-layers_{source_model}_randnetw_{target}_{value_of_interest}_nolegend.png'), dpi=180)
            # compile csv for randnetw
            df_save_randnetw = pd.DataFrame(d_randnetw, index=layer_reindex)
            df_save_randnetw['layer_pos'] = layer_reindex
            df_save_randnetw['pos'] = pos
            df_save_randnetw['rel_pos'] = rel_pos
            df_save_randnetw['layer_legend'] = layer_legend
            
            df_save_randnetw.to_csv(join(save, f'across-layers_{source_model}_randnetw_{target}_{value_of_interest}.csv'))
        plt.show()


def obtain_best_layer_per_comp(source_model, target, randnetw='False', value_of_interest='median_r2_test',
                               sem_of_interest='sem_r2_test', save=True):
    """Load the across-layers data for components (csv with rows as layers and columns as components).
    Requires plot_comp_across_layers to be run first.
    
    Simply takes the best layer index for each component (dependent procedure, argmax procedure)
    
    randnetw is 'True' or 'False'. If 'True', load the across-layers data for randnetw components.
    """
    df = pd.read_csv(
        join(RESULTDIR_ROOT, source_model, 'outputs', f'across-layers_{source_model}{d_randnetw[randnetw]}_{target}_{value_of_interest}.csv')).rename(
        columns={'Unnamed: 0': 'layer'})
    
    # Find best layer for each component
    lst_per_comp_best_layer = []
    for comp in ['lowfreq', 'highfreq', 'envsounds', 'pitch', 'speech', 'music']:
        
        col_of_interest_r2_test = f'{comp}_{value_of_interest}'
        col_of_interest_r2_sem = f'{comp}_{sem_of_interest}' # across CV splits
        
        # Find the best layer for the component and
        pos = df[col_of_interest_r2_test].idxmax() + 1 # +1 to align with MATLAB and neural data
        
        # Package into a one-row df (per component)
        d = {f'{value_of_interest}': df.iloc[pos-1][col_of_interest_r2_test],
             f'{sem_of_interest}': df.iloc[pos-1][col_of_interest_r2_sem],
             'pos': pos,
             'rel_pos': df.iloc[pos-1]['rel_pos'],
             'layer_pos': df.iloc[pos-1]['layer'], # uses Python indexing, so -1 to get the correct layer name
             'layer_legend': df.iloc[pos-1]['layer_legend'],
             'model': source_model}
        
        df_comp_best_layer = pd.DataFrame(d, index=[comp])
        df_comp_best_layer['randnetw'] = randnetw
        lst_per_comp_best_layer.append(df_comp_best_layer)
    
    df_all_comp = pd.concat(lst_per_comp_best_layer)  # checked Dec 2021, matches across layer comp plots
    
    if save:
        df_all_comp.to_csv(
            join(RESULTDIR_ROOT, source_model, 'outputs', f'best-layer-argmax_per-comp_{source_model}{d_randnetw[randnetw]}_{target}_{value_of_interest}.csv'))


def scatter_NH2015comp_resp_vs_pred(source_model,
                                    target,
                                    value_of_interest='median_r2_test',
                                    save=False,
                                    comp_to_plot=['lowfreq', 'highfreq', 'envsounds', 'pitch', 'speech', 'music'],
                                    randnetw='False',
                                    alphalimit='50',
                                    mapping='Ridge',
                                    generate_scatter_plot=True,
                                    obtain_partial_corr=True,
                                    nit=10,
                                    add_savestr='',
                                    alpha_dot=1):
    """Plot scatter of component responses versus actual responses, colored according to sound categories
    
    Selection of which layer to plot: select most frequently occurring layer for each component (across 10 iterations
    of the layer selection procedure): return most frequent layer for each component. If ties, return a random one.
    
    """
    # Set random seed to 0 for reproducibility
    np.random.seed(0)

    # Load the actual component data
    comp_data = load_comp_responses()
    
    # Load the component best layers for the source model of interest
    # Get the output folder name
    if source_model == 'spectemp':
        # Get output folder
        df_best_layer_comp = obtain_output_folders_for_any_comp_model(target=target,
                                                                      source_model='spectemp',
                                                                      value_of_interest=value_of_interest,)
        # Get the aggregate spectemp scores (which were also obtained over 10 iterations)
        df_agg_scores = pd.read_csv(join(RESULTDIR_ROOT,
                                         source_model,
                                         'outputs',
                                         f'best-layer-CV-splits-nit-{nit}_'
                                         f'per-comp_spectemp_{target}{d_randnetw[randnetw]}_{value_of_interest}.csv')).set_index('comp', drop=False)

        
        
    else: # We have to load the layer selection proecedure results, otherwise we do not know which layer to plot
        df_best_layer_across_it = pd.read_csv(join(RESULTDIR_ROOT,
                                                   source_model,
                                                   'outputs',
                                                   f'best-layer-CV-splits-nit-{nit}_per-comp_'
                                                   f'{source_model}{d_randnetw[randnetw]}_'
                                                   f'{target}_{value_of_interest}_stats.csv')).set_index('comp', drop=False)
        ### WHICH LAYER TO PLOT ###
        # select most frequently occurring layer for each component: return most frequent layer for each component. If ties, return a random one of the best ones
        lst_best_layer_per_comp = []
        for c in df_best_layer_across_it.comp.unique():
            df_comp = df_best_layer_across_it.query(f'comp == "{c}"')
            # Count number of times each layer occurs
            df_comp_count = df_comp.groupby('layer_pos').count()['comp_idx']
            # If the top one is a tie, select a random one
            if df_comp_count.max() > 1:
                selected_layer = np.random.choice(df_comp_count.index[df_comp_count == df_comp_count.max()].values)
            else: # just select the top one
                selected_layer = df_comp_count.index[0]
                
            df_best_layer_comp = pd.DataFrame({'layer_pos': [selected_layer],
                                              'layer_legend': [df_comp.query(f'layer_pos == "{selected_layer}"').layer_legend.values[0]],
                                              'comp': [c],
                                               'source_model':source_model}, index=[c])
                                              
            lst_best_layer_per_comp.append(df_best_layer_comp)
        
        df_best_layer_comp = pd.concat(lst_best_layer_per_comp)

        # Append output folders
        df_best_layer_comp['output_folder'] = [f'{RESULTDIR_ROOT}/{source_model}/' \
                                               f'AUD-MAPPING-{mapping}_TARGET-{target}_SOURCE-{source_model}-' \
                                               f'{x[1].layer_pos}_RANDNETW-{randnetw}_ALPHALIMIT-{alphalimit}' for x in df_best_layer_comp.iterrows()]
            
        ### WHICH R2 VALUE TO SHOW ###
        # Get a df_best_layer_comp df with output folders and value of interest based on CVsplits nit 10 (aggregated across 10 iterations)
        df_agg_scores = pd.read_csv(join(RESULTDIR_ROOT,
                                          source_model,
                                          'outputs',
                                           f'best-layer-CV-splits-nit-10_per-comp_'
                                           f'{source_model}{d_randnetw[randnetw]}_'
                                           f'{target}_{value_of_interest}.csv')).set_index('comp', drop=False)
    
    if generate_scatter_plot:
        for i, comp in enumerate(comp_to_plot):
            # Load the predicted component data
            comp_data_pred = load_pred_comp_responses(output_folder_path=df_best_layer_comp.query(f'comp == "{comp}"').output_folder.values[0],
                                                      target=target, )

            # Get min and max of component responses and predicted component responses
            min_resp = min(comp_data[comp].min(), comp_data_pred[comp].min())
            max_resp = max(comp_data[comp].max(), comp_data_pred[comp].max())
            # Offset the min and max by 5% of the range for axis limits
            min_resp -= 0.05 * (max_resp - min_resp)
            max_resp += 0.05 * (max_resp - min_resp)
            
            assert(comp_data.index == comp_data_pred.index).all()
            comp_data['category_label'] = comp_data_pred.category_label
    
            color_order = [d_sound_category_colors[x] for x in comp_data_pred.category_label.values]
            
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.set_box_aspect(1) # ensure square axes
            plt.scatter(comp_data[comp].values, comp_data_pred[comp].values, c=color_order, alpha=alpha_dot)
            plt.title(
                f'Component {i+1}: "{d_comp_names[comp]}-selective" {d_randnetw[randnetw][1:]}\n'
                f'{df_best_layer_comp.loc[comp].source_model} {df_best_layer_comp.loc[comp].layer_legend},'
                f' median $R^2$: {df_agg_scores.loc[comp][value_of_interest]:.2}', fontsize=11)
            plt.xlabel('Actual', fontsize=15)
            plt.xticks(size=15)
            plt.yticks(size=15)
            plt.xlim(min_resp, max_resp)
            plt.ylim(min_resp, max_resp)
            plt.ylabel('Predicted', fontsize=15)
            classes = list(d_sound_category_colors.keys())
            classes_polished = [d_sound_category_names[x] for x in classes]
            class_colours = list(d_sound_category_colors.values())
            recs = []
            for i in range(0, len(class_colours)):
                recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
            plt.legend(recs, classes_polished, bbox_to_anchor=(1, 1.))
            # Plot the layer name (df_best_layer_comp.loc[comp].layer_legend) in the top left corner of the axes, with a bit of offset
            plt.text(min_resp + 0.035 * (max_resp - min_resp), max_resp - 0.07 * (max_resp - min_resp),
                     df_best_layer_comp.loc[comp].layer_legend, fontsize=15)
            # Plot the R2 value in the lower right corner of the axes, with a bit of offset
            plt.text(max_resp - 0.27 * (max_resp - min_resp), min_resp + 0.02 * (max_resp - min_resp),
                     f'$R^2$: {df_agg_scores.loc[comp][value_of_interest]:.2}', fontsize=15)
            plt.tight_layout()
            if save:
                save_str = f'scatter-comp-{comp}_resp-vs-pred_' \
                           f'{df_best_layer_comp.loc[comp].source_model}{d_randnetw[randnetw]}_' \
                           f'{df_best_layer_comp.loc[comp].layer_pos}_' \
                           f'{target}_{value_of_interest}{add_savestr}'
                plt.savefig(join(save, f'{save_str}.svg'), dpi=180)
                # plt.savefig(join(save, f'{save_str}.png'), dpi=180)
            plt.show()
        
    if obtain_partial_corr: # correlate only e.g. only the speech category sounds
        lst_partial_corr_category = []
        lst_partial_corr_category_broad = []
        
        for i, comp in enumerate(comp_to_plot):
            # Load the predicted component data
            comp_data_pred = load_pred_comp_responses(output_folder_path=df_best_layer_comp.query(f'comp == "{comp}"').output_folder.values[0],
                target=target, )
        
            assert (comp_data.index == comp_data_pred.index).all()
            comp_data['category_label'] = comp_data_pred.category_label
            # Append a more broad category label!
            comp_data['category_label_broad'] = comp_data.category_label.apply(lambda x: d_sound_category_broad[x])
            comp_data_pred['category_label_broad'] = comp_data_pred.category_label.apply(lambda x: d_sound_category_broad[x])
            
            # Sanity check plot
            # color_order = [d_sound_category_colors[x] for x in comp_data_pred.category_label.values]
            # fig, ax = plt.subplots(figsize=(7, 5))
            # ax.set_box_aspect(1)  # ensure square axes
            # plt.scatter(comp_data[comp].values, comp_data_pred[comp].values, c=color_order)
            # plt.show()
            
            # Obtain partial correlation for each sound category
            for category in d_sound_category_colors.keys():
                # Get the sound category data
                comp_data_category = comp_data.query(f'category_label == "{category}"')
                comp_data_pred_category = comp_data_pred.query(f'category_label == "{category}"')
                
                # Correlate for the component of interest
                r, p = stats.pearsonr(comp_data_category[comp].values, comp_data_pred_category[comp].values)
                # r = np.corrcoef(comp_data_category[comp].values, comp_data_pred_category[comp].values)[0, 1]
                r2 = r**2
                
                # Also obtain mean actual and predicted values for the component of interest
                mean_actual = np.mean(comp_data_category[comp].values)
                mean_pred = np.mean(comp_data_pred_category[comp].values)
                
                # Store to a df and then csv
                lst_partial_corr_category.append({'comp': comp,
                                           'category_polished': d_sound_category_names[category],
                                          'category': category,
                                          'r': r,
                                          'r2': r2,
                                            'p': p,
                                            'num_sounds': len(comp_data_category[comp].values),
                                          'mean_actual': mean_actual,
                                          'mean_pred': mean_pred,
                                          'output_folder_path': df_best_layer_comp.query(
                                            f'comp == "{comp}"').output_folder.values[0]}, )
                
            # Do the same for broad categories
            for category in np.unique(list(d_sound_category_broad.values())):
                # Get the sound category data
                comp_data_category = comp_data.query(f'category_label_broad == "{category}"')
                comp_data_pred_category = comp_data_pred.query(f'category_label_broad == "{category}"')
                
                # Correlate for the component of interest
                r, p = stats.pearsonr(comp_data_category[comp].values, comp_data_pred_category[comp].values)
                r2 = r**2
                
                # Also obtain mean actual and predicted values for the component of interest
                mean_actual = np.mean(comp_data_category[comp].values)
                mean_pred = np.mean(comp_data_pred_category[comp].values)
                
                # Store to a df and then csv
                lst_partial_corr_category_broad.append({'comp': comp,
                                           'category_polished': category,
                                             'category': category,
                                             'r': r,
                                             'r2': r2,
                                             'p': p,
                                              'num_sounds': len(comp_data_category[comp].values),
                                              'mean_actual': mean_actual,
                                              'mean_pred': mean_pred,
                                              'output_folder_path': df_best_layer_comp.query(f'comp == "{comp}"').output_folder.values[0]},)
                
        df_partial_corr_category_broad = pd.DataFrame(lst_partial_corr_category_broad)
        df_partial_corr_category = pd.DataFrame(lst_partial_corr_category)

        # Merge the two csvs
        df_partial_corr = pd.concat([df_partial_corr_category, df_partial_corr_category_broad])
        df_partial_corr['source_model'] = source_model
        df_partial_corr['target'] = target
        df_partial_corr['randnetw'] = randnetw
        
        if save:
            save_str =  f'partial_corrs_comp-all_resp-vs-pred_' \
                       f'{source_model}{d_randnetw[randnetw]}_' \
                       f'{target}_{value_of_interest}{add_savestr}'
            df_partial_corr.to_csv(os.path.join(save, save_str + '.csv'))


def load_comp_responses():
    """Load the component responses from mat file"""
    
    ## Stimuli (original indexing, activations are extracted in this order) ##
    sound_meta = np.load(os.path.join(DATADIR, f'neural/NH2015/neural_stim_meta.npy'))

    # Extract wav names in order (this is how the neural data is ordered --> order all activations in the same way)
    stimuli_IDs = []
    for i in sound_meta:
        stimuli_IDs.append(i[0][:-4].decode("utf-8")) # remove .wav

    comp_data = loadmat(os.path.join(DATADIR, f'neural/NH2015comp/components.mat'))
    comp_stimuli_IDs = comp_data['stim_names']
    comp_stimuli_IDs = [x[0] for x in comp_stimuli_IDs[0]]
    
    comp_names = comp_data['component_names']
    comp_names = [x[0][0] for x in comp_names]
    
    comp = comp_data['R']  # R is the response matrix
    
    # add to df with index stimuli IDs and component names as columns
    df_comp = pd.DataFrame(comp, index=comp_stimuli_IDs, columns=comp_names)
    
    # reindex so it is indexed the same way as the neural data and activations (stimuli_IDs)
    voxel_data = df_comp.reindex(stimuli_IDs)
    
    return voxel_data

def load_pred_comp_responses(output_folder_path,
                             target,
                             load_rescaled=True):
    """Load the predicted component responses which is a 3D np array:
    Sounds (165); CV splits (10); components (6)
    
    Given that not all sounds are in each split, obtain the nanmean across splits.

    If load_rescaled is True, load the rescaled version of the predicted component responses (which means that we added
    the mean back, such that the test values are not zero-centered). This is the default because the axes between true
    component values and predicted components values align better.

    """
    if output_folder_path.startswith('/rdma/vast-') and user == 'gt': # om path, fix to local path if running locally
        output_folder_path = output_folder_path.replace('/rdma/vast-rdma/vast/evlab/gretatu/', ROOT)

    if load_rescaled:
        suffix = '_rescaled'
    else:
        suffix = ''

    y_preds_test_data = pd.read_pickle(join(output_folder_path, f'y_preds_test{suffix}.pkl'))
    
    # Given that not all sounds are in each test split, obtain the nanmean across splits.
    y_preds_test = np.nanmean(y_preds_test_data, axis=1)
    
    # Make it into a df with metadata:
    df_meta_roi = pd.read_pickle(join(DATADIR, 'neural', target, 'df_roi_meta.pkl'))
    df_stimuli_meta = load_stimuli_metadata()

    df = pd.DataFrame(y_preds_test, index=df_stimuli_meta.index, columns=df_meta_roi.comp)
    
    # Append the sound category metadata:
    df['category_label'] = df_stimuli_meta.category_label
    
    return df
    # np.corrcoef(voxel_data['pitch'].values, g[:, 3])
    

def load_stimuli_metadata():
    df_stimuli_meta = pd.read_pickle(os.path.join(DATADIR, f'stimuli/df_stimuli_meta.pkl'))
    return df_stimuli_meta


def obtain_output_folders_for_best_comp_layers(target, value_of_interest='median_r2_test',
                                               mapping='Ridge', randemb='False', randnetw='False', alphalimit='50',
                                               save=True):
    """Get strings of the output folder names for the best model/layer for each component.
    Requires find_best_layer_per_component() to be run first.
    Across models.
    """
    
    df = pd.read_csv(join(SAVEDIR_CENTRALIZED, f'best-layer-per-comp_{value_of_interest}_{target}_{value_of_interest}.csv'))
    df = df.set_index('component').reindex(['lowfreq', 'highfreq', 'envsounds', 'pitch', 'speech', 'music'])
    
    output_folders = [join(RESULTDIR_ROOT, x[1].model,
          f'AUD-MAPPING-{mapping}_TARGET-{target}_SOURCE-{x[1].model}-{x[1].layer}_RANDEMB-{randemb}_RANDNETW-{randnetw}_ALPHALIMIT-{alphalimit}') for x in df.iterrows()]
    
    df['output_folder'] = output_folders
    
    if save: # store new version w output folder
        df.to_csv(join(SAVEDIR_CENTRALIZED, f'best-layer-per-comp_{value_of_interest}_{target}_{value_of_interest}_w_outputfolder.csv'))
    
    return df

def obtain_output_folders_for_any_comp_model(target,
                                             source_model,
                                             value_of_interest='median_r2_test',
                                             mapping='Ridge',
                                             randnetw='False',
                                             alphalimit='50',):
    """Get strings of the output folder names (return as df) for a given source model of interest.
    Take the best layer for each component for that source model. Return which output folders (i.e.,
    which layer it was"""
    
    if source_model == 'spectemp':
        df = obtain_NH2015comp_spectemp_val(target=target, value_of_interest=value_of_interest)
    else:
        raise ValueError(f'Make sure which output CV you want to load!')
        df = pd.read_csv(
                join(RESULTDIR_ROOT, source_model, 'outputs', f'best-layer-per-comp_{source_model}{d_randnetw[randnetw]}_{target}.csv')).rename(
            columns={'Unnamed: 0': 'component'})
    
    # Make sure that component order is as expected
    df = df.set_index('comp').reindex(['lowfreq', 'highfreq', 'envsounds', 'pitch', 'speech', 'music'])
    
    output_folders = [join(RESULTDIR_ROOT,
                           source_model,
                           f'AUD-MAPPING-{mapping}_'
                           f'TARGET-{target}_'
                           f'SOURCE-{source_model}-{x[1].source_layer}_'
                           f'RANDNETW-{randnetw}_ALPHALIMIT-{alphalimit}')
                           for x in df.iterrows()]

    df['output_folder'] = output_folders
    
    return df



#### ACROSS MODELS FUNCTIONS ####
def barplot_across_models(source_models,
                          target,
                          roi=None,
                          value_of_interest='median_r2_test_c',
                          randnetw='False',
                          save=None,
                          aggregation='CV-splits-nit-10',
                          sort_by=True,
                          yerr_type='within_subject_sem',
                          add_savestr='',
                          alpha=1,
                          box_aspect=0.8,
                          grouped_bars=False):
    """
    Plot median variance explained across models for voxels in a given ROI or all ROIs.
    The score is loaded using CV-splits-nit-10 (default) layer selection procedure (can be changed if the user
    stored csv files with other layer selection procedures).

    If grouped_bars is True, then the bars are grouped two and two (assumes 8 source models for the clean speech networks.

    :param source_models:
    :param roi:
    :param value_of_interest:
    :return:
    """
    # Obtain LOSO/voxelwise scores for the ROI of interest!
    df_lst = []
    for source_model in source_models:
        if aggregation == 'best_voxelwise':
            load_str = f'best-layer_voxelwise_{source_model}{d_randnetw[randnetw]}_{target}_{value_of_interest}.csv'
        elif aggregation == 'LOSO':
            load_str = f'best-layer_LOSO_roi-{roi}_{source_model}{d_randnetw[randnetw]}_{target}_{value_of_interest}.csv' # obs _{value_of_interest} only appended 20220108
        elif aggregation.startswith('CV'):
            load_str = f'best-layer_{aggregation}_roi-{roi}_{source_model}{d_randnetw[randnetw]}_{target}_{value_of_interest}.csv'
        else:
            raise ValueError(f'aggregation {aggregation} not recognized')
        
        df = pd.read_csv(join(RESULTDIR_ROOT,
                               source_model,
                              'outputs',
                               load_str))
        df.rename(columns={'Unnamed: 0': 'subj_idx'}, inplace=True)
        # Sometimes subj_idx already exists, so if it is repeated, drop it
        df = df.loc[:,~df.columns.duplicated()]
        
        df_lst.append(df)
    
    df_all = pd.concat(df_lst)

    if roi == 'all': # extract specific rois
        for roi_col in df_all.roi.unique():
            piv_spectemp = obtain_spectemp_val(roi=roi_col, target=target)
            
            df_all = df_all.query(f'`roi` == "{roi_col}"')
            df_all = df_all.append(piv_spectemp)

            ## Obtain mean and within-subject error bar (i.e. within-subject error bar)
            # for each cond (model), subtract the mean across cond. then, each subject will have a demeaned array
            if yerr_type.startswith('within_subject'):
                # 1. Get subject x model pivot (no aggregation going on here)
                # df_all['subj_idx'] = df_all.index
                piv = df_all.pivot_table(index='subj_idx', columns='source_model', values=value_of_interest)

                # 2. Obtain error bar (within-subject error bar) by subtracting the mean across cond (models) for each subject
                demeaned = piv.subtract(piv.mean(axis=1).values,
                                        axis=0)  # get a mean value for each subject across all conds (piv.mean(axis=1), and subtract it from individual models

                if yerr_type == 'within_subject_sem':
                    yerr = np.std(demeaned.values.T, axis=1) / np.sqrt(piv.shape[0] - 1)

                # piv.shape[0] is the number of subjects (rows)
                if yerr_type == 'within_subject_std':
                    yerr = np.std(demeaned.values.T, axis=1)

                # 3. Now obtain the mean across models
                mean_per_model = piv.mean(axis=0)  # the same as taking mean and getting across-subject SEM as below (only the error bar differs)

                # 4. Zip into one dataframe and sort by performance
                df_grouped = pd.DataFrame({f'{value_of_interest}_mean': mean_per_model.values,
                                           f'{value_of_interest}_yerr': yerr, 'source_model': mean_per_model.index},
                                          index=mean_per_model.index)

            if yerr_type == 'across_subject_sem':
                ## Obtain mean and SEM across subjects (i.e. across-subject error bar)
                df_grouped = df_all.groupby('source_model').agg({value_of_interest: ['mean', 'sem']}).reset_index()
    
                # index according to source models
                df_grouped = df_grouped.set_index('source_model').reindex(source_models)
    
                # flatten multiindex
                df_grouped.columns = ['_'.join(col).strip() for col in df_grouped.columns.values]
                df_grouped.rename(
                    columns={'source_model_': 'source_model', f'{value_of_interest}_sem': f'{value_of_interest}_yerr'},
                    inplace=True)

            if sort_by == 'performance':
                df_grouped = df_grouped.sort_values(f'{value_of_interest}_mean', ascending=False)

            df_grouped_w_spectemp = df_grouped.copy(deep=True)

            # drop the spectemp row (we want to plot it separately)
            df_spectemp = df_grouped.loc[df_grouped['source_model'] == 'spectemp']
            df_grouped.drop(df_grouped.index[df_grouped['source_model'] == 'spectemp'], inplace=True)

            # plot specs
            bar_placement = np.arange(0, len(df_grouped) / 2, 0.5)
            color_order = [d_model_colors[x] for x in df_grouped.index.values]

            title_str = f'Median noise-corrected $R^2$ across models\nROI: {roi_col}, {target} {d_randnetw[randnetw][1:]} ({aggregation})'
            
            # Obtain xmin and xmax for spectemp line
            xmin = np.unique((bar_placement[0] - np.diff(bar_placement) / 2))
            xmax = np.unique((bar_placement[-1] + np.diff(bar_placement) / 2))

            fig, ax = plt.subplots(figsize=(6, 7)) # change to ax.set_box_aspect(0.9? or like in comp plot?)
            ax.hlines(xmin=xmin, xmax=xmax, y=df_spectemp[f'{value_of_interest}_mean'].values, color='darkgrey',
                      zorder=2)
            plt.fill_between(
                [(bar_placement[0] - np.diff(bar_placement) / 2)[0],
                 (bar_placement[-1] + np.diff(bar_placement) / 2)[0]],
                df_spectemp[f'{value_of_interest}_mean'].values - df_spectemp[f'{value_of_interest}_yerr'].values,
                # plot yerr for spectemp too
                df_spectemp[f'{value_of_interest}_mean'].values + df_spectemp[f'{value_of_interest}_yerr'].values,
                color='gainsboro')
            ax.bar(bar_placement, df_grouped[f'{value_of_interest}_mean'].values,
                   yerr=df_grouped[f'{value_of_interest}_yerr'].values,
                   width=0.3, color=color_order, zorder=2, alpha=alpha)
            plt.xticks(bar_placement, df_grouped.index.values, rotation=80)
            plt.ylim([0, 1])
            plt.ylabel(d_value_of_interest[value_of_interest])
            plt.title(title_str)
            plt.tight_layout()
            if save:
                plt.savefig(join(save,
                                 f'across-models_roi-{roi_col}_{target}{d_randnetw[randnetw]}_{aggregation}_{yerr_type}_{value_of_interest}.png'), dpi=180)
                plt.savefig(join(save,
                                 f'across-models_roi-{roi_col}_{target}{d_randnetw[randnetw]}_{aggregation}_{yerr_type}_{value_of_interest}.svg'), dpi=180)
    
                # save csv
                df_grouped_w_spectemp.to_csv(join(save,
                                                  f'across-models_roi-{roi_col}_{target}{d_randnetw[randnetw]}_{aggregation}_{yerr_type}_{value_of_interest}.csv'))
            fig.show()
        
    else: # For all voxels and voxels in any ROI
        if aggregation.startswith('CV-splits-nit'): # also obtain spectemp value based on random splits
            load_str = f'best-layer_{aggregation}_roi-{roi}_' \
                       f'spectemp{d_randnetw[randnetw]}_{target}_{value_of_interest}.csv'

            piv_spectemp = pd.read_csv(join(RESULTDIR_ROOT,
                                            'spectemp',
                                            'outputs',
                                            load_str))
            # Sometimes subj_idx already exists, so if it is repeated, drop it
            piv_spectemp = piv_spectemp.loc[:, ~piv_spectemp.columns.duplicated()].drop(
                columns=['nit.1', 'subj_idx.1'])

        else:
            piv_spectemp = obtain_spectemp_val(roi=roi, target=target, value_of_interest=value_of_interest)

        df_all = df_all.append(piv_spectemp)

        ## Obtain mean and within-subject error bar (i.e. within-subject error bar)
        # for each cond (model), subtract the mean across cond. then, each subject will have a demeaned array
        if yerr_type.startswith('within_subject'):
            # 1. Get subject x model pivot (no aggregation going on here)
            piv = df_all.pivot_table(index='subj_idx', columns='source_model', values=value_of_interest)
            
            # 2. Obtain error bar (within-subject error bar) by subtracting the mean across cond (models) for each subject
            demeaned = piv.subtract(piv.mean(axis=1).values, axis=0)  # get a mean value for each subject across all conds (piv.mean(axis=1), and subtract it from individual models
            
            if yerr_type == 'within_subject_sem':
                yerr = np.std(demeaned.values.T, axis=1) / np.sqrt(piv.shape[0] - 1)
    
            # piv.shape[0] is the number of subjects (rows)
            if yerr_type == 'within_subject_std':
                yerr = np.std(demeaned.values.T, axis=1)
            
            # 3. Now obtain the mean across models
            mean_per_model = piv.mean(axis=0) # the same as taking mean and getting across-subject SEM as below (only the error bar differs)
            
            # 4. Zip into one dataframe and sort by performance
            df_grouped = pd.DataFrame({f'{value_of_interest}_mean': mean_per_model.values,
                  f'{value_of_interest}_yerr': yerr, 'source_model':mean_per_model.index},
                  index=mean_per_model.index)
        
        if yerr_type == 'across_subject_sem':
            ## Obtain mean and SEM across subjects (i.e. across-subject error bar)
            df_grouped = df_all.groupby('source_model').agg({value_of_interest: ['mean', 'sem']}).reset_index()
            
            # index according to source models
            df_grouped = df_grouped.set_index('source_model').reindex(source_models)
            
            # flatten multiindex
            df_grouped.columns = ['_'.join(col).strip() for col in df_grouped.columns.values]
            df_grouped.rename(columns={'source_model_': 'source_model', f'{value_of_interest}_sem': f'{value_of_interest}_yerr'},
                              inplace=True)

        df_grouped_w_spectemp = df_grouped.copy(deep=True)

        # drop the spectemp row (we want to plot it separately)
        df_spectemp = df_grouped.loc[df_grouped['source_model'] == 'spectemp']
        df_grouped.drop(df_grouped.index[df_grouped['source_model'] == 'spectemp'], inplace=True)
        
        # Sort
        if sort_by == 'performance':
            sort_str = '_performance_sorted'
            df_grouped = df_grouped.sort_values(by=f'{value_of_interest}_mean', ascending=False)
        if type(sort_by) == list:
            sort_str = '_manually_sorted'
            df_grouped = df_grouped.set_index('source_model', inplace=False, drop=False)
            df_grouped = df_grouped.reindex(sort_by)
        
        # plot specs
        if grouped_bars:
            # Group first two bars and last two bars together
            offset = 0.35
            bar_placement = [0, 0.35, 0.7 + offset, 1.05 + offset, 1.4 + offset * 2, 1.75 + offset * 2,
                             2.1 + offset * 3, 2.45 + offset * 3]
        else:
            bar_placement = np.arange(0, len(df_grouped) / 2, 0.5)
        color_order = [d_model_colors[x] for x in df_grouped.index.values]
        model_legend = [d_model_names[x] for x in df_grouped.index.values]
        
        if roi == 'any':
            title_str = f'{d_value_of_interest[value_of_interest]} across models\nVoxels in any ROI, {target} {d_randnetw[randnetw][1:]} ({aggregation})'
        else:  # None
            title_str = f'{d_value_of_interest[value_of_interest]} across models\nAll voxels, {target} {d_randnetw[randnetw][1:]} ({aggregation})'
        
        # Obtain xmin and xmax for spectemp line
        if grouped_bars:
            # Get xmin and xmax which should be +- 0.25 of the min and max of bar_placement
            xmin = np.min(bar_placement) - 0.25
            xmax = np.max(bar_placement) + 0.25
        else:
            xmin = np.unique((bar_placement[0] - np.diff(bar_placement) / 2))
            xmax = np.unique((bar_placement[-1] + np.diff(bar_placement) / 2))


        fig, ax = plt.subplots(figsize=(6, 7.5))
        ax.set_box_aspect(box_aspect) # 0.8 for all barplots, 1 for clean speech
        ax.hlines(xmin=xmin, xmax=xmax,
                  y=df_spectemp[f'{value_of_interest}_mean'].values, color='darkgrey',
                  zorder=2)
        plt.fill_between(
            [(bar_placement[0] - np.diff(bar_placement) / 2)[0],
             (bar_placement[-1] + np.diff(bar_placement) / 2)[0]],
            df_spectemp[f'{value_of_interest}_mean'].values - df_spectemp[f'{value_of_interest}_yerr'].values,
            # plot yerr for spectemp too
            df_spectemp[f'{value_of_interest}_mean'].values + df_spectemp[f'{value_of_interest}_yerr'].values,
            color='gainsboro')
        ax.bar(bar_placement, df_grouped[f'{value_of_interest}_mean'].values,
               yerr=df_grouped[f'{value_of_interest}_yerr'].values,
               width=0.3, color=color_order, zorder=2, alpha=alpha)
        # xticks dont really align, fix
        plt.xticks(bar_placement, model_legend, rotation=80, fontsize=13,
                   ha='right',rotation_mode='anchor')
        plt.ylim([0, 1])
        plt.ylabel(d_value_of_interest[value_of_interest], fontsize=15)
        plt.yticks(fontsize=15)
        plt.title(title_str)
        plt.tight_layout(pad=2.5)
        if save:
            save_str = f'across-models_roi-{roi}_{target}{d_randnetw[randnetw]}_' \
                       f'{aggregation}_{yerr_type}_' \
                       f'{value_of_interest}{sort_str}{add_savestr}'
            plt.savefig(join(save, f'{save_str}.png'), dpi=180)
            plt.savefig(join(save, f'{save_str}.svg'), dpi=180)

            # save csv and log more info
            df_grouped_w_spectemp['roi'] = roi
            df_grouped_w_spectemp['target'] = target
            df_grouped_w_spectemp['randnetw'] = randnetw
            df_grouped_w_spectemp['aggregation'] = aggregation
            df_grouped_w_spectemp['yerr_type'] = yerr_type
            df_grouped_w_spectemp['value_of_interest'] = value_of_interest
            df_grouped_w_spectemp['sort_by'] = [sort_by] * len(df_grouped_w_spectemp)
            df_grouped_w_spectemp['add_savestr'] = add_savestr
            df_grouped_w_spectemp['n_models'] = len(df_grouped_w_spectemp)
            df_grouped_w_spectemp.to_csv(join(save, f'{save_str}.csv'))
        fig.show()


def scatter_across_models(source_models,
                          models1,
                          models2,
                          target,
                          roi=None,
                          value_of_interest='median_r2_test_c',
                          randnetw='False',
                          save=None,
                          aggregation='CV-splits-nit-10',
                          ylim=[0.6,0.8],
                          xlim=[0.6,0.8],
                          yerr_type='within_subject_sem',
                          add_savestr=''):
    """
    Plot median variance explained across models for all voxels.
    The score is loaded using CV-splits-nit-10 (default) layer selection procedure.

    Intended for checking whether Seed1 vs Seed2 models show similar variance explained.

    Models1 should contain the models to go on the x-axis, models2 on the y-axis.

    :param source_models:
    :param roi:
    :param value_of_interest:
    :return:
    """
    # Obtain LOSO/voxelwise scores for the ROI of interest!
    df_lst = []
    for source_model in source_models:
        if aggregation == 'best_voxelwise':
            load_str = f'best-layer_voxelwise_{source_model}{d_randnetw[randnetw]}_{target}_{value_of_interest}.csv'
        elif aggregation == 'LOSO':
            load_str = f'best-layer_LOSO_roi-{roi}_{source_model}{d_randnetw[randnetw]}_{target}_{value_of_interest}.csv'  # obs _{value_of_interest} only appended 20220108
        elif aggregation.startswith('CV'):
            load_str = f'best-layer_{aggregation}_roi-{roi}_{source_model}{d_randnetw[randnetw]}_{target}_{value_of_interest}.csv'
        else:
            raise ValueError(f'aggregation {aggregation} not recognized')

        df = pd.read_csv(join(RESULTDIR_ROOT,
                              source_model,
                              'outputs',
                              load_str))
        df.rename(columns={'Unnamed: 0': 'subj_idx'}, inplace=True)
        # Sometimes subj_idx already exists, so if it is repeated, drop it
        df = df.loc[:, ~df.columns.duplicated()]

        df_lst.append(df)

    df_all = pd.concat(df_lst)

    # For all voxels and voxels in any ROI

    ## Obtain mean and within-subject error bar (i.e. within-subject error bar)
    # for each cond (model), subtract the mean across cond. then, each subject will have a demeaned array
    if yerr_type.startswith('within_subject'):
        # 1. Get subject x model pivot (no aggregation going on here)
        piv = df_all.pivot_table(index='subj_idx', columns='source_model', values=value_of_interest)

        # 2. Obtain error bar (within-subject error bar) by subtracting the mean across cond (models) for each subject
        demeaned = piv.subtract(piv.mean(axis=1).values,
                                axis=0)  # get a mean value for each subject across all conds (piv.mean(axis=1), and subtract it from individual models

        if yerr_type == 'within_subject_sem':
            yerr = np.std(demeaned.values.T, axis=1) / np.sqrt(piv.shape[0] - 1)

        # piv.shape[0] is the number of subjects (rows)
        if yerr_type == 'within_subject_std':
            yerr = np.std(demeaned.values.T, axis=1)

        # 3. Now obtain the mean across models
        mean_per_model = piv.mean(
            axis=0)  # the same as taking mean and getting across-subject SEM as below (only the error bar differs)

        # 4. Zip into one dataframe and sort by performance
        df_grouped = pd.DataFrame({f'{value_of_interest}_mean': mean_per_model.values,
                                   f'{value_of_interest}_yerr': yerr, 'source_model': mean_per_model.index},
                                  index=mean_per_model.index)

    if yerr_type == 'across_subject_sem':
        ## Obtain mean and SEM across subjects (i.e. across-subject error bar)
        df_grouped = df_all.groupby('source_model').agg({value_of_interest: ['mean', 'sem']}).reset_index()

        # index according to source models
        df_grouped = df_grouped.set_index('source_model').reindex(source_models)

        # flatten multiindex
        df_grouped.columns = ['_'.join(col).strip() for col in df_grouped.columns.values]
        df_grouped.rename(
            columns={'source_model_': 'source_model', f'{value_of_interest}_sem': f'{value_of_interest}_yerr'},
            inplace=True)

    # Take the models in models1 and models2 and put them on the x and y axis, respectively. Retain the order of models1 and models2.
    assert len(models1) == len(models2), 'models1 and models2 should have the same length'
    assert all([x in df_grouped['source_model'].values for x in models1]), 'models1 should be a subset of source_models'
    assert all([x in df_grouped['source_model'].values for x in models2]), 'models2 should be a subset of source_models'

    df_models1 = df_grouped[df_grouped['source_model'].isin(models1)].reindex(models1)
    df_models2 = df_grouped[df_grouped['source_model'].isin(models2)].reindex(models2)

    # plot specs
    color_order_models1 = [d_model_colors[x] for x in df_models1.index.values]
    color_order_models2 = [d_model_colors[x] for x in df_models2.index.values]
    assert color_order_models1 == color_order_models2, 'color order should be the same for models1 and models2'

    title_str = f'{d_value_of_interest[value_of_interest]} across models\nAll voxels, {target} {d_randnetw[randnetw][1:]} ({aggregation})'

    # Plot scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_box_aspect(1)
    # Plot scatter with error bars. Iterate over each point because otherwise we can't have different colors for each point
    for i in range(len(df_models1)):
        # Transform color to RGBA
        color = mcolors.to_rgba(color_order_models1[i])
        ax.errorbar(df_models1[f'{value_of_interest}_mean'].values[i],
                    df_models2[f'{value_of_interest}_mean'].values[i],
                    xerr=df_models1[f'{value_of_interest}_yerr'].values[i],
                    yerr=df_models2[f'{value_of_interest}_yerr'].values[i],
                    fmt='o', markersize=10, color=color,
                    capsize=0, elinewidth=2, markeredgewidth=2)
    # Also plot correlation r between models1 and models2
    r, p = stats.pearsonr(df_models1[f'{value_of_interest}_mean'].values, df_models2[f'{value_of_interest}_mean'].values)
    r2 = r ** 2
    # Plot in lower right
    plt.text(0.95, 0.05, f'$R^2$={r2:.2f}, p={p:.2f}', horizontalalignment='right', verticalalignment='bottom',
             transform=ax.transAxes, fontsize=15)
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    plt.xlabel(f'{d_value_of_interest[value_of_interest]} Seed 1', fontsize=15)
    plt.ylabel(f'{d_value_of_interest[value_of_interest]} Seed 2', fontsize=15)
    # Add legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=x, markerfacecolor=mcolors.to_rgba(d_model_colors[x]),
                              markersize=10) for x in df_models1.index.values]
    # Plot legend outside of plot
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    # Make ticks bigger
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(title_str)
    add_identity(ax, color='black', ls='--')
    plt.tight_layout(pad=2.5)
    if save:
        save_str = f'across-models_roi-{roi}_{target}{d_randnetw[randnetw]}_' \
                   f'{aggregation}_{yerr_type}_' \
                   f'n-models={len(df_grouped)}_' \
                   f'{value_of_interest}{add_savestr}'
        plt.savefig(join(save, f'{save_str}.png'), dpi=180)
        plt.savefig(join(save, f'{save_str}.svg'), dpi=180)

        # save csv and log more info
        df_grouped['roi'] = roi
        df_grouped['target'] = target
        df_grouped['randnetw'] = randnetw
        df_grouped['aggregation'] = aggregation
        df_grouped['yerr_type'] = yerr_type
        df_grouped['value_of_interest'] = value_of_interest
        df_grouped['add_savestr'] = add_savestr
        df_grouped['n_models'] = len(df_grouped)
        df_grouped['models1'] = [models1] * len(df_grouped)
        df_grouped['models2'] = [models2] * len(df_grouped)
        df_grouped['r'] = r
        df_grouped['r2'] = r2
        df_grouped['p'] = p
        df_grouped.to_csv(join(save, f'{save_str}.csv'))
    fig.show()


def scatter_anat_roi_across_models(source_models,
                                   target,
                                   randnetw='False',
                                   save=True,
                                   annotate=False,
                                   condition_col='roi_label_general',
                                   collapse_over_val_layer='median',
                                   primary_rois=['Primary'],
                                   non_primary_rois=['Anterior'],
                                   yerr_type='within_subject_sem',
                                   save_str='',
                                   value_of_interest='median_r2_test_c',
                                   layer_value_of_interest='rel_pos',
                                   layers_to_exclude=None,
                                   stats=True,
                                   alpha=1):
    """Plot a scatter of the best relative layer position (where each scatter point is a model) as obtained in
    the function barplot_best_layer_per_anat_ROI. The best layer was obtained for each subject, by taking the
    median across relative layer positions, and then we obtain the mean across subjects in this function.
    The error is within-subject SEM (with condition as models)
    
    Example call:
	scatter_anat_roi_across_models(source_models, target=target, save=SAVEDIR_CENTRALIZED, randnetw=randnetw,
								   condition_col='roi_label_general',
								   primary_rois=['Primary'],
								   non_primary_rois=['Anterior'], annotate=True)
    
    :param output: pd df, output df
    :param meta: pd df, meta df
    :param source_model: string, source model
    :param target: string, target model
    :param val: string, metric to use to obtain the argmax across layers
    :param val_layer: string, metric to use in plotting the argmax. 'pos' is the actual position (int) of the argmax layer,
                    while 'rel_pos' is the relative position [0 1]
                    'pos' is NOT zero indexed, i.e. 1 is the first layer, and if the model has 11 layers, then 11 is the top layer.

    :return: plot

    """
    if layers_to_exclude:
        layers_exclusion_savestr = f'_layer-exclude-{"-".join(layers_to_exclude)}' # suffix with this
    else:
        layers_exclusion_savestr = '' # otherwise no suffix
    
    # Obtain argmax best layer scores across models!
    lst_primary = []
    lst_non_primary = []
    
    for source_model in source_models:
        if source_model.startswith('Kell2018') or source_model.startswith('ResNet50'):
            if layers_to_exclude:
                layer_exlusion_str = '-'.join(layers_to_exclude)
            else:
                layer_exlusion_str = 'None'
        else:
            layer_exlusion_str = 'None'
                
        df = pd.read_csv(
            join(RESULTDIR_ROOT,
                 source_model, 'outputs',
                 f'best-layer_barplot_{condition_col}_'
                 f'{source_model}{d_randnetw[randnetw]}_'
                 f'{target}_{yerr_type}_'
                 f'{layer_value_of_interest}-{collapse_over_val_layer}_'
                 f'{value_of_interest}_'
                 f'layer-exclude-{layer_exlusion_str}.csv')).rename(columns={'Unnamed: 0': 'index'}).set_index('index')
        
        df.drop(index='yerr', inplace=True) # drop the within-subject error across conditions. We want across models within-subject error.
        assert(len(primary_rois) == 1 & len(non_primary_rois) == 1)
        
        lst_primary.append(df[primary_rois].rename(columns={primary_rois[0]:source_model}))
        lst_non_primary.append(df[non_primary_rois].rename(columns={non_primary_rois[0]:source_model}))

    df_primary_across_models = pd.concat(lst_primary, axis=1)
    df_non_primary_across_models = pd.concat(lst_non_primary, axis=1)

    # 2. Obtain error bar (within-subject error bar) by subtracting the mean across cond (models) for each subject
    primary_demeaned = df_primary_across_models.subtract(df_primary_across_models.mean(axis=1).values, axis=0)  # get a mean value for each subject across all conds (piv.mean(axis=1), and subtract it from individual models
    non_primary_demeaned = df_non_primary_across_models.subtract(df_non_primary_across_models.mean(axis=1).values, axis=0)
    
    if yerr_type == 'within_subject_sem':
        primary_yerr = np.std(primary_demeaned.values.T, axis=1) / np.sqrt(df_primary_across_models.shape[0] - 1)
    # piv.shape[0] is the number of subjects (rows)
    non_primary_yerr = np.std(non_primary_demeaned.values.T, axis=1) / np.sqrt(df_non_primary_across_models.shape[0] - 1)
    
    if yerr_type == 'within_subject_std':
        primary_yerr = np.std(primary_demeaned.values.T, axis=1)
        non_primary_yerr = np.std(non_primary_demeaned.values.T, axis=1)

    # 3. Now obtain the mean across subjects for each model
    primary_mean_per_model = df_primary_across_models.mean(axis=0)  # the same as taking mean and getting across-subject SEM as below (only the error bar differs)
    non_primary_mean_per_model = df_non_primary_across_models.mean(axis=0)  # the same as taking mean and getting across-subject SEM as below (only the error bar differs)

    # Plot
    color_order = [d_model_colors[x] for x in source_models]
    
    # Figure out limits
    if not layer_value_of_interest.startswith('dim_'): # i.e. using some layer position metric, not dim
        if randnetw == 'True' or save_str == '_all-models':
            ylim = [-0.02, 1.02]
        else:
            ylim = [0.1, 0.9]
    if layer_value_of_interest.startswith('dim_'):
        ylim = [0, 130]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(primary_mean_per_model, non_primary_mean_per_model,
                xerr=primary_yerr, yerr=non_primary_yerr, fmt='none',
                ecolor=color_order, alpha=alpha)
    ax.scatter(primary_mean_per_model, non_primary_mean_per_model, color=color_order, s=55, alpha=alpha)
    add_identity(ax, color='grey', ls='--', alpha=0.4)
    ax.set_xlim(ylim)
    ax.set_ylim(ylim)
    if not layer_value_of_interest.startswith('dim') and randnetw == 'False' and save_str != '_all-models':
        ax.set_yticks(np.arange(ylim[0] * 10, (ylim[1] + 0.2) * 10, 2) / 10)
        ax.set_xticks(np.arange(ylim[0] * 10, (ylim[1] + 0.2) * 10, 2) / 10)
    ax.tick_params(axis='x', labelsize=17)
    ax.tick_params(axis='y', labelsize=17)
    ax.set_xlabel(f'Relative best layer: {primary_rois}', size=17)
    ax.set_ylabel(f'Relative best layer {non_primary_rois}', size=17)
    ax.set_title(f'{target} {d_randnetw[randnetw][1:]} collapse: {collapse_over_val_layer}, {layer_value_of_interest}', size='medium')
    ax.set_aspect('equal')
    if annotate:
        for i, txt in enumerate(source_models):
            ax.annotate(txt, (primary_mean_per_model[i] + 0.005, non_primary_mean_per_model[i] + 0.005), size='x-small')
    classes_polished_name = [d_model_names[x] for x in source_models]
    class_colours = color_order
    recs = []
    for i in range(0, len(class_colours)):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
    fig.legend(recs, classes_polished_name, bbox_to_anchor=(1.0, 1.))
    plt.tight_layout()
    if save:
        save_str = f'across-models_scatter{d_annotate[annotate]}{save_str}_' \
                   f'{yerr_type}_{condition_col}_' \
                   f'{primary_rois}-{non_primary_rois}' \
                   f'_{target}{d_randnetw[randnetw]}_' \
                   f'{layer_value_of_interest}-{collapse_over_val_layer}_' \
                   f'{value_of_interest}_' \
                   f'layer-exclude-{layers_exclusion_savestr}'
        # save csv
        df_save = pd.DataFrame(
            data={'primary_mean': primary_mean_per_model, 'non_primary_mean': non_primary_mean_per_model,
                  'primary_yerr': primary_yerr, 'non_primary_yerr': non_primary_yerr})
        df_save['primary_roi'] = primary_rois * len(df_save)
        df_save['non_primary_roi'] = non_primary_rois * len(df_save)
        df_save.to_csv(join(save, f'{save_str}.csv'))
        
        fig.savefig(join(save, f'{save_str}.png'), dpi=180)
        fig.savefig(join(save, f'{save_str}.svg'), dpi=180)
    fig.show()
    
    if stats:
        w, p = wilcoxon(primary_mean_per_model.values, non_primary_mean_per_model.values)
        print(f'Wilcoxon signed-rank test between {primary_rois} and {non_primary_rois}: {w}, p-value: {p:.5f} for {target} {d_randnetw[randnetw]}')
        if save:
            # Package w, p into a dataframe and save
            df_save = pd.DataFrame(data={'w': w, 'p': p}, index=[0])
            df_save['primary_roi'] = primary_rois * len(df_save)
            df_save['non_primary_roi'] = non_primary_rois * len(df_save)
            df_save['target'] = target * len(df_save)
            df_save['randnetw'] = randnetw * len(df_save)
            df_save['test'] = 'wilcoxon' * len(df_save)
            df_save['yerr_type'] = yerr_type * len(df_save)
            df_save['condition_col'] = condition_col * len(df_save)
            df_save['save_str'] = save_str * len(df_save)
            df_save['collapse_over_val_layer'] = collapse_over_val_layer * len(df_save)
            df_save['datetag'] = datetag * len(df_save)
            df_save.to_csv(join(RESULTDIR_ROOT,'STATS_across-models', f'{save_str}.csv'))


def load_scatter_anat_roi_best_layer(target,
                                    randnetw='False',
                                    annotate=False,
                                    condition_col='roi_label_general',
                                    collapse_over_val_layer='median',
                                    primary_rois=['Primary'],
                                    non_primary_rois=['Anterior'],
                                    yerr_type='within_subject_sem',
                                    save_str='',
                                    value_of_interest='median_r2_test_c',
                                    layer_value_of_interest='rel_pos',
                                    layers_to_exclude='',
                                     RESULTDIR_ROOT='/om2/user/gretatu/results/AUD/20210915_median_across_splits_correction/PLOTS_ACROSS_MODELS/',
                                     ):
    """
    Load the best layer values that were used for the anatomical ROI scatter plots (generated using scatter_anat_roi_across_models())
    
    Params used in the main paper figure are condition_col = "roi_label_general", collapse_over_val_layer = "median",
        yerr_type = "within_subject_sem", save_str = "", value_of_interest = "median_r2_test_c", layer_value_of_interest = "rel_pos"
        
        layers_to_exclude was None none in trained, but 'input_after_preproc' for permuted.

    """
    # We didn't exclude any layers, so the file is suffixed with layer-exclude-.csv
    layers_exclusion_savestr = f'_layer-exclude-{"-".join(layers_to_exclude)}' # suffix with this

    load_str = f'across-models_scatter{d_annotate[annotate]}{save_str}_' \
               f'{yerr_type}_{condition_col}_' \
               f'{primary_rois}-{non_primary_rois}' \
               f'_{target}{d_randnetw[randnetw]}_' \
               f'{layer_value_of_interest}-{collapse_over_val_layer}_' \
               f'{value_of_interest}' \
               f'{layers_exclusion_savestr}'

    df = pd.read_csv(join(RESULTDIR_ROOT,
                          f'{load_str}.csv'))

    return df
    


def scatter_components_across_models(source_models,
                                     target,
                                     randnetw='False',
                                     value_of_interest='median_r2_test',
                                     sem_of_interest='median_r2_test_sem_over_it',
                                     save=None,
                                     include_spectemp=True,
                                     save_str='',ylim=[0,1],
                                     symbols=True,
                                     aggregation='CV-splits-nit-10'):
    """Load the best layer per compponent csv files and make a plot across all models
    Also plots scatter plot, one comp versus the other comp.
    
    :param source_models:
    :param target:
    :param value_of_interest:
    :param sem_of_interest: If 'sem_r2_test', uses SEM over CV splits.
    :param ylim: for 2D scatter plots.
    """

    # Obtain best layer r2 test scores for the components!
    df_lst = []
    for source_model in source_models:
        df = pd.read_csv(
            join(RESULTDIR_ROOT, source_model, 'outputs',
                 f'best-layer-{aggregation}_per-comp_{source_model}{d_randnetw[randnetw]}_{target}_{value_of_interest}.csv')).\
            rename(columns={'Unnamed: 0': 'comp'})
        df_lst.append(df)

    df_all = pd.concat(df_lst)
    
    # Obtain spectemp val
    if include_spectemp:
        if aggregation.startswith('CV-splits-nit'):  # also obtain spectemp value based on random splits
            df_spectemp = pd.read_csv(join(RESULTDIR_ROOT, 'spectemp', 'outputs',
                                             f'best-layer-CV-splits-nit-{int(aggregation.split("-")[-1])}'
                                             f'_per-comp_spectemp_{target}_{value_of_interest}.csv'))
        else:
            df_spectemp = obtain_NH2015comp_spectemp_val(target=target, value_of_interest=value_of_interest,)
        
    # Get lists for every component
    lowfreq_r2 = df_all.query('comp == "lowfreq"')[value_of_interest].values
    lowfreq_sem = df_all.query('comp == "lowfreq"')[sem_of_interest].values
    
    highfreq_r2 = df_all.query('comp == "highfreq"')[value_of_interest].values
    highfreq_sem = df_all.query('comp == "highfreq"')[sem_of_interest].values
    
    envsounds_r2 = df_all.query('comp == "envsounds"')[value_of_interest].values
    envsounds_sem = df_all.query('comp == "envsounds"')[sem_of_interest].values
    
    pitch_r2 = df_all.query('comp == "pitch"')[value_of_interest].values
    pitch_sem = df_all.query('comp == "pitch"')[sem_of_interest].values
    
    speech_r2 = df_all.query('comp == "speech"')[value_of_interest].values
    speech_sem = df_all.query('comp == "speech"')[sem_of_interest].values
    
    music_r2 = df_all.query('comp == "music"')[value_of_interest].values
    music_sem = df_all.query('comp == "music"')[sem_of_interest].values
    
    color_order = [d_model_colors[x] for x in source_models]
    if randnetw == 'False':
        alpha_dot = 1
    else:
        alpha_dot = 0.8
    
    bar_placement = np.arange(0, 6 / 2, 0.5)
    
    # Plot across all models and components
    plt.figure(figsize=(9, 5))
    plt.scatter(np.repeat(bar_placement, len(source_models)),
                [lowfreq_r2, highfreq_r2, envsounds_r2, pitch_r2, speech_r2, music_r2], c=color_order * 6, s=50)
    # plot spectemp separately
    if include_spectemp:
        plt.scatter(bar_placement, [df_spectemp[value_of_interest].values], c='grey', s=50)
        plt.plot(bar_placement, df_spectemp[value_of_interest].values, alpha=alpha_dot, c='grey', ls='--')
    for i in range(len(source_models)):
        plt.plot(bar_placement,
                 [lowfreq_r2[i], highfreq_r2[i], envsounds_r2[i], pitch_r2[i], speech_r2[i], music_r2[i]],
                 c=color_order[i], alpha=0.3)
    plt.ylim([0, 1])
    plt.ylabel(f'Median $R^2$')
    plt.xticks(bar_placement, ['lowfreq', 'highfreq', 'envsounds', 'pitch', 'speech', 'music'])
    plt.title(f'Predictivity of components across all models {d_randnetw[randnetw]}')
    classes = source_models
    classes_polished_name = [d_model_names[x] for x in classes]
    class_colours = color_order
    recs = []
    for i in range(0, len(class_colours)):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
    plt.legend(recs, classes_polished_name, bbox_to_anchor=(1.05, 1.1))
    plt.tight_layout()
    if save:
        save_str = f'across-models_scatter_{aggregation}_{target}{d_randnetw[randnetw]}{save_str}_{value_of_interest}'
        plt.savefig(join(save, f'{save_str}.svg'), dpi=180)
        plt.savefig(join(save, f'{save_str}.png'), dpi=180)
    plt.show()
    
    ## 2D scatter plot ##
    for comp1 in ['music']: # ['lowfreq', 'highfreq', 'envsounds', 'pitch', 'speech', 'music']
        for comp2 in ['pitch', 'speech']:
            if comp1 == comp2:
                continue
            
            # Get comp idx (for legend)
            comp1_idx = np.unique(df_all.query(f'comp == "{comp1}"')['comp_idx'] + 1)[0]
            comp2_idx = np.unique(df_all.query(f'comp == "{comp2}"')['comp_idx'] + 1)[0]
                
            # Get lists for every component
            val1 = df_all.query(f'comp == "{comp1}"')[value_of_interest].values
            val2 = df_all.query(f'comp == "{comp2}"')[value_of_interest].values
            
            # err1 = df_all.query(f'comp == "{comp1}"')[sem_of_interest].values
            # err2 = df_all.query(f'comp == "{comp2}"')[sem_of_interest].values
            err1 = np.zeros(len(val1)) # If not plotting err bars
            err2 = np.zeros(len(val2))
            
            # Obtain values for ST
            if include_spectemp:
                val1_spectemp = df_spectemp.query(f'comp == "{comp1}"')[value_of_interest].values
                val2_spectemp = df_spectemp.query(f'comp == "{comp2}"')[value_of_interest].values
                # err1_spectemp = df_spectemp.query(f'comp == "{comp1}"')[sem_of_interest].values
                # err2_spectemp = df_spectemp.query(f'comp == "{comp2}"')[sem_of_interest].values
                err1_spectemp = np.zeros(len(val1_spectemp)) # If not plotting err bars
                err2_spectemp = np.zeros(len(val2_spectemp))
            else:
                val1_spectemp, val2_spectemp, err1_spectemp, err2_spectemp = None, None, None, None
                
            # Plot
            scatter_2_components(source_models=source_models, target=target, randnetw=randnetw,
                                 lst_val_1=val1, lst_val_2=val2,
                                 lst_err_1=err1, lst_err_2=err2,
                                 name_1=f'Component {comp1_idx}: "{d_comp_names[comp1]}-selective"',
                                 name_2=f'Component {comp2_idx}: "{d_comp_names[comp2]}-selective"',
                                 name_val='Median $R^2$',
                                 spectemp=[val1_spectemp, err1_spectemp, val2_spectemp, err2_spectemp],
                                 save=save, save_str=save_str, ylim=ylim, symbols=symbols)


def scatter_comp_best_layer_across_models(source_models,
                                            target,
                                            randnetw='False',
                                            load_value_of_interest='median_r2_test',
                                            value_of_interest='rel_pos',
                                            save=None,
                                            save_str='',
                                            ylim=[0, 1],
                                            symbols=True,
                                            aggregation='argmax'):
    """Load the best layer per compponent csv files and make a plot across all models (argmax)
    Also plots scatter plot, one comp versus the other comp. Relative layer position values.

    :param source_models:
    :param target:
    :param value_of_interest:
    :param sem_of_interest: If 'sem_r2_test', uses SEM over CV splits.
    :param ylim: for 2D scatter plots.
    """
    
    # Obtain best layer r2 test scores for the components!
    df_lst = []
    for source_model in source_models:
        df = pd.read_csv(
            join(RESULTDIR_ROOT, source_model, 'outputs',
                 f'best-layer-{aggregation}_per-comp_{source_model}{d_randnetw[randnetw]}_{target}_{load_value_of_interest}.csv')). \
            rename(columns={'Unnamed: 0': 'comp'})
        df_lst.append(df)
    
    df_all = pd.concat(df_lst)
    
    # Get lists for every component
    lowfreq_val = df_all.query('comp == "lowfreq"')[value_of_interest].values
    highfreq_val = df_all.query('comp == "highfreq"')[value_of_interest].values
    envsounds_val = df_all.query('comp == "envsounds"')[value_of_interest].values
    pitch_val = df_all.query('comp == "pitch"')[value_of_interest].values
    speech_val = df_all.query('comp == "speech"')[value_of_interest].values
    music_val = df_all.query('comp == "music"')[value_of_interest].values
    
    color_order = [d_model_colors[x] for x in source_models]
    if randnetw == 'False':
        alpha_dot = 1
    else:
        alpha_dot = 0.8
    
    bar_placement = np.arange(0, 6 / 2, 0.5)
    
    # Plot across all models and components
    plt.figure(figsize=(9, 5))
    plt.scatter(np.repeat(bar_placement, len(source_models)),
                [lowfreq_val, highfreq_val, envsounds_val, pitch_val, speech_val, music_val], c=color_order * 6, s=50)
    for i in range(len(source_models)):
        plt.plot(bar_placement,
                 [lowfreq_val[i], highfreq_val[i], envsounds_val[i], pitch_val[i], speech_val[i], music_val[i]],
                 c=color_order[i], alpha=0.3)
    plt.ylim(ylim)
    plt.ylabel(f'Relative best layer')
    plt.xticks(bar_placement, ['lowfreq', 'highfreq', 'envsounds', 'pitch', 'speech', 'music'])
    plt.title(f'Relative best layer of components across models {d_randnetw[randnetw][1:]}')
    classes = source_models
    classes_polished_name = [d_model_names[x] for x in classes]
    class_colours = color_order
    recs = []
    for i in range(0, len(class_colours)):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
    plt.legend(recs, classes_polished_name, bbox_to_anchor=(1.05, 1.1))
    plt.tight_layout()
    if save:
        save_str = f'across-models_scatter_{aggregation}_{target}{d_randnetw[randnetw]}{save_str}_{value_of_interest}'
        plt.savefig(join(save, f'{save_str}.svg'), dpi=180)
        plt.savefig(join(save, f'{save_str}.png'), dpi=180)
    plt.show()
    
    ## 2D scatter plot ##
    for comp1 in ['lowfreq', 'highfreq', 'envsounds', 'pitch', 'speech', 'music']:
        for comp2 in ['speech', 'music']:
            if comp1 == comp2:
                continue
            
            # Get comp idx (for legend)
            comp1_idx = np.unique(df_all.query(f'comp == "{comp1}"').index + 1)[0]
            comp2_idx = np.unique(df_all.query(f'comp == "{comp2}"').index + 1)[0]
            
            # Get lists for every component
            val1 = df_all.query(f'comp == "{comp1}"')[value_of_interest].values
            val2 = df_all.query(f'comp == "{comp2}"')[value_of_interest].values

            val1_spectemp, val2_spectemp, err1_spectemp, err2_spectemp = None, None, None, None
            
            # Plot
            scatter_2_components(source_models=source_models, target=target, randnetw=randnetw,
                                 lst_val_1=val1, lst_val_2=val2,
                                 lst_err_1=None, lst_err_2=None,
                                 name_1=f'Component {comp1_idx}: "{d_comp_names[comp1]}-selective"',
                                 name_2=f'Component {comp2_idx}: "{d_comp_names[comp2]}-selective"',
                                 name_val='Relative best layer',
                                 spectemp=[val1_spectemp, err1_spectemp, val2_spectemp, err2_spectemp],
                                 save=save, save_str=save_str, ylim=ylim, symbols=symbols)


def scatter_2_components(source_models, target,
                         lst_val_1, lst_val_2, lst_err_1, lst_err_2,
                         randnetw='False',
                         name_1='', name_2='', name_val='', ylim=None,
                         annotate=False, spectemp=None, save=None, save_str='', symbols=True,
                         black=True):

    """Plot 2 components against each other. Individual points are models
    
    spectemp is a list of the component values of interest, formatted as
    [val1_spectemp, err1_spectemp, val2_spectemp, err2_spectemp]
    
    """
    if black:
        color_order = ['black'] * len(source_models)
    else:
        color_order = [d_model_colors[x] for x in source_models]
    
    marker_order = [d_model_markers[x] for x in source_models]
   
    if randnetw == 'False':
        alpha_dot = 1
    else:
        alpha_dot = 0.8
    s=75

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_box_aspect(1)
    ax.errorbar(lst_val_1, lst_val_2, xerr=lst_err_1, yerr=lst_err_2, fmt='none',
                ecolor=color_order, zorder=0, alpha=alpha_dot)
    if symbols:
        for x, y, c, m, sm in zip(lst_val_1, lst_val_2, color_order, marker_order, source_models):
            if sm.endswith('multitask') or sm.endswith('multitaskSeed2'):  # make open circle
                ax.scatter([x], [y], color=c, s=s, alpha=alpha_dot, marker=m, facecolors='none', label=sm)
            else:
                ax.scatter([x], [y], color=c, s=s, alpha=alpha_dot, marker=m, label=sm)
        fig.legend(bbox_to_anchor=(1, 0.9))
    else:
        ax.scatter(lst_val_1, lst_val_2, color=color_order, s=s, alpha=alpha_dot)
    if spectemp is not None:
        ax.errorbar(spectemp[0], spectemp[2], xerr=spectemp[1], yerr=spectemp[3], fmt='none',
                    ecolor='grey')
        ax.scatter(spectemp[0], spectemp[2], c='grey', s=s)
    add_identity(ax, color='grey', ls='--', alpha=0.4)
    if ylim is not None:
        ax.set_ylim(ylim)
        ax.set_xlim(ylim)
    else:
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlabel(f'{name_val} {name_1}', size='medium')  # Median $R^2$
    ax.set_ylabel(f'{name_val} {name_2}', size='medium')  # Median $R^2$
    ax.set_title(f'{name_val} {d_randnetw[randnetw][1:]}', size='small')
    if annotate:
        for i, txt in enumerate(source_models):
            ax.annotate(txt, (lst_val_1[i], lst_val_2[i]), size='x-small')
    fig.tight_layout()
    if save:
        name_1 = name_1.split(':')[0].replace(' ','-')
        name_2 = name_2.split(':')[0].replace(' ','-')
        plt.savefig(join(save, f'{name_1}-{name_2}_scatter_{target}{d_randnetw[randnetw]}{save_str}_symbols-{symbols}_black-{black}.svg'), dpi=180)
        plt.savefig(join(save, f'{name_1}-{name_2}_scatter_{target}{d_randnetw[randnetw]}{save_str}_symbols-{symbols}_black-{black}.png'), dpi=180)
    fig.show()


def barplot_components_across_models(source_models,
                                     target,
                                     randnetw='False',
                                     value_of_interest='median_r2_test',
                                     yerr_type='median_r2_test_sem_over_it',
                                     sort_by='performance',
                                     aggregation='CV-splits-nit-10',
                                     save=None,
                                     components=['lowfreq', 'highfreq', 'envsounds', 'pitch', 'speech','music'],
                                     include_spectemp=True,
                                     add_savestr='',
                                     alpha=1,
                                     ylim=[0,1],
                                     add_in_spacing_bar=True,
                                     box_aspect=0.8):
    """Load the best layer per component csv files and make a plot across all models, one for each component.

    :param source_models:
    :param target:
    :param value_of_interest:
    :param yerr_type: If ''median_r2_test_sem_over_it', uses SEM over the layer selection procedure.
    :param ylim: for 2D scatter plots.
    """
    
    # Obtain best layer r2 test scores for the components!
    df_lst = []
    for source_model in source_models:
        df = pd.read_csv(
            join(RESULTDIR_ROOT, source_model, 'outputs',
                 f'best-layer-{aggregation}_'
                 f'per-comp_{source_model}{d_randnetw[randnetw]}_'
                 f'{target}_{value_of_interest}.csv')).rename(
            columns={'Unnamed: 0': 'comp'})
        df_lst.append(df)
    df_all = pd.concat(df_lst)
    
    # Obtain spectemp val
    if include_spectemp:
        if aggregation.startswith('CV-splits-nit'):  # also obtain spectemp value based on random splits
            df_spectemp = pd.read_csv(join(RESULTDIR_ROOT, 'spectemp', 'outputs',
                                             f'best-layer-CV-splits-nit-{int(aggregation.split("-")[-1])}'
                                             f'_per-comp_spectemp_{target}_{value_of_interest}.csv'))
        else:
            df_spectemp = obtain_NH2015comp_spectemp_val(target=target, value_of_interest=value_of_interest,)


    # plot specs
    if add_in_spacing_bar: # create an empty bar before multitask
        bar_placement = np.arange(0, (len(df_all.source_model.unique())+1) / 2, 0.5)
        # Group bars closer together two and two. Add +.1 to odd numbers to make them closer together but retain even number positions
        bar_placement[1:8:2] -= 0.1
        bar_placement[-1] -= 0.1
        # Obtain xmin and xmax for spectemp line
        xmin = bar_placement[0] - 0.2
        xmax = bar_placement[-1] + 0.2
    else:
        bar_placement = np.arange(0, len(df_all.source_model.unique()) / 2, 0.5)
        # Obtain xmin and xmax for spectemp line
        xmin = np.unique((bar_placement[0] - np.diff(bar_placement) / 2))
        xmax = np.unique((bar_placement[-1] + np.diff(bar_placement) / 2))

    fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(28, 8))
    for i, comp in enumerate(components):
        # Get lists for every component
        df_comp = df_all[df_all.comp == comp]
        
        # Sort
        if sort_by == 'performance':
            sort_str = '_performance_sorted'
            df_comp = df_comp.sort_values(by=value_of_interest, ascending=False)
        if type(sort_by) == list:
            sort_str = '_manually_sorted'
            df_comp = df_comp.set_index('source_model', inplace=False, drop=False)
            df_comp = df_comp.reindex(sort_by)
            if add_in_spacing_bar: # add in empty additional row after ResNetaudioset
                line = pd.DataFrame({f"{value_of_interest}": 0, "sem_of_interest": 0,
                                     'source_model': 'mock'}, index=['mock_row'])
                df_comp = pd.concat([df_comp.iloc[:-2], line, df_comp.iloc[-2:]]).reset_index(drop=True)
        if include_spectemp:
            df_spectemp_comp = df_spectemp[df_spectemp.comp == comp]
        color_order = [d_model_colors[x] for x in df_comp.source_model]
        r2 = df_comp[value_of_interest].values
        sem = df_comp[yerr_type].values
        ax[i].set_box_aspect(box_aspect)
        if include_spectemp:
            ax[i].hlines(xmin=xmin, xmax=xmax, y=df_spectemp_comp[f'{value_of_interest}'].values, color='darkgrey',zorder=2)
            ax[i].fill_between(
                [(bar_placement[0] - np.diff(bar_placement) / 2)[0], (bar_placement[-1] + np.diff(bar_placement) / 2)[0]],
                df_spectemp_comp[f'{value_of_interest}'].values - df_spectemp_comp[f'{yerr_type}'].values,
                df_spectemp_comp[f'{value_of_interest}'].values + df_spectemp_comp[f'{yerr_type}'].values,
                color='gainsboro')
        ax[i].bar(bar_placement,
                  r2,
                  yerr=sem,
                  width=0.3, alpha=alpha, color=color_order, zorder=3)
        ax[i].set_title(f'Component number {i + 1}, "{comp}"', size='medium')
        ax[i].set_ylim(ylim)
        ax[i].set_xticks(bar_placement)
        ax[i].set_xticklabels([d_model_names[x] for x in df_comp.source_model], rotation=80,
                              fontsize=13,
                              ha='right', rotation_mode='anchor')
        ax[i].set_ylabel(d_value_of_interest[value_of_interest], fontsize=15)
        # Make yticks larger
        ax[i].tick_params(axis='y', labelsize=15)
    plt.suptitle(f'{d_value_of_interest[value_of_interest]} across models {sort_str[1:]}\n{target} {d_randnetw[randnetw][1:]}')
    plt.tight_layout(pad=1.8)
    if save:
        save_str = f'across-models_barplot_components_{target}{d_randnetw[randnetw]}_' \
                   f'{aggregation}_{yerr_type}_' \
                   f'{value_of_interest}{sort_str}{add_savestr}'
        plt.savefig(join(save, f'{save_str}.svg'), dpi=180)
        plt.savefig(join(save, f'{save_str}.png'), dpi=180)

        # save csv and log more info
        # Append df_spectemp to df_all
        df_all_w_spectemp = pd.concat([df_all,df_spectemp])
        df_all_w_spectemp['target'] = target
        df_all_w_spectemp['randnetw'] = randnetw
        df_all_w_spectemp['aggregation'] = aggregation
        df_all_w_spectemp['yerr_type'] = yerr_type
        df_all_w_spectemp['value_of_interest'] = value_of_interest
        df_all_w_spectemp['sort_by'] = [sort_by] * len(df_all_w_spectemp)
        df_all_w_spectemp['add_savestr'] = add_savestr
        df_all_w_spectemp['n_models'] = len(df_all_w_spectemp)
        df_all_w_spectemp.to_csv(join(save, f'{save_str}.csv'))

    plt.show()


def barplot_components_across_models_clean(source_models,
                                     target,
                                     randnetw='False',
                                     value_of_interest='median_r2_test',
                                     yerr_type='median_r2_test_sem_over_it',
                                     sort_by='performance',
                                     aggregation='CV-splits-nit-10',
                                     save=None,
                                     components=['lowfreq', 'highfreq', 'envsounds', 'pitch', 'speech', 'music'],
                                     include_spectemp=True,
                                     add_savestr='',
                                     alpha=1,
                                     ylim=[0, 1],
                                     box_aspect=1):
    """Load the best layer per component csv files and make a plot across all models, one for each component.
    Intended for use with clean speech (word/speaker) networks because we want to group together the first two bars and
    the last two bars.

    :param source_models:
    :param target:
    :param value_of_interest:
    :param yerr_type: If ''median_r2_test_sem_over_it', uses SEM over the layer selection procedure.
    :param ylim: for 2D scatter plots.
    """

    # Obtain best layer r2 test scores for the components!
    df_lst = []
    for source_model in source_models:
        df = pd.read_csv(
            join(RESULTDIR_ROOT, source_model, 'outputs',
                 f'best-layer-{aggregation}_'
                 f'per-comp_{source_model}{d_randnetw[randnetw]}_'
                 f'{target}_{value_of_interest}.csv')).rename(
            columns={'Unnamed: 0': 'comp'})
        df_lst.append(df)
    df_all = pd.concat(df_lst)

    # Obtain spectemp val
    if include_spectemp:
        if aggregation.startswith('CV-splits-nit'):  # also obtain spectemp value based on random splits
            df_spectemp = pd.read_csv(join(RESULTDIR_ROOT, 'spectemp', 'outputs',
                                           f'best-layer-CV-splits-nit-{int(aggregation.split("-")[-1])}'
                                           f'_per-comp_spectemp_{target}_{value_of_interest}.csv'))
        else:
            df_spectemp = obtain_NH2015comp_spectemp_val(target=target, value_of_interest=value_of_interest, )

    # Group first two bars and last two bars together
    offset = 0.35
    bar_placement = [0, 0.35, 0.7+offset, 1.05+offset, 1.4+offset*2, 1.75+offset*2, 2.1+offset*3, 2.45+offset*3]

    # Get xmin and xmax which should be +- 0.25 of the min and max of bar_placement
    xmin = np.min(bar_placement) - 0.25
    xmax = np.max(bar_placement) + 0.25

    fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(28, 8))
    for i, comp in enumerate(components):
        # Get lists for every component
        df_comp = df_all[df_all.comp == comp]

        # Sort
        if sort_by == 'performance':
            sort_str = '_performance_sorted'
            df_comp = df_comp.sort_values(by=value_of_interest, ascending=False)
        if type(sort_by) == list:
            sort_str = '_manually_sorted'
            df_comp = df_comp.set_index('source_model', inplace=False, drop=False)
            df_comp = df_comp.reindex(sort_by)
        if include_spectemp:
            df_spectemp_comp = df_spectemp[df_spectemp.comp == comp]
        color_order = [d_model_colors[x] for x in df_comp.source_model]
        r2 = df_comp[value_of_interest].values
        sem = df_comp[yerr_type].values
        ax[i].set_box_aspect(box_aspect)
        if include_spectemp:
            ax[i].hlines(xmin=xmin, xmax=xmax, y=df_spectemp_comp[f'{value_of_interest}'].values, color='darkgrey',
                         zorder=2)
            ax[i].fill_between(
                [(bar_placement[0] - np.diff(bar_placement) / 2)[0],
                 (bar_placement[-1] + np.diff(bar_placement) / 2)[0]],
                df_spectemp_comp[f'{value_of_interest}'].values - df_spectemp_comp[f'{yerr_type}'].values,
                df_spectemp_comp[f'{value_of_interest}'].values + df_spectemp_comp[f'{yerr_type}'].values,
                color='gainsboro')
        ax[i].bar(bar_placement,
                  r2,
                  yerr=sem,
                  width=0.3, alpha=alpha, color=color_order, zorder=3)
        ax[i].set_title(f'Component number {i + 1}, "{comp}"', size='medium')
        ax[i].set_ylim(ylim)
        ax[i].set_xticks(bar_placement)
        ax[i].set_xticklabels([d_model_names[x] for x in df_comp.source_model], rotation=80,
                              fontsize=13,
                              ha='right', rotation_mode='anchor')
        ax[i].set_ylabel(d_value_of_interest[value_of_interest], fontsize=15)
        # Make yticks larger
        ax[i].tick_params(axis='y', labelsize=15)
    plt.suptitle(
        f'{d_value_of_interest[value_of_interest]} across models {sort_str[1:]}\n{target} {d_randnetw[randnetw][1:]}')
    plt.tight_layout(pad=1.8)
    if save:
        save_str = f'across-models_barplot_components_{target}{d_randnetw[randnetw]}_' \
                   f'{aggregation}_{yerr_type}_' \
                   f'{value_of_interest}{sort_str}{add_savestr}'
        plt.savefig(join(save, f'{save_str}.svg'), dpi=180)
        plt.savefig(join(save, f'{save_str}.png'), dpi=180)

        # save csv and log more info
        # Append df_spectemp to df_all
        df_all_w_spectemp = pd.concat([df_all, df_spectemp])
        df_all_w_spectemp['target'] = target
        df_all_w_spectemp['randnetw'] = randnetw
        df_all_w_spectemp['aggregation'] = aggregation
        df_all_w_spectemp['yerr_type'] = yerr_type
        df_all_w_spectemp['value_of_interest'] = value_of_interest
        df_all_w_spectemp['sort_by'] = [sort_by] * len(df_all_w_spectemp)
        df_all_w_spectemp['add_savestr'] = add_savestr
        df_all_w_spectemp['n_models'] = len(df_all_w_spectemp)
        df_all_w_spectemp.to_csv(join(save, f'{save_str}.csv'))

    plt.show()


def scatter_components_across_models_seed(source_models,
                            models1,
                            models2,
                            target,
                            roi=None,
                            value_of_interest='median_r2_test',
                            yerr_type='median_r2_test_sem_over_it',
                            randnetw='False',
                            components=['lowfreq', 'highfreq', 'envsounds', 'pitch', 'speech', 'music'],
                            save=None,
                            aggregation='CV-splits-nit-10',
                            ylim=[0.6,0.8],
                            xlim=[0.6,0.8],
                          add_savestr=''):
    """
    Plot median variance explained across models for all voxels.
    The score is loaded using CV-splits-nit-10 (default) layer selection procedure.

    Intended for checking whether Seed1 vs Seed2 models show similar variance explained.

    Models1 should contain the models to go on the x-axis, models2 on the y-axis.

    :param source_models:
    :param roi:
    :param value_of_interest:
    :return:
    """
    # Obtain best layer r2 test scores for the components!
    df_lst = []
    for source_model in source_models:
        df = pd.read_csv(
            join(RESULTDIR_ROOT, source_model, 'outputs',
                 f'best-layer-{aggregation}_'
                 f'per-comp_{source_model}{d_randnetw[randnetw]}_'
                 f'{target}_{value_of_interest}.csv')).rename(
            columns={'Unnamed: 0': 'comp'})
        df_lst.append(df)
    df_all = pd.concat(df_lst)
    df_grouped = df_all.copy(deep=True) # Make name consistent with corresponding function for neural data

    # Take the models in models1 and models2 and put them on the x and y axis, respectively. Retain the order of models1 and models2.
    assert len(models1) == len(models2), 'models1 and models2 should have the same length'
    assert all([x in np.unique(df_grouped['source_model'].values) for x in models1]), 'models1 should be a subset of source_models'
    assert all([x in df_grouped['source_model'].values for x in models2]), 'models2 should be a subset of source_models'

    title_str = f'{d_value_of_interest[value_of_interest]} across models\nAll voxels, {target} {d_randnetw[randnetw][1:]} ({aggregation})'

    fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(28, 8))
    # Plot for each component
    for i, comp in enumerate(components):

        df_comp = df_grouped[df_grouped['comp'] == comp]

        df_models1 = df_comp[df_comp['source_model'].isin(models1)]
        df_models2 = df_comp[df_comp['source_model'].isin(models2)]
        # Remove "Seed2" from source_model in df_models2 and assert
        if 'Seed2' in df_models2.source_model.values[0]:
            source_models2_no_seed2 = [x.replace('Seed2', '') for x in df_models2.source_model.values]
            assert (df_models1.source_model.values == source_models2_no_seed2).all(), 'models1 and models2 should have the same order'

        # plot specs
        color_order_models1 = [d_model_colors[x] for x in df_models1.source_model.values]
        color_order_models2 = [d_model_colors[x] for x in df_models2.source_model.values]
        assert color_order_models1 == color_order_models2, 'color order should be the same for models1 and models2'

        # Plot scatter
        ax[i].set_box_aspect(1)

        # Add component name to plot
        ax[i].set_title(comp)

        # Plot scatter with error bars. Iterate over each point because otherwise we can't have different colors for each point
        for model_i in range(len(df_models1)):
            # Transform color to RGBA
            color = mcolors.to_rgba(color_order_models1[model_i])
            ax[i].errorbar(df_models1[f'{value_of_interest}'].values[model_i],
                        df_models2[f'{value_of_interest}'].values[model_i],
                        xerr=df_models1[f'{yerr_type}'].values[model_i],
                        yerr=df_models2[f'{yerr_type}'].values[model_i],
                        fmt='o', markersize=10, color=color,
                        capsize=0, elinewidth=2, markeredgewidth=2)
            add_identity(ax[i], color='black', ls='--')
            if ylim is not None:
                ax[i].set_ylim(ylim)
            if xlim is not None:
                ax[i].set_xlim(xlim)
            plt.xlabel(f'{d_value_of_interest[value_of_interest]} Seed 1', fontsize=15)
            plt.ylabel(f'{d_value_of_interest[value_of_interest]} Seed 2', fontsize=15)

            # Make ticks bigger
            ax[i].tick_params(axis='both', which='major', labelsize=15)

        # Also plot correlation r between models1 and models2
        r, p = stats.pearsonr(df_models1[f'{value_of_interest}'].values,
                              df_models2[f'{value_of_interest}'].values)
        r2 = r ** 2
        # Plot in lower right
        ax[i].text(0.95, 0.05, f'$R^2$={r2:.2f}, p={p:.2f}', horizontalalignment='right', verticalalignment='bottom',
                   fontsize=15, transform=ax[i].transAxes)

        # Store these r, r2 and p values in df_grouped
        df_grouped.loc[df_grouped['comp'] == comp, 'r'] = r
        df_grouped.loc[df_grouped['comp'] == comp, 'r2'] = r2
        df_grouped.loc[df_grouped['comp'] == comp, 'p'] = p

    # Add legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=x, markerfacecolor=mcolors.to_rgba(d_model_colors[x]),
                              markersize=10) for x in df_models1.source_model.values]
    # Plot legend outside of plot
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.9), fontsize=15)
    plt.suptitle(title_str, fontsize=20)
    plt.tight_layout(pad=2.5)

    if save:
        save_str = f'across-models_roi-{roi}_{target}{d_randnetw[randnetw]}_' \
                   f'{aggregation}_{yerr_type}_' \
                   f'n-models={len(df_grouped)}' \
                   f'{value_of_interest}{add_savestr}'
        plt.savefig(join(save, f'{save_str}.png'), dpi=180)
        plt.savefig(join(save, f'{save_str}.svg'), dpi=180)

        # save csv and log more info
        df_grouped['roi'] = roi
        df_grouped['target'] = target
        df_grouped['randnetw'] = randnetw
        df_grouped['aggregation'] = aggregation
        df_grouped['yerr_type'] = yerr_type
        df_grouped['value_of_interest'] = value_of_interest
        df_grouped['add_savestr'] = add_savestr
        df_grouped['n_models'] = len(df_grouped)
        df_grouped['models1'] = [models1] * len(df_grouped)
        df_grouped['models2'] = [models2] * len(df_grouped)
        df_grouped.to_csv(join(save, f'{save_str}.csv'))
    fig.show()



def find_best_layer_per_component(source_models, target, value_of_interest='median_r2_test',include_spectemp=True):
    """Load the best layer per compponent csv files and find the very best layer (across models) for each component.
    """
    
    # Obtain best layer r2 test scores for the components!
    df_lst = []
    for source_model in source_models:
        df = pd.read_csv(
            join(RESULTDIR_ROOT, source_model, 'outputs',
                 f'best-layer-per-comp_{source_model}_{target}_{value_of_interest}.csv')).rename(columns={'Unnamed: 0': 'component'})
        df_lst.append(df)
    
    df_all = pd.concat(df_lst)
    
    # Obtain spectemp val
    if include_spectemp:
        df_spectemp = obtain_NH2015comp_spectemp_val(target=target, value_of_interest='median_r2_test',
                                                     yerr='std_r2_test')
        df_all = df_all.append(df_spectemp)

    df_best_layer_per_comp = df_all.sort_values(value_of_interest, ascending=False).drop_duplicates(['component'])
    df_best_layer_per_comp.to_csv(join(SAVEDIR_CENTRALIZED, f'best-layer-per-comp_{target}_{value_of_interest}.csv'), index=False)
    



#### ACROSS MODELS AND ACROSS TARGETS ####
def modelwise_scores_across_targets(source_models,
                                    target1,
                                    target2,
                                    roi=None,
                                    aggregation='CV-splits-nit-10',
                                    value_of_interest='median_r2_test_c',
                                    randnetw='False',
                                    target1_loadstr_suffix='_performance_sorted',
                                    target2_loadstr_suffix='_performance_sorted',
                                    RESULTDIR_ROOT='/Users/gt/Documents/GitHub/auditory_brain_dnn/results/',
                                    save=False,
                                    add_savestr='',
                                    ylim=(0.2,0.8)):
    """
    For two targets, plot their model-wise scores against each other as a scatter plot.
    
    :param source_models: list of source models
    :param target1: str, identifier of target
    :param target2: str, identifier of target
    :param roi: str or None, region of interest
    :param aggregation: str, aggregation method (how the best layer was selected, e.g. LOSO or CV-splits-nit-10)
    :param value_of_interest: str, name of the value of interest (e.g. median_r2_test_c)
    :param target1_loadstr_suffix: str, suffix of the loadstr for target1 (if the compiled across-models csv/plots were
        saved using an additional save_str)
    :param target2_loadstr_suffix: str, suffix of the loadstr for target2 (if the compiled across-models csv/plots were
        saved using an additional save_str)
    :param RESULTDIR_ROOT: str, result directory.
        We want to load from RESULTDIR_ROOT/PLOTS_across-models/ where the across-models csv/plots are saved
        (which holds the compiled across-models csv/plots)
    :return:
    """
    print('OBS, if using these plots (which we are currently NOT), check errorbars -- these error bars are computed'
          ' based on n=20 models which we might not use here')
    
    df_target1 = pd.read_csv(join(RESULTDIR_ROOT,
                                  'PLOTS_across-models',
                              f'across-models_roi-{roi}_'
                              f'{target1}{d_randnetw[randnetw]}_'
                              f'{aggregation}_within_subject_sem_'
                              f'{value_of_interest}{target1_loadstr_suffix}.csv'))
    df_target1 = df_target1.drop(columns=['source_model.1'])
    df_target1['target'] = target1
    df_target1['loadstr_suffix'] = target1_loadstr_suffix
    df_target1 = df_target1.query('source_model in @source_models')
    
    df_target2 = pd.read_csv(join(RESULTDIR_ROOT,
                                  'PLOTS_across-models',
                               f'across-models_roi-{roi}_'
                               f'{target2}{d_randnetw[randnetw]}_'
                               f'{aggregation}_within_subject_sem_'
                               f'{value_of_interest}{target2_loadstr_suffix}.csv'))
    df_target2 = df_target2.drop(columns=['source_model.1'])
    df_target2['target'] = target2
    df_target2['loadstr_suffix'] = target2_loadstr_suffix
    df_target2 = df_target2.query('source_model in @source_models')
    
    
    assert (df_target1.source_model == df_target2.source_model).all()
    
    color_order = [d_model_colors[model] for model in df_target1.source_model.values]
    
    # Plot scatter with colors according to models
    r, p = stats.pearsonr(df_target1[f'{value_of_interest}_mean'], df_target2[f'{value_of_interest}_mean']) # mean here refers to the fact
    # that the subject-wise scores were meaned in barplot_across_models()
    r2 = r**2
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(x=df_target1[f'{value_of_interest}_mean'],
                y=df_target2[f'{value_of_interest}_mean'],
                xerr=df_target1[f'{value_of_interest}_yerr'],
                yerr=df_target2[f'{value_of_interest}_yerr'],
                color=color_order, ecolor=color_order,
                linestyle='None', fmt='none',
                linewidth=2)
    ax.scatter(x=df_target1[f'{value_of_interest}_mean'],
               y=df_target2[f'{value_of_interest}_mean'],
                color=color_order,
                s=60, zorder=4)
    if ylim is not None:
        ax.set_xlim(ylim)
        ax.set_ylim(ylim)
        # ax.set_yticks(np.arange(ylim[0] * 10, (ylim[1] + 0.2) * 10, 2) / 10)
        # ax.set_xticks(np.arange(ylim[0] * 10, (ylim[1] + 0.2) * 10, 2) / 10)
    add_identity(ax, color='grey', ls='--', alpha=0.4)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlabel(f'{target1} {value_of_interest}', size=11)
    ax.set_ylabel(f'{target2} {value_of_interest}', size=11)
    ax.set_title(f'{target2} vs. {target1} {d_randnetw[randnetw][1:]} '
                 f'{value_of_interest}.\nPearson r2={r2:.3f}, p={p:.3e}',
                 size='medium')
    ax.set_aspect('equal')
    classes_polished_name = [d_model_names[x] for x in df_target1.source_model.values]
    class_colours = color_order
    recs = []
    for i in range(0, len(class_colours)):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
    fig.legend(recs, classes_polished_name, bbox_to_anchor=(1.0, 1.))

    # Merge the two dataframes and save it
    df_merged = pd.concat([df_target1, df_target2])
    df_merged['roi'] = roi
    df_merged['randnetw'] = randnetw
    df_merged['aggregation'] = aggregation
    df_merged['value_of_interest'] = value_of_interest
    df_merged['r'] = r
    df_merged['r2'] = r2
    df_merged['p-val'] = p

    if save:
        save_str = f'across-models_{target1}-vs-{target2}' \
                   f'{d_randnetw[randnetw]}_' \
                   f'scatter{add_savestr}_' \
                   f'roi-{roi}_{value_of_interest}_{aggregation}_ylim-{ylim}'
        fig.savefig(join(save, f'{save_str}.png'), dpi=180)
        fig.savefig(join(save, f'{save_str}.svg'), dpi=180)
        df_merged.to_csv(join(save, f'{save_str}.csv'), index=False)
        
    fig.show()
    
    return df_merged


def permuted_vs_trained_scatter(df_across_models_regr,
                                df_across_models_regr_randnetw,
                                target,
                                val_of_interest='median_r2_test_c_mean',
                                lim=[0, 0.8],
                                save=False):
    """
    Plot modelwise scores for permuted vs. trained models.

    Args
        df_across_models_regr: pd.DataFrame with rows for each model, and a column with the modelwise scores in val_of_interest
        (we exclude the spectemp model because it cannot be permuted)
        df_across_models_regr_randnetw: pd.DataFrame with rows for each model, and a column with the modelwise scores in val_of_interest

    """
    # First, get the trained model scores and drop source_model = spectemp
    df_across_models_regr_trained = df_across_models_regr.query('target == @target and source_model != "spectemp"')
    # Then, get the permuted model scores
    df_across_models_regr_permuted = df_across_models_regr_randnetw.query('target == @target')

    # Assert that they are sorted in the same way
    assert all(df_across_models_regr_trained['source_model'] == df_across_models_regr_permuted['source_model'])

    model_colors = [d_model_colors[source_model] for source_model in df_across_models_regr_trained['source_model']]

    # Plot scatter
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df_across_models_regr_trained[f'{val_of_interest}'],  # mean across participants
               df_across_models_regr_permuted[f'{val_of_interest}'],
               c=model_colors)
    ax.set_xlabel('Trained model', fontsize=14)
    ax.set_ylabel('Permuted model', fontsize=14)
    ax.set_title(f'{target}')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    # Make ticks larger
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    # Only add 5 ticks
    ax.locator_params(nbins=5)
    # Add identity line
    add_identity(ax, color='grey', ls='--', alpha=0.4)
    # Add legend
    classes_polished_name = [d_model_names[x] for x in df_across_models_regr_trained.source_model.values]
    class_colours = [d_model_colors[x] for x in df_across_models_regr_trained.source_model.values]
    recs = []
    for i in range(0, len(class_colours)):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
    # Add legend outside of plot
    fig.legend(recs, classes_polished_name, bbox_to_anchor=(1.33, 1.), fontsize=8)
    # Add R2
    r, p = stats.pearsonr(df_across_models_regr_trained[f'{val_of_interest}'],
                          df_across_models_regr_permuted[f'{val_of_interest}'])
    r2 = r ** 2
    # Add in upper left corner
    ax.text(0.05, 0.88, f'$R^2$={r2:.3f}\np={p:.3f}', transform=ax.transAxes, fontsize=13)
    plt.tight_layout(pad=3)
    if save:
        plt.savefig(join(save, f'permuted-vs-trained-scatter_' \
                               f'{target}_{val_of_interest}.svg'), dpi=180)
    plt.show()


def layerwise_scores_across_targets(source_models,
                                    target1,
                                    target2,
                                    d_across_layers_target1,
                                    d_across_layers_target2,
                                    roi=None,
                                    value_of_interest='median_r2_test_c',
                                    randnetw='False',
                                    yerr_type='within_subject_sem',
                                    save=False,
                                    add_savestr='',
                                    ylim=(0.2, 0.8),
                                    plot_identity=True,):
    """
    For two targets, plot their model-wise scores against each other as a scatter plot.

    :param source_models: list of source models
    :param target1: str, identifier of target
    :param target2: str, identifier of target
    :param d_across_layers_target1: dict, containing the dataframes for the target1.
            Dict of dataframes for each source model. Key = source_model. Value = df with index as the subj idx
		    and columns are value_of_interest for each layer.
		    From function load_score_across_layers_across_models() or load_rsa_scores_across_layers_across_models()
    :param d_across_layers_target2: dict, containing the dataframes for the target2.
            Dict of dataframes for each source model. Key = source_model. Value = df with index as the subj idx
            and columns are value_of_interest for each layer.
    :param roi: str or None, region of interest
    :param value_of_interest: str, name of the value of interest (e.g. median_r2_test_c)
    :return:
    """

    
    # Combine each df_layer (from lst) into one df
    df_model_layer_target1 = pd.concat(d_across_layers_target1, axis=1)
    df_model_layer_target1.columns = df_model_layer_target1.columns.map('_'.join)  # Instead of the column multiindex of model / layer, merge to model_layer
    df_model_layer_target2 = pd.concat(d_across_layers_target2, axis=1)
    df_model_layer_target2.columns = df_model_layer_target2.columns.map('_'.join)  # Instead of the column multiindex of model / layer, merge to model_layer
    
    # See if we have nans
    if df_model_layer_target1.isna().values.any():
        # Find which columns
        nans = df_model_layer_target1.isna().any()
        nans = nans[nans == True]
        
        print(f'Found nans in {nans.index} for target {target1}. DROPPING.')
        
        # Drop these columns in both dfs
        df_model_layer_target1 = df_model_layer_target1.drop(columns=nans.index)
        df_model_layer_target2 = df_model_layer_target2.drop(columns=nans.index)
        
    if df_model_layer_target2.isna().values.any():
        print(f'{target2} has nans')
        # Find which columns
        nans = df_model_layer_target2.isna().any()
        nans = nans[nans == True]
        
        print(f'Found nans in {nans.index} for target {target2}. DROPPING.')
        
        # Drop these columns in both dfs
        df_model_layer_target1 = df_model_layer_target1.drop(columns=nans.index)
        df_model_layer_target2 = df_model_layer_target2.drop(columns=nans.index)
    
    # Find discrepancy in layers, find models not in both
    if df_model_layer_target1.columns.tolist() != df_model_layer_target2.columns.tolist():
        diff = list(set(df_model_layer_target1.columns.tolist()) - set(df_model_layer_target2.columns.tolist()))
        print(f'Found discrepancy in layers between {target1} and {target2}: {diff}')
        assert len(df_model_layer_target1.columns) == len(df_model_layer_target2.columns)

    # get num layers (columns)
    n_layers = len(df_model_layer_target1.columns)
    
    # Now we have subj x layer (unique to each model). Let's compute within-subject SEM
    #  1. Get subject x model_layer pivot (no aggregation going on here): df_layer (piv)
    
    # 2. Obtain error bar (within-subject error bar) by subtracting the mean across cond (model_layer) for each subject
    demeaned_target1 = df_model_layer_target1.subtract(df_model_layer_target1.mean(axis=1).values, axis=0)  # get a mean value for each subject across all conds (piv.mean(axis=1), and subtract it from individual models
    demeaned_target2 = df_model_layer_target2.subtract(df_model_layer_target2.mean(axis=1).values, axis=0)  # get a mean value for each subject across all conds (piv.mean(axis=1), and subtract it from individual models
    
    if yerr_type == 'within_subject_sem':
        yerr_target1 = np.std(demeaned_target1.values.T, axis=1) / np.sqrt(demeaned_target1.shape[0] - 1)
        yerr_target2 = np.std(demeaned_target2.values.T, axis=1) / np.sqrt(demeaned_target2.shape[0] - 1)

    # demeaned_target1.shape[0] is the number of subjects (rows)
    if yerr_type == 'within_subject_std':
        yerr_target1 = np.std(demeaned_target1.values.T, axis=1)
        yerr_target2 = np.std(demeaned_target2.values.T, axis=1)

    # 3. Now obtain the mean across model_layer
    mean_per_model_target1 = df_model_layer_target1.mean(axis=0)
    mean_per_model_target2 = df_model_layer_target2.mean(axis=0)
    
    assert (mean_per_model_target1.index == mean_per_model_target2.index).all()
    source_models_from_df = [model.split('_')[0] for model in mean_per_model_target1.index]
    
    # 4. Zip into one dataframe and sort by performance
    df_grouped = pd.DataFrame({f'{value_of_interest}_mean_{target1}': mean_per_model_target1.values,
                               f'{value_of_interest}_yerr_{target1}': yerr_target1,
                               f'{value_of_interest}_mean_{target2}': mean_per_model_target2.values,
                                f'{value_of_interest}_yerr_{target2}': yerr_target2,
                               'source_model': source_models_from_df,
                                'source_model_layer': mean_per_model_target1.index},
                                index=mean_per_model_target1.index)
    
    # Correlate the two target values for values of interest
    r, p = stats.pearsonr(df_grouped[f'{value_of_interest}_mean_{target1}'],
                          df_grouped[f'{value_of_interest}_mean_{target2}'])
    r2 = r ** 2
    
    color_order = [d_model_colors[model] for model in df_grouped.source_model.values]
    color_order_unique = [d_model_colors[model] for model in df_grouped.source_model.unique()]
    color_order_unique_names = [d_model_names[model] for model in df_grouped.source_model.unique()]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(x=df_grouped[f'{value_of_interest}_mean_{target1}'],
                y=df_grouped[f'{value_of_interest}_mean_{target2}'],
                xerr=df_grouped[f'{value_of_interest}_yerr_{target1}'],
                yerr=df_grouped[f'{value_of_interest}_yerr_{target2}'],
                color=color_order, ecolor=color_order,
                linestyle='None', fmt='none',
                linewidth=2)
    ax.scatter(x=df_grouped[f'{value_of_interest}_mean_{target1}'],
               y=df_grouped[f'{value_of_interest}_mean_{target2}'],
                color=color_order,
                s=40, zorder=4)
    if ylim is not None:
        ax.set_xlim(ylim)
        ax.set_ylim(ylim)
        # ax.set_yticks(np.arange(ylim[0] * 10, (ylim[1] + 0.2) * 10, 2) / 10)
        # ax.set_xticks(np.arange(ylim[0] * 10, (ylim[1] + 0.2) * 10, 2) / 10)
    if plot_identity:
        add_identity(ax, color='grey', ls='--', alpha=0.4)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    if ylim is not None:
        if ylim[1] == 1:
            # If max ylim is 1, add ticks for 0, 0.2, 0.4, 0.6, 0.8, 1 on both x and  y axis
            ax.set_xticks(np.arange(0, 1.2, 0.2))
            ax.set_yticks(np.arange(0, 1.2, 0.2))
        if ylim[1] == 0.8:
            # set 0, 0.2, 0.4, 0.6, 0.8 on both x and y axis
            ax.set_xticks(np.arange(0, 0.9, 0.2))
            ax.set_yticks(np.arange(0, 0.9, 0.2))
    ax.set_xlabel(f'{target1} {value_of_interest}', size=11)
    ax.set_ylabel(f'{target2} {value_of_interest}', size=11)
    ax.set_title(f'{target2} vs. {target1} {d_randnetw[randnetw][1:]} '
                 f'{value_of_interest}.\nPearson r2={r2:.3f}, p={p:.3e}',
                 size='medium')
    ax.set_aspect('equal')
    # add r2 in bottom right corner
    ax.text(0.95, 0.02, f'$R^2$={r2:.3f}', horizontalalignment='right', verticalalignment='bottom',
            transform=ax.transAxes, size=16)
    # Legend according to scatter colors
    class_colours = color_order_unique
    recs = []
    for i in range(0, len(class_colours)):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
    fig.legend(recs, color_order_unique_names, bbox_to_anchor=(1.0, 1.),
               fontsize=9)
    if save:
        save_str = f'across-layers-n={n_layers}-models_' \
                   f'{target1}-vs-{target2}' \
                   f'{d_randnetw[randnetw]}_' \
                   f'scatter{add_savestr}_' \
                   f'roi-{roi}_{value_of_interest}_ylim-{ylim}'
        fig.savefig(join(save, f'{save_str}.png'), dpi=180)
        fig.savefig(join(save, f'{save_str}.svg'), dpi=180)
        
        # Add metadata to df_grouped
        df_grouped['randnetw'] = randnetw
        df_grouped['roi'] = roi
        df_grouped['value_of_interest'] = value_of_interest
        df_grouped['r'] = r
        df_grouped['r2'] = r2
        df_grouped['p-val'] = p
        df_grouped['datetag'] = datetag
        df_grouped.to_csv(join(save, f'{save_str}.csv'))
    plt.tight_layout(pad=3)
    fig.show()

def add_int_to_meta_col(df_meta_roi,
                        col_name):
    """
    Transform a column in df_meta_roi to integer values instead of strings. Requires mapping dictionaries.
    Currently only supports roi_label_general.
    
    :param df_meta_roi: df_meta_roi with meta columns and voxel_id
    :param col_name: str
    :return:
    """
    
    if col_name == 'roi_label_general':
        unique_roi_label_general = df_meta_roi.roi_label_general.unique()
    
        # Create a new column roi_label_general_int
        df_meta_roi['roi_label_general_int'] = df_meta_roi.copy(deep=True).roi_label_general.map(d_roi_label_general_int)
        # Fill this col with 0 if nan
        df_meta_roi.roi_label_general_int.fillna(0, inplace=True)
    else:
        raise ValueError(f'{col_name} is not supported yet.')
    
    return df_meta_roi


#### SURFACE PLOTTING FUNCTIONS ####
def direct_plot_val_surface(output,
                            df_meta_roi,
                            val='kell_r_reliability',
                            selected_layer=None,
                            ):
    """takes an output df and packages it in a df where the val of interest directly will be used as the value to
        plot on the surface.
        Ultimately just creates a df with index voxel id and a column with the plotting value of interest.

    If we plot median r2 test, we need a selected layer.
        

    :return: df with index as voxel id and "layer_pos" column as the argmax layer as a str, and
            "rel_pos" column as the argmax relative layer position (given the model)

            The column "rel_pos" is zero indexed, i.e. 0 is the first layer, and if the model has 11 layers,
            then 10 is the top layer.
    """
    if val in output.columns and val in df_meta_roi.columns:
        # Check whether meta aligns with the output df
        arbitrary_test_layer = output.source_layer.unique()[0]
        assert (output.query(f'source_layer == "{arbitrary_test_layer}"')[val].values == df_meta_roi[val].values).all()
    else:
        print(f'{val} not in output df. Skipping check between output and df meta roi.')

    # Various transformations to get the value of interest to e.g. int
    if val == 'roi_label_general':
        # Transform from str to int
        df_meta_roi = add_int_to_meta_col(df_meta_roi=df_meta_roi,
                                          col_name=val)
    elif val == 'kell_r_reliability' or val == 'pearson_r_reliability':
        # Multiply by 10
        df_meta_roi[f'{val}*10'] = df_meta_roi[val] * 10

    elif val == 'median_r2_test_c':
        print(f'Using selected layer {selected_layer} to get median r2 test c.')

        # Check voxel_id aligns with the output df
        assert (output.query(f'source_layer == "{selected_layer}"').voxel_id.values == df_meta_roi.voxel_id.values).all()

        # Add in the median r2 test c and multiply by 10 and +1  to offset from zero
        df_meta_roi['median_r2_test_c'] = (output.query(f'source_layer == "{selected_layer}"').median_r2_test_c.values * 10) + 1

    elif val == 'shared_by':
        df_meta_roi['shared_by'] = df_meta_roi['shared_by'].astype(int)

    else:
        raise ValueError(f'{val} is not supported yet.')
        
    # Create df with voxel id and value of interest
    df_plot_direct = df_meta_roi.set_index('voxel_id').copy()
    
    return df_plot_direct

def best_layer_voxelwise(output,
                         source_model,
                         target,
                         df_meta_roi,
                         roi=None,
                         value_of_interest='median_r2_test_c',
                         randnetw='False',
                         save=True):
    """
    Loads output from a neural dataset, creates a layer x value of interest pivot (no aggregation)
    Obtains the best layer for each voxel, and its associated score. Take median across subjects, and save the
    [subject x val] pivot table.
    
    :param output:
    :param source_model:
    :param value_of_interest:
    :param randnetw:
    :param save:
    :return:
    """
    print(f'\nFUNC: best_layer_voxelwise\nMODEL: {source_model}, value_of_interest: {value_of_interest}, randnetw: {randnetw}\n')
    
    if roi:
        output = output.loc[output['roi'] == roi]
    
    p = get_vox_by_layer_pivot(output, source_model, val_of_interest=value_of_interest) # indexed by voxel id
    assert (np.sum(np.isnan(p)).values == 0).all()
    
    idxmax_layer, num_layers = layer_position_argmax(p, source_model)
    idxmax_layer.rename(columns={'val':value_of_interest}, inplace=True) # rename the value of interest, as used in the get_vox_by_layer_pivot
    # to the value that was used, to keep logging
    
    # Group by subject (as in df_meta_roi) and take median across subjects
    assert(idxmax_layer.index == df_meta_roi.voxel_id).all() # assert that voxel id matches
    df_meta_roi = df_meta_roi.set_index('voxel_id', inplace=False)
    
    idxmax_layer['subj_idx'] = df_meta_roi['subj_idx']
    
    idxmax_layer_aggregate = idxmax_layer.groupby('subj_idx').median()
    
    # log
    idxmax_layer_aggregate['source_model'] = source_model
    idxmax_layer_aggregate['target'] = target
    idxmax_layer_aggregate['roi'] = roi
    idxmax_layer_aggregate['randnetw'] = randnetw
    
    if save:
        idxmax_layer_aggregate.to_csv(join(save, f'best-layer_voxelwise_'
                                                 f'{source_model}{d_randnetw[randnetw]}_'
                                                 f'{target}_{value_of_interest}.csv'))
        
    


def layer_position_argmax(p,
                          source_model,
                          layers_to_exclude=None):
    """
    :param p: pd df, pivot table with number voxels as rows, and number layers as columns
    :param source_model: str, name of the source model

    :return: idxmax_layer, pd df with number voxels as rows, and the argmax layer (string) as the value (in column: 'layer_pos').
             and the int position of the argmax layer (int). +1 to make the min layer 1 and not 0 (in column: 'pos')
             and the relative position (in column: 'rel_pos') where 0 is the first layer, and 1 is the last layer.
    """
    np.random.seed(0) # set seed for reproducibility

    ## Get num, min, max layers to start with ##
    num_layers = p.shape[1]
    layer_reindex = d_layer_reindex[source_model]
    layer_legend = [d_layer_names[source_model][l] for l in layer_reindex]
    assert(len(layer_legend) == num_layers)
    if layers_to_exclude:
        for layer in layers_to_exclude:
            if layer == 'input_after_preproc':
                layer_legend_name = 'Cochleagram'
            else:
                raise ValueError(f'Not implemented yet for layers_to_exclude: {layers_to_exclude}')
            layer_legend = [l for l in layer_legend if l != layer_legend_name]

    print(f'{source_model} has {num_layers} layers')

    d_rel_layer_reindex = {}  # get idx of each layer position
    idx = 0
    for l in (layer_reindex):
        if layers_to_exclude is not None:
            if l in layers_to_exclude:
                print(f'Omitting layer {l} from layer_position_argmax analysis for {source_model}')
                continue
        d_rel_layer_reindex[l] = idx
        idx += 1 # only make the index dependent on included layers

    # Obtain the MIN and MAX layer positions (int)
    min_possible_layer = min(d_rel_layer_reindex.values()) + 1  # for MATLAB, and to align with "pos"
    max_possible_layer = max(d_rel_layer_reindex.values()) + 1  # for MATLAB and to align with "pos"
    assert (max_possible_layer == num_layers)
    
    if layers_to_exclude is not None:
        layer_reindex = [l for l in d_rel_layer_reindex if l not in layers_to_exclude]
        
    assert(p.columns == layer_reindex).all() # make sure ordering is ok
    
    ## Iterate over voxels/components, and get the argmax layer position ##
    idxmax_layer_pos = [] # layer position (string) of the argmax layer
    idxmax_layer_legend = [] # layer legend (string) of the argmax layer
    idxmax_pos = [] # position of the argmax layer (+1 to make it start at 1)
    idxmax_associated_val = [] # for storing the associated value of the argmax layer
    
    # Quantify how many voxels have ties with either all layers = 0 or all layers = 1
    sum_pred_across_layers = p.sum(axis=1)
    num_vox_with_all_layers_0 = sum_pred_across_layers[sum_pred_across_layers == 0].shape[0]
    num_vox_with_all_layers_1 = sum_pred_across_layers[sum_pred_across_layers == num_layers].shape[0]
    
    print(f'{num_vox_with_all_layers_0} voxels have all layers = 0, and {num_vox_with_all_layers_1} voxels have all layers = 1')
    # but we can also have ties if just e.g., 2 layers are 1.

    # If ties exist, take a random of the tied (idxmax) values for each voxel (row):
    # First find max val and check how many columns have that value
    max_val_each_vox = p.max(axis=1)
    tie_counter = 0
    tie_bool = []
    tie_nums = []
    tie_nums_if_exist = [] # if a tie exists, how many values were tied (ONLY IF THEY EXIST)
    unique_tie_values = []
    for idx, (i, row) in enumerate(p.iterrows()):
        ties = row == max_val_each_vox.iloc[idx]
        # print(max_val_each_vox.iloc[idx])
        if np.sum(ties) > 1:
            tie_counter += 1
            tie_bool.append(1)
            tie_nums_if_exist.append(np.sum(ties))
            tie_nums.append(np.sum(ties))
            # print(f'{ties},\n {row}')
            # print(f'Voxel/comp {idx}, ties exist: {np.sum(ties)} with the value: {max_val_each_vox.iloc[idx]}')
            unique_tie_values.append(max_val_each_vox.iloc[idx])
        else:
            tie_bool.append(0)
            tie_nums.append(0)
            
        # Random_layer_selected is the argmax layer. If there are ties, take a random of the tied values
        random_layer_selected = np.random.choice(np.flatnonzero(ties)) # provides the index of the layer. If no ties, it is just the argmax layer
        # print(f'i is {i} and idx is {idx}')
        # Check that the layer selected is indeed the argmax layer!
        assert(p.iloc[idx, random_layer_selected] == max_val_each_vox.iloc[idx])
        
        # if np.sum(ties) > 1:
            # print(f'Randomly selected layer: {p.columns[random_layer_selected]} of out possible {p.columns[ties.values].values}\n')
        
        # Log (by indexing using the best layer position)
        idxmax_layer_pos.append(p.columns[random_layer_selected])
        idxmax_layer_legend.append(layer_legend[random_layer_selected])
        idxmax_pos.append(d_rel_layer_reindex[p.columns[random_layer_selected]] + 1)
        idxmax_associated_val.append(p.iloc[idx, random_layer_selected])
    
    # Package into a df and compute relative position
    df_best_layer = pd.DataFrame({'layer_pos': idxmax_layer_pos,
                                 'layer_legend': idxmax_layer_legend,
                                 'pos': idxmax_pos,
                                 'rel_pos': np.divide((np.subtract(idxmax_pos, min_possible_layer)),
                                        (max_possible_layer - min_possible_layer)),
                                 'val': idxmax_associated_val,
                                 'tie_bool': tie_bool,
                                'tie_nums': tie_nums,
                                 }, index=p.index)
    
    # If values in unique_tie_values are smaller than e-6, set them to 0 for count purposes:
    unique_tie_values = [x if x > 1e-6 else 0 for x in unique_tie_values]
    unique, count = np.unique(unique_tie_values, return_counts=True)
    counts = dict(zip(unique, count))
    print(f'\nTIE COUNTER: {tie_counter} with unique tie values: {counts}')
    if not len(tie_nums_if_exist) == 0:
        print(f'\nMean/median number of values that were tied (IF any): {np.mean(tie_nums_if_exist):.3}, {np.median(tie_nums_if_exist)}'
              f' (std: {np.std(tie_nums_if_exist):.3})')

    return df_best_layer, num_layers


def barplot_best_layer_per_anat_ROI(output,
                                    meta,
                                    source_model,
                                    target,
                                    randnetw='False',
                                    value_of_interest='median_r2_test_c',
                                    val_layer='rel_pos',
                                    yerr_type='within_subject_sem',
                                    save=False,
                                    condition_col='roi_anat_hemi',
                                    collapse_over_val_layer='median',
                                    layers_to_exclude=None):
    """takes an output df and find the argmax best layer of interest for a metric (value_of_interest).
    Separate into ROIs and plot a barplot.
    Take median across voxels for each subject. Obtain SEM (within-subject).
    
    Edit 20220729: add in option to add in dim values! Does not change anything about the rest of the script.
    
    Specifically, the process is:
    1. Obtain vox by layer matrix [7694;11]
    2. Obtain idxmax layer matrix [7694;1] where the column is the relative position (rel_pos) of that voxel’s best predicting layer
    3. Obtain p_plot: this takes the median over the rel_pos values per subject for each condition of interest (collapse_over_val_layer)
        (for instance, 8: ['Anterior_rh', 'Anterior_lh', 'Primary_rh', 'Primary_lh', 'Lateral_rh', 'Lateral_lh', 'Posterior_rh', 'Posterior_lh', ])
    4. This yields a matrix of size [num subjects; num cond] ([8;8]) where each value in the matrix is the median best performing relative layer for that subject
    5. Obtain mean value per condition (8 values)
    6. Obtain between subject error bars (8 values)

    :param output: pd df, output df
    :param meta: pd df, meta df
    :param source_model: string, source model
    :param target: string, target model
    :param value_of_interest: string, metric to use to obtain the argmax across layers
    :param val_layer: string, metric to use in plotting the argmax. 'pos' is the actual position (int) of the argmax layer,
            while 'rel_pos' is the relative position [0 1]
            'pos' is NOT zero indexed, i.e. 1 is the first layer, and if the model has 11 layers, then 11 is the top layer.
    :param yerr_type: string, type of error bar to use. 'within_subject_sem' is the within-subject SEM, 'within_subject_std' is the within-subject STD
    :param save: boolean, whether to save the plot
    :param condition_col: string, column in meta df that contains the condition of interest (i.e. the ROI labels)
    :param collapse_over_val_layer: string, metric to use to collapse across conditions. 'median' is the median across conditions,
            which is the default -- hence, the median across layers positions for each voxel, within each ROI, is obtained.
    :param layers_to_exclude: list of strings, layers to exclude from the analysis (default is None!)
    

    :return: plot

    """
    label_rotation = 70
    print(f'\nFUNC: barplot_best_layer_per_anat_ROI\nMODEL: {source_model}, TARGET: {target}, VALUE OF INTEREST: {value_of_interest}, VAL_LAYER: {val_layer}, RANDNETW: {randnetw}')
    
    if condition_col == 'roi_anat_hemi':
        roi_anat_reindex = ['Anterior_rh', 'Anterior_lh', 'Primary_rh', 'Primary_lh', 'Lateral_rh', 'Lateral_lh', 'Posterior_rh', 'Posterior_lh', ]
        bar_placement = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    if condition_col == 'roi_label_general':
        roi_anat_reindex = ['Anterior', 'Primary', 'Lateral', 'Posterior']
        bar_placement = [0, 0.5, 1, 1.5]

    # assert that the values of interest exist in the df
    if not value_of_interest in output.columns:
        print(f'Column {value_of_interest} does not exist in the output df')
        return
    
    output2 = output.copy(deep=True)
    source_model_check = output2['source_model'].unique()
    assert (len(source_model_check) == 1)
    
    p = get_vox_by_layer_pivot(output=output2, source_model=source_model,
                               val_of_interest=value_of_interest)
    
    ### Exclude layers from the analysis if layers_to_exclude is not None
    if layers_to_exclude is not None:
        p = p.drop(layers_to_exclude, axis=1)
    
    # assert that pivot vals and meta match
    assert (np.array_equal(p.index, meta.voxel_id.values))
    
    idxmax_layer, num_layers = layer_position_argmax(p=p,  # p has vox idx as the index
                                                     source_model=source_model,
                                                     layers_to_exclude=layers_to_exclude,)
    meta_indexed = meta.set_index('voxel_id', inplace=False)
    
    # Count how many ties exist (tie_bool) in the condition col
    meta_indexed[condition_col] = meta_indexed[condition_col].fillna(0, inplace=False) # We have strings and NaNs, so fill NaNs with 0
    idxmax_in_cond_col = idxmax_layer.loc[meta_indexed.query(f'{condition_col} != 0').index]
    print(f'Percent of ties in {condition_col}: {(idxmax_in_cond_col.tie_bool.sum() / idxmax_in_cond_col.shape[0]) * 100:.3}%')
    
    # Obtain bars per ROI
    df_plot = pd.concat([idxmax_layer, meta_indexed], axis=1)

    # Create a column with the ROI annotation per voxel -- then I can make a subject X ROI cond pivot and compute the error bars
    p_plot = df_plot.pivot_table(index='subj_idx', columns=condition_col, values=val_layer, aggfunc=collapse_over_val_layer)
    p_plot = p_plot[roi_anat_reindex] # reindex the conditions
    
    # Check p_plot by manually extracting values
    # df_plot_test = df_plot.query('subj_idx == 0 & roi_label_general == "Primary"')
    # df_plot_test.median()['rel_pos']
    
    demeaned = p_plot.subtract(p_plot.mean(axis=1).values, axis=0)  # get a mean value for each subject across all conds, and subtract it from individual layers
    
    if yerr_type == 'within_subject_sem':
        yerr = np.std(demeaned.values.T, axis=1) / np.sqrt(p_plot.shape[0] - 1)
    
    # Assert that plotted results have no nans
    assert(p_plot.isna().sum().sum() == 0)
    
    color_order = [d_roi_colors[x] for x in p_plot.columns]

    fig, ax = plt.subplots(figsize=(4, 5))
    plt.title(f'{source_model}{d_randnetw[randnetw]}\n{target}', size='medium')
    ax.bar(bar_placement,
           p_plot.mean(),
           yerr=yerr,
           width=0.3, color=color_order, alpha=0.8)
    plt.xticks(bar_placement, p_plot.columns, rotation=label_rotation)
    plt.ylabel(d_value_of_interest[val_layer])
    plt.tight_layout()
    if save:
        if layers_to_exclude:
            layer_exlusion_str = '-'.join(layers_to_exclude)
        else:
            layer_exlusion_str = 'None'
        
        save_str = f'best-layer_barplot_{condition_col}_' \
                   f'{source_model}{d_randnetw[randnetw]}_' \
                   f'{target}_{yerr_type}_' \
                   f'{val_layer}-{collapse_over_val_layer}_' \
                   f'{value_of_interest}_' \
                   f'layer-exclude-{layer_exlusion_str}'
        plt.savefig(join(save, f'{save_str}.svg'))
        plt.savefig(join(SAVEDIR_CENTRALIZED, f'{save_str}.svg'))

        # compile csv
        p_plot_save = p_plot.copy()
        df_yerr = pd.DataFrame([yerr], columns=p_plot_save.columns,
                               index=['yerr'])  # append yerr to the pivot table that is plotted
        p_plot_save = p_plot_save.append(df_yerr)
        p_plot_save.to_csv(join(save, f'{save_str}.csv'))

        # Also store the df_plot
        df_plot['datetag'] = datetag
        df_plot['value_of_interest'] = value_of_interest
        df_plot['val_layer'] = val_layer

        df_plot.to_csv(join(save, f'{save_str}_df-plot.csv'))

    plt.show()



def surface_argmax(output,
                   source_model,
                   target,
                   randnetw='False',
                   value_of_interest='median_r2_test_c',
                   hist=True,
                   save=True,
                   save_full_idxmax_layer=False):
    """Takes an output df and find the argmax best layer of interest for a metric (value_of_interest).
       Performs this over all voxels and subjects, and then plots the distribution of the argmax layer.

    :return: df with index as voxel id and "layer_pos" column as the argmax layer as a str, and
            "rel_pos" column as the argmax relative layer position (given the model)
            
            The columns "pos" and "rel_pos" is NOT zero indexed, i.e. 1 is the first layer, and if the model has 11 layers,
            then 11 is the top layer.
    """
    label_rotation = 70
    layer_reindex = d_layer_reindex[source_model]
    layer_legend = [d_layer_names[source_model][layer] for layer in layer_reindex]
    
    # assert that the values of interest exist in the df
    if not value_of_interest in output.columns:
        print(f'Column {value_of_interest} does not exist in the output df')
        return
    
    output2 = output.copy(deep=True)
    source_model_check = output2['source_model'].unique()
    assert (len(source_model_check) == 1)
    
    p = get_vox_by_layer_pivot(output=output2,
                               source_model=source_model,
                               val_of_interest=value_of_interest)
    
    idxmax_layer, num_layers = layer_position_argmax(p=p, source_model=source_model,)

    if save_full_idxmax_layer:
        idxmax_layer_save = idxmax_layer.copy(deep=True)
        idxmax_layer_save['target'] = target
        idxmax_layer_save['source_model'] = source_model
        idxmax_layer_save['randnetw'] = randnetw
        idxmax_layer_save['value_of_interest'] = value_of_interest
        idxmax_layer_save['datetag'] = datetag
        idxmax_layer_save['num_layers'] = num_layers
        idxmax_layer_save.to_csv(join(save, f'idxmax_layer_full_{source_model}_{target}_{randnetw}_{value_of_interest}.csv'))


    if hist:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_box_aspect(0.6)
        plt.hist(idxmax_layer.pos.values, bins=num_layers)
        plt.xlabel('Layer')
        plt.ylabel('Voxel count')
        plt.xticks(np.arange(1, num_layers + 1), layer_legend, rotation=label_rotation)
        plt.title(f'{d_model_names[source_model]}{d_randnetw[randnetw]}: Relative layer preference histogram\nAll subjects, {target} (n={idxmax_layer.shape[0]} voxels)',
                  size='small')
        plt.tight_layout(pad=2)
        if save:
            plt.savefig(join(save, f'TYPE=subj-argmax_METRIC={value_of_interest}_'
                                   f'{source_model}{d_randnetw[randnetw]}_'
                                   f'{target}.png'))
        plt.show()
        
        if save:
            # Save the layer number of the argmax layer, i.e. the most preferred layer across all voxels
            # (actually, just save the histogram values)
            pos_count_save = pd.DataFrame(
                idxmax_layer.groupby('layer_pos').count().reindex(layer_reindex)['pos']).rename(
                columns={0: 'voxel_count'})
            pos_count_save['layer_legend'] = layer_legend
            pos_count_save['target'] = target
            pos_count_save['source_model'] = source_model
            pos_count_save['randnetw'] = randnetw
            pos_count_save['value_of_interest'] = value_of_interest
            pos_count_save['datetag'] = datetag
            pos_count_save['num_layers'] = num_layers
            pos_count_save.to_csv(join(save, f'hist-preferred-layer_'
                                             f'TYPE=subj-argmax_'
                                             f'METRIC={value_of_interest}_'
                                             f'{source_model}{d_randnetw[randnetw]}_'
                                             f'{target}.csv'))

    return idxmax_layer, p.columns.values

def surface_argmax_hist_merge_datasets(df_plot1,
                                       df_plot2,
                                       source_model,
                                       save,
                                       layer_names,
                                       target='NH2015-B2021',
                                       randnetw='False',
                                       value_of_interest='median_r2_test_c',
                                       save_full_idxmax_layer=True,):
    """Obtain a layer preference histogram to determine which layers to use for colorscale
    by taking both datasets into account.

    df_plot1 and df_plot2 should come from each dataset, and are the outputs of surface_argmax().

    """
    label_rotation = 70
    target_check = 'NH2015-B2021'
    assert (target == target_check)
    layer_reindex = d_layer_reindex[source_model]
    layer_legend = [d_layer_names[source_model][layer] for layer in layer_reindex]
    
    assert (layer_names == layer_reindex).all()
    
    # Merge the two datasets to be plotted
    df = pd.concat([df_plot1, df_plot2])
    if save_full_idxmax_layer:
        idxmax_layer_save = df.copy(deep=True)
        idxmax_layer_save['target'] = target
        idxmax_layer_save['source_model'] = source_model
        idxmax_layer_save['randnetw'] = randnetw
        idxmax_layer_save['value_of_interest'] = value_of_interest
        idxmax_layer_save['datetag'] = datetag
        idxmax_layer_save['num_layers'] = len(layer_names)
        idxmax_layer_save.to_csv(join(save, f'idxmax_layer_full_{source_model}_{target}_{randnetw}_{value_of_interest}.csv'))

    plt.figure(figsize=(7, 5))
    plt.hist(df.pos.values, bins=len(layer_names))
    plt.xlabel('Layer')
    plt.ylabel('Voxel count')
    plt.xticks(np.arange(1, len(layer_names) + 1), layer_legend, rotation=label_rotation)
    plt.title(f'{d_model_names[source_model]}{d_randnetw[randnetw]}: '
              f'Relative layer preference histogram\nAll subjects, {target} (n={df.shape[0]} voxels)',
              size='small')
    plt.tight_layout()
    if save:
        plt.savefig(join(save, f'TYPE=subj-argmax_'
                               f'METRIC={value_of_interest}_'
                               f'{source_model}{d_randnetw[randnetw]}_'
                               f'{target}.png'))
    plt.show()

    # Save the layer number of the argmax layer, i.e. the most preferred layer across all voxels
    # (actually, just save the histogram value_of_interestues)
    pos_count_save = pd.DataFrame(df.groupby('layer_pos').count().reindex(layer_reindex)['pos']).rename(
        columns={0: 'voxel_count'})
    pos_count_save['layer_legend'] = layer_legend
    pos_count_save['target'] = target
    pos_count_save['source_model'] = source_model
    pos_count_save['randnetw'] = randnetw
    pos_count_save['value_of_interest'] = value_of_interest
    pos_count_save['datetag'] = datetag
    pos_count_save['num_layers'] = len(layer_names)
    if save:
        pos_count_save.to_csv(join(save,
                                   f'hist-preferred-layer_'
                                   f'TYPE=subj-argmax_'
                                   f'METRIC={value_of_interest}_'
                                   f'{source_model}{d_randnetw[randnetw]}_'
                                   f'{target}.csv'))



def create_avg_subject_surface(df_plot,
                               meta,
                               save,
                               source_model,
                               target,
                               val_of_interest,
                               randnetw='False',
                               plot_val_of_interest='pos',):
    """
    Takes the median over an array of values to plot across the same brain coordinate (based on x and y ras) across subjects

    NOTE: this function is named _avg, but by default we use the median!

    :param df_plot: df with plotting values (in the column "plot_vals"), index is voxel_id.
    :param meta: df with voxel metadata, has a "voxel_id" column.
    :return: df with rows corresponding to unique voxel coordinates, and the "averaged" plot value as "median_plot_val"
    """
    unique_coords = meta.coord_id.unique()
    
    # assert that vals and meta match
    assert (np.array_equal(df_plot.index, meta.voxel_id.values))
    
    # append the plot vals to the meta plot df
    meta['plot_vals'] = df_plot[f'{plot_val_of_interest}'].values
    meta_unique_coords = pd.DataFrame(columns=['coord_id', 'x_ras', 'y_ras', 'hemi', 'shared_by', 'median_plot_val'])

    # Iterate over the unique brain coordinates we have (to find the median plot value for each coordinate)
    for i, coord_val in enumerate(unique_coords):
        c = coord_val.split('_')
        x = int(c[0])
        y = int(c[1])
        # append to df
        meta_unique_coords.loc[i, 'coord_id'] = coord_val
        meta_unique_coords.loc[i, 'x_ras'] = int(x)
        meta_unique_coords.loc[i, 'y_ras'] = int(y)
        # look up hemi in the other, meta df
        match_row_idx = meta.loc[np.logical_and(meta['x_ras'] == x, meta['y_ras'] == y)].index.values
        # assert that the coordinate only belongs to one, unique hemi
        unique_hemi = meta.loc[match_row_idx, 'hemi'].unique()
        unique_shared_by = meta.loc[match_row_idx, 'shared_by'].unique()
        unique_plot_vals = meta.loc[match_row_idx, 'plot_vals'].values
        unique_voxel_ids = meta.loc[match_row_idx, 'voxel_id'].values
        assert (len(unique_hemi) == 1)
        assert (len(unique_shared_by) == 1)
        meta_unique_coords.loc[i, 'hemi'] = unique_hemi[0]
        meta_unique_coords.loc[i, 'shared_by'] = unique_shared_by[0] # How many subjects share this coordinate
        
        # average (i.e. median...!!) across plot vals and assert that the number of values match the shared_by col
        assert (len(unique_plot_vals) == unique_shared_by)
        meta_unique_coords.loc[i, 'median_plot_val'] = np.median(unique_plot_vals)

    # if 50 < i < 80:
    # 	print(f'Unique plotting vals to average over: {unique_plot_vals} and median: {np.median(unique_plot_vals)}')
    
    if save: # store the median values for all brain coords
        meta_unique_coords_save = meta_unique_coords.copy(deep=True)
        meta_unique_coords_save['source_model'] = source_model
        meta_unique_coords_save['target'] = target
        meta_unique_coords_save['randnetw'] = randnetw
        meta_unique_coords_save['val_of_interest'] = val_of_interest # The original name of the value, pre transformations
        meta_unique_coords_save['plot_val_of_interest'] = plot_val_of_interest # The name of the value, post transformations
        meta_unique_coords_save['datetag'] = datetag
        meta_unique_coords_save.to_csv(join(save, f'surf_coords_full_{source_model}_'
                                                  f'{target}_'
                                                  f'{randnetw}_'
                                                  f'{val_of_interest}_'
                                                  f'{plot_val_of_interest}.csv'))
    
    return meta_unique_coords

def create_avg_model_surface(source_models,
                             target,
                             PLOTSURFDIR,
                             val_of_interest='median_r2_test_c',
                             randnetw='False',
                             plot_val_of_interest='rel_pos',
                             quantize=False):
    """
    Takes the surf_coords_full output (unique coordinates) and take the median across the source models.
    The surf_coords_full is already with the median obtained across subjects (e.g. for NH2015, it has 1945 unique coords,
    corresponding to unique brain coordinates which are rows in the df).
    """

    df_lst = []
    lst_coord_ids = [] # for checking

    # Iterate over source models and load the surf_coords_full df
    for source_model in source_models:
        meta_unique_coords = pd.read_csv(join(PLOTSURFDIR,
                                              f'surf_coords_full_{source_model}_'
                                              f'{target}_'
                                              f'{randnetw}_'
                                              f'{val_of_interest}_'
                                              f'{plot_val_of_interest}.csv'))
        df_lst.append( meta_unique_coords)
        assert (len(meta_unique_coords) == len(meta_unique_coords.coord_id.unique()))

        # Also assert that coord_id is the same across all meta_unique_coords df loaded!
        lst_coord_ids.append(meta_unique_coords.coord_id.values)
        
    df_all = pd.concat(df_lst)
    assert(len(df_all) == len(meta_unique_coords)*len(source_models)) # ensure that all models are stacked as rows
    
    # Take the last df as a template for the coordinates (all dfs are grouped by coord_id)
    df_template = meta_unique_coords.copy(deep=True).drop(columns=['median_plot_val'])
    
    # Obtain the median across models for each unique coordinate
    df_across_models = pd.DataFrame(df_all.groupby('coord_id')['median_plot_val'].median() * 10, columns=['median_plot_val'])
    df_across_models['median_plot_val'] = df_across_models['median_plot_val'] + 1 # avoid 0 values in the KNN interpolation
    # first layer (which before was 0) will now be 1 and the last layer is now e.g., 11 (in principle, if the median across models is 11 at some voxel).
    
    print(f'Min and max median across models: {df_across_models.median_plot_val.min()} and {df_across_models.median_plot_val.max()} ({val_of_interest})\n'
          f'Unique coords: {len(df_across_models)} for {target}.\n')
    
    if quantize:
        df_across_models['median_plot_val'] = df_across_models['median_plot_val'].apply(lambda x: round(x))
        print(
            f'Post quantization: Min and max median across models: {df_across_models.median_plot_val.min()} and {df_across_models.median_plot_val.max()}\n')

    # We want to retain hemi, x_ras, y_ras, shared_by as in the original meta_unique_coords
    # Assert whether the coord_id in df_template matches the grouped df
    assert (np.asarray(
        [lst_coord_ids[i] == lst_coord_ids[i - 1] for i in range(len(lst_coord_ids))]).ravel() == True).all()
    
    # The groupby operation messes up the coord_id order, so reindex to match the template
    df_across_models = df_across_models.reindex(df_template.coord_id)
    
    assert (df_template.coord_id.values == df_across_models.index.values).all()
    
    # Add in the new median_plot_val which is now the median across models
    df_template['median_plot_val'] = df_across_models['median_plot_val'].values
    
    return df_template.drop(columns=['Unnamed: 0', 'source_model'])


def dump_for_surface_writing_direct(df_plot_direct,
                                    source_model,
                                    SURFDIR,
                                    randnetw,
                                    val='kell_r_reliability',
                                    subfolder_name='direct-plot', ):
    """
    Takes a df (df_plot_direct) with x_ras, y_ras columns. The column to be plotted is by default: median_plot_val
    Writes a matlab structure in SURFDIR for each hemisphere, for all voxels (median where several subject share them)

    To actually write the surfaces, use the write_surfs.m matlab function (which takes advantage of freesurfer's matlab library).

    :param median_subj: df
    :param source_model: str
    :param SURFDIR: str
    :param subfolder_name: str
    :return:
    """
    SAVEFOLDER = os.path.join(SURFDIR, f'{source_model}{d_randnetw[randnetw]}', f'{subfolder_name}')
    Path(SAVEFOLDER).mkdir(parents=True, exist_ok=True)
    
    hemis = list(set(df_plot_direct['hemi']))
    for hemi in hemis:
        is_hemi = df_plot_direct['hemi'].values == hemi
        file = os.path.join(f'{source_model}_'
                            f'{val}_{hemi}.mat')
        
        dict_ = {'vals': df_plot_direct[val][is_hemi].astype('int64').values,
                 'x_ras': df_plot_direct['x_ras'][is_hemi].astype('int64').values,
                 'y_ras': df_plot_direct['y_ras'][is_hemi].astype('int64').values}
        savemat(os.path.join(SAVEFOLDER, file), dict_)
        print(f'Saved mat file for median subject: {file}, number of vertices {np.sum(is_hemi)}')
    

def dump_for_surface_writing_avg(median_subj,
                                 source_model,
                                 SURFDIR,
                                 PLOTSURFDIR,
                                 randnetw,
                                 subfolder_name='subj-median-argmax', ):
    """
    Takes a df (median_subj) with x_ras, y_ras columns. The column to be plotted is by default: median_plot_val
    Writes a matlab structure in SURFDIR for each hemisphere, for all voxels (median where several subjects share them)

    To actually write the surfaces, use the write_surfs.m matlab function (which takes advantage of freesurfer's matlab library).

    NOTE: this function is named _avg, but by default we use the median!

    In PLOTSURDIR, we store the csv file that was used to generate the matlab structure.

    :param median_subj: df
    :param source_model: str
    :param SURFDIR: str
    :param subfolder_name: str
    :return:
    """

    if SURFDIR:
        SAVEFOLDER = os.path.join(SURFDIR, f'{source_model}{d_randnetw[randnetw]}', f'{subfolder_name}')
        Path(SAVEFOLDER).mkdir(parents=True, exist_ok=True)

    if PLOTSURFDIR:
        fname = f'{subfolder_name}_{source_model}{d_randnetw[randnetw]}_subj-avg.csv'
        median_subj.to_csv(os.path.join(PLOTSURFDIR, fname), index=False)
    
    hemis = list(set(median_subj['hemi']))
    for hemi in hemis:
        is_hemi = median_subj['hemi'].values == hemi
        file = os.path.join(f'{source_model}_'
                            f'subj-avg_{hemi}.mat')
        
        dict_ = {'vals': median_subj['median_plot_val'][is_hemi].astype('int64').values,
                 'x_ras': median_subj['x_ras'][is_hemi].astype('int64').values,
                 'y_ras': median_subj['y_ras'][is_hemi].astype('int64').values}

        if SURFDIR:
            savemat(os.path.join(SAVEFOLDER, file), dict_)
            print(f'Saved mat file for median subject: {SAVEFOLDER}/{file}, number of vertices {np.sum(is_hemi)}')


def dump_for_surface_writing(vals,
                             meta,
                             source_model,
                             SURFDIR,
                             randnetw,
                             subfolder_name='subj-argmax'):
    """
    Iterates over subjects specified in meta df, and writes a subject-wise surface file for each.
    
    Take <vals> and x_ras, y_ras (from meta) and dump a matlab structure in SURFDIR.
    Writes a unique mat for each subject.

    to actually write the surfaces, use the write_surfs.m matlab function
        (which takes advantage of freesurfer's matlab library)
    
    Base function from Alex Kell.
    
    :param vals: Set of values to plot, indexed by voxel_id
    :param meta: df with metadata, indexed by voxel_id
    :param source_model: str
    :param SURFDIR: str
    :param subfolder_name: str

    :return: stores .mat files in SURFDIR
    """
    # must be one d
    if vals.ndim > 1:
        raise Exception('need to have 1-d input')
    
    # assert that vals and meta match
    assert (np.array_equal(vals.index, meta.voxel_id.values))

    if SURFDIR:
        SAVEFOLDER = os.path.join(SURFDIR, f'{source_model}{d_randnetw[randnetw]}', f'{subfolder_name}')
        Path(SAVEFOLDER).mkdir(parents=True, exist_ok=True)

    # Get unique subject ids
    subj_ids = list(set(meta['subj_idx']))
    
    for subj_id in subj_ids:
        is_subj = meta['subj_idx'].values == subj_id
        hemis = list(set(meta['hemi'][is_subj]))
        
        for hemi in hemis:
            is_hemi = meta['hemi'].values == hemi
            
            file = os.path.join(f'{source_model}_'
                                f'{subj_id}_{hemi}.mat')
            
            is_subhemi = np.logical_and(is_subj, is_hemi)

            # print(f'Number of speech selective voxels for subject {subj_id}, hemi {hemi}: {int(np.sum(vals[is_subhemi]))}')
            dict_ = {'vals': vals.values[is_subhemi],
                     'x_ras': meta['x_ras'][is_subhemi].astype('int64').values,
                     'y_ras': meta['y_ras'][is_subhemi].astype('int64').values}

            if SURFDIR:
                savemat(os.path.join(SAVEFOLDER, file), dict_)
                print(f'Saved mat file for subject {subj_id} named: {file}, number of vertices {np.sum(is_subhemi)}')

def determine_surf_layer_colorscale(target,
                                    source_models,
                                    save,
                                    randnetw='False',
                                    value_of_interest='median_r2_test_c',):

    """Iterate through the layer preference (across voxels) histogram csv files and -- for each model --
    determine which layer to plot.

    We plot the layer that is AFTER the argmax layer (i.e. the layer that most voxels prefer).

    This function requires surface_argmax() to have been run first for each model (it stores the layer preference values
    for each voxel, which we need to determine the most preferred layer).

    """
    
    lst_source_models = []
    lst_argmax_pos = [] # pos is always 1 indexed
    lst_argmax_pos_plus1 = []
    lst_argmax_rel_pos = []
    lst_argmax_rel_pos_plus1 = []
    lst_argmax_layer = []
    lst_argmax_layer_plus1 = []
    lst_argmax_layer_legend = []
    lst_argmax_layer_plus1_legend = []


    for source_model in source_models: # Load the file that is not averaged across subjects
        df = pd.read_csv(join(save,
                        f'hist-preferred-layer_'
                        f'TYPE=subj-argmax_'
                        f'METRIC={value_of_interest}_'
                        f'{source_model}{d_randnetw[randnetw]}_'
                        f'{target}.csv'))

        # Num layers
        layer_reindex = d_layer_reindex[source_model]
        min_possible_layer = 1  # for MATLAB, and to align with "pos"
        max_possible_layer = len(layer_reindex)  # for MATLAB and to align with "pos"

        # Find argmax layer based on column 'pos' (has the pos counts, i.e. counts of position)
        argmax_layer = df.iloc[df['pos'].idxmax()]['layer_pos']
        try:
            argmax_layer_plus1 = df.iloc[df['pos'].idxmax()+1]['layer_pos']
        except:
            argmax_layer_plus1 = np.nan # If the argmax layer is the last one
        # Get layer legend too
        argmax_layer_legend = df.iloc[df['pos'].idxmax()]['layer_legend']
        try:
            argmax_layer_plus1_legend = df.iloc[df['pos'].idxmax()+1]['layer_legend']
        except:
            argmax_layer_plus1_legend = np.nan
        
        # Append variables to lists
        lst_source_models.append(source_model)
        lst_argmax_pos.append(df['pos'].idxmax() + 1) # idxmax is 0 indexed, so if we have 7 layers total, and the best one is
        # the 5th one (1-indexed), Python will return 4, so we add 1 to get 5. Pos is always 1-indexed to align with MATLAB.
        lst_argmax_pos_plus1.append(df['pos'].idxmax() + 2)
        lst_argmax_layer.append(argmax_layer)
        lst_argmax_layer_plus1.append(argmax_layer_plus1)
        lst_argmax_layer_legend.append(argmax_layer_legend)
        lst_argmax_layer_plus1_legend.append(argmax_layer_plus1_legend)
        
        # It would also be handy to have the rel pos of the argmax layer + 1 (which is to be plotted).
        # normalize by the number of layers for each model
        # Count number of layers for each model in d_layer_reindex (the value list)
        rel_pos = np.divide((np.subtract(df['pos'].idxmax() + 1, min_possible_layer)),
					 (max_possible_layer - min_possible_layer))
        lst_argmax_rel_pos.append(rel_pos)

        # Get the rel pos of the argmax layer + 1 (which is to be plotted).
        rel_pos_plus1 = np.divide((np.subtract(df['pos'].idxmax() + 2, min_possible_layer)), # +2 because +1 is from MATLAB indexing, and other one is +1
                            (max_possible_layer - min_possible_layer))
        lst_argmax_rel_pos_plus1.append(rel_pos_plus1)

    df_all = pd.DataFrame({'source_model': lst_source_models,
                           'argmax_pos': lst_argmax_pos,
                           'argmax_rel_pos': lst_argmax_rel_pos,
                           'argmax_layer': lst_argmax_layer,
                           'argmax_layer_legend': lst_argmax_layer_legend,
                           'argmax_pos_plus1': lst_argmax_pos_plus1,
                           'argmax_rel_pos_plus1': lst_argmax_rel_pos_plus1,
                           'argmax_layer_plus1': lst_argmax_layer_plus1,
                           'argmax_layer_plus1_legend': lst_argmax_layer_plus1_legend,})

    if save:
        df_all.to_csv(join(save, f'layer_pref_colorscale_'
                                 f'TYPE=subj-argmax_METRIC={value_of_interest}_'
                                 f'{target}{d_randnetw[randnetw]}.csv'), index=False)

#### STATISTICS FUNCTIONS #####
def compare_CV_splits_nit(source_models,
                          target,
                          save,
                          save_str='',
                          roi=None,
                          models1=['Kell2018word', 'Kell2018speaker', 'Kell2018multitask'],
                          models2=['Kell2018audioset', 'Kell2018music'],
                          aggregation='CV-splits-nit-10',
                          include_spectemp=True,
                          randnetw='False',
                          value_of_interest='median_r2_test',
                          bootstrap=True):
    """Obtain best layer r2 test scores for the components of each model (these were independently obtained across CV-splits-nit-10,
    meaning that there is a value per iteration which is the median of 5 r2 values of the held-out CV splits)
    
    If bootstrap is True, then model1 vs model2 is compared using bootstrap. Else: use parametric tests.

    Intended for use with components (where we don't have individual subjects to bootstrap over).
    """
    source_models_copy = copy.deepcopy(source_models)
    models1_copy = copy.deepcopy(models1)
    models2_copy = copy.deepcopy(models2)
    if include_spectemp:
        source_models_copy.append('spectemp')
        models1_copy.append('spectemp')
        models2_copy.append('spectemp')
        
    df_lst = []
    for source_model in source_models_copy:
        if target == 'NH2015comp': # saved with comp specific name
            load_str = f'best-layer-{aggregation}_per-comp_{source_model}{d_randnetw[randnetw]}_{target}_{value_of_interest}_stats.csv'
        else:
            load_str = f'best-layer_{aggregation}_roi-{roi}_{source_model}{d_randnetw[randnetw]}_{target}_{value_of_interest}_stats.csv'
        df = pd.read_csv(
            join(RESULTDIR_ROOT, source_model, 'outputs',load_str)). \
            rename(columns={'Unnamed: 0': 'comp'})
        df_lst.append(df)
    
    df_all = pd.concat(df_lst)
    
    lst_df_stat = []
    for model1 in tqdm(models1_copy):
        for model2 in models2_copy:
            if model1 == model2:
                continue
                
            if target == 'NH2015comp': # Run stats for each component
                for comp_of_interest in df_all.comp.unique():
                    if bootstrap:
                        df_stat = pairwise_model_comparison_comp_boostrap(df_all, model1, model2, comp_of_interest,
                                                                          value_of_interest=value_of_interest, n_bootstrap=10000)
                        print(f'\n{model1} vs {model2} - {comp_of_interest} - {value_of_interest} - bootstrap test!\n')
                    else:
                        df_stat = pairwise_model_comparison_comp(df_all, model1, model2, comp_of_interest,
                                                                 value_of_interest=value_of_interest)
                        print(f'\n{model1} vs {model2} - {comp_of_interest} - {value_of_interest} - various parametric tests!\n')
        
                    lst_df_stat.append(df_stat)
                    
            elif target == 'NH2015' or target == 'B2021':
                if bootstrap:
                    df_stat = pairwise_model_comparison_boostrap(df_all, model1, model2, roi=roi,
                                                                 value_of_interest=value_of_interest, n_bootstrap=10000)
                    lst_df_stat.append(df_stat)
            else:
                raise ValueError('target must be NH2015, B2021 or NH2015comp')
                    
    
    # Additional logging
    df_stat_all = pd.concat(lst_df_stat)
    df_stat_all['value_of_interest'] = value_of_interest
    df_stat_all['target'] = target
    df_stat_all['aggregation'] = aggregation
    df_stat_all['randnetw'] = randnetw
    df_stat_all['save_str'] = save_str
    df_stat_all['bootstrap'] = bootstrap
    df_stat_all['datetag'] = datetag
    
    if save:
        df_stat_all.to_csv(join(STATSDIR_CENTRALIZED, f'{save_str}_'
                                                      f'{aggregation}_'
                                                      f'{target}_'
                                                      f'{value_of_interest}{d_randnetw[randnetw]}_stats.csv'), index=False)

    
    
    
def pairwise_model_comparison_comp(df_all,
                                   model1,
                                   model2,
                                   comp_of_interest,
                                   value_of_interest='median_r2_test'):
    """Compare the best layer r2 test scores for the components of two models.
    The values are obtained across iterations.
    
    Return mean of the two models' values of interest, as well as statistics

    Intended for use with components.
    """
    
    model1_val = df_all.query(f'comp == "{comp_of_interest}" & source_model == "{model1}"')[value_of_interest].values
    model2_val = df_all.query(f'comp == "{comp_of_interest}" & source_model == "{model2}"')[value_of_interest].values

    # Run simple t-test on the value of interest for model1 vs model2
    stat_ttest, p_ttest = stats.ttest_rel(model1_val, model2_val)
    
    # Sign test
    ds.sign_test(model1_val, model2_val)
    stat_sign, p_sign = ds.sign_test(model1_val - model2_val)
    
    # Wilcoxon test
    stat_wilcoxon, p_wilcoxon = wilcoxon(model1_val, model2_val)
    
    # Package into a df
    df_stats = pd.DataFrame({'model1': model1,
                             'model2': model2,
                             'model1_mean_across_it': np.mean(model1_val),
                             'model2_mean_across_it': np.mean(model2_val),
                             'comp': comp_of_interest,
                             'p_ttest': p_ttest,
                             'stat_ttest': stat_ttest,
                             'p_sign': p_sign,
                             'stat_sign': stat_sign,
                             'p_wilcoxon': p_wilcoxon,
                             'stat_wilcoxon': stat_wilcoxon},index=[0])
    
    return df_stats

def pairwise_model_comparison_comp_boostrap(df_all,
                                            model1,
                                            model2,
                                            comp_of_interest,
                                            value_of_interest='median_r2_test',
                                            n_bootstrap=10000,
                                            show_distrib=False):
    """Compare the best layer r2 test scores for the components of two models.
    The values are obtained across iterations.

    Generate the true delta between the means of model1 and model2 scores.
    Then generate permuted deltas from the mean of the two models' scores shuffled.
    
    Intended for use for components.
    """
    np.random.seed(0) # set seed for reproducibility

    model1_val = df_all.query(f'comp == "{comp_of_interest}" & source_model == "{model1}"')[
        value_of_interest].values
    model2_val = df_all.query(f'comp == "{comp_of_interest}" & source_model == "{model2}"')[
        value_of_interest].values
    
    true_delta = np.mean(model1_val) - np.mean(model2_val)

    # If true_delta is nan, we are probably quering a wrong value
    if np.isnan(true_delta):
        raise ValueError(f'true_delta is nan for component {comp_of_interest} between {model1} and {model2}')
    
    # Run bootstrap on the value of interest for model1 vs model2 by shuffling the values
    permuted_deltas = []
    for i in range(n_bootstrap):
        # shuffle the model1 and model2 values and assign randomly to two lists
        model1_model2_val = np.concatenate([model1_val, model2_val])
        model1_model2_val_copy = copy.deepcopy(model1_model2_val)
        np.random.shuffle(model1_model2_val)
        # Ensure that the shuffled values are not identical to the original values
        if np.array_equal(model1_model2_val, model1_model2_val_copy):
            raise ValueError('The shuffled values are identical to the original values!')
        model1_val_shuffled = model1_model2_val[:len(model1_val)]
        model2_val_shuffled = model1_model2_val[len(model1_val):]
        
        # calculate the difference between the shuffled values
        permuted_deltas.append(np.mean(model1_val_shuffled) - np.mean(model2_val_shuffled))
        
    if show_distrib:
        plt.hist(permuted_deltas)
        plt.axvline(x=true_delta, color='red')
        plt.title(f'{comp_of_interest}: {model1} vs {model2}')
        plt.show()
        
    # Get p-value
    p_value = np.sum(np.array(permuted_deltas) > true_delta) / n_bootstrap

    # Package into a df
    df_stats = pd.DataFrame({'model1': model1,
                             'model2': model2,
                             'model1_mean_across_it': np.mean(model1_val),
                             'model2_mean_across_it': np.mean(model2_val),
                             'comp': comp_of_interest,
                             'true_delta_model1_model2': true_delta,
                             'bootstrap_delta_mean_distrib': np.mean(permuted_deltas),
                             'p_value': p_value,
                             'n_bootstrap': n_bootstrap}, index=[0])

    return df_stats


def pairwise_model_comparison_boostrap(df_all,
                                        model1,
                                        model2,
                                        roi=None,
                                        value_of_interest='median_r2_test',
                                        n_bootstrap=10000,
                                        show_distrib=False,
                                        aggregate_across_subjects=False):
    """Compare the best layer r2 test scores of two models. The loaded file has a value per subject. To make it as comparable
     as possible to the component approach (comparing 10 vals vs. 10 vals (n_it)), we average across subjects.
     That means that per model, we have 10 values, meaned across subjects, per iteration.

    The values are obtained across iterations.

    Generate the true delta between the means of model1 and model2 scores.
    Then generate permuted deltas from the mean of the two models' scores shuffled.

    Intended for use for neural data.
    """
    if aggregate_across_subjects:
        model1_val = df_all.query(f'source_model == "{model1}"').groupby('nit').mean()[value_of_interest].values # get mean across subjects for iterations
        model2_val = df_all.query(f'source_model == "{model2}"').groupby('nit').mean()[value_of_interest].values
    else: # use the values to sample from using individual subjects
        model1_val = df_all.query(f'source_model == "{model1}"')[value_of_interest].values
        model2_val = df_all.query(f'source_model == "{model2}"')[value_of_interest].values
    
    true_delta = np.mean(model1_val) - np.mean(model2_val)
    
    # Run bootstrap on the value of interest for model1 vs model2 by shuffling the values
    permuted_deltas = []
    for i in range(n_bootstrap):
        # shuffle the model1 and model2 values and assign randomly to two lists
        model1_model2_val = np.concatenate([model1_val, model2_val])
        print(f'SET SEED')
        np.random.shuffle(model1_model2_val)
        model1_val_shuffled = model1_model2_val[:len(model1_val)]
        model2_val_shuffled = model1_model2_val[len(model1_val):]
        
        # calculate the difference between the shuffled values
        permuted_deltas.append(np.mean(model1_val_shuffled) - np.mean(model2_val_shuffled))
    
    if show_distrib:
        plt.hist(permuted_deltas, bins=int(n_bootstrap/50))
        plt.axvline(x=true_delta, color='red')
        plt.title(f'{model1} vs {model2}')
        plt.show()
    
    # Get p-value
    p_value = np.sum(np.array(permuted_deltas) > true_delta) / n_bootstrap
    
    # Package into a df
    df_stats = pd.DataFrame({'model1': model1,
                             'model2': model2,
                             'model1_mean_across_it': np.mean(model1_val),
                             'model2_mean_across_it': np.mean(model2_val),
                             'roi': roi,
                             'true_delta_model1_model2': true_delta,
                             'bootstrap_delta_mean_distrib': np.mean(permuted_deltas),
                             'p_value': p_value,
                             'n_bootstrap': n_bootstrap}, index=[0])
    
    return df_stats

def compare_models_subject_bootstrap(source_models,
                                    target,
                                    save,
                                    save_str='',
                                    roi=None,
                                    models1=['Kell2018word', 'Kell2018speaker', 'Kell2018multitask'],
                                    models2=['Kell2018audioset', 'Kell2018music'],
                                    aggregation='CV-splits-nit-10',
                                    randnetw='False',
                                    value_of_interest='median_r2_test',
                                    include_spectemp=True,):
    """
    Load the aggregated values that are plotted in the barplot across models. This has number of subjects as rows,
    and the model score in the columns (along with metadata).
    If spectemp is true, include it.

    """
    df_lst = []
    if include_spectemp:
        source_models.append('spectemp')
        models1.append('spectemp')
        models2.append('spectemp')
    
    for source_model in source_models:
        if aggregation.startswith('CV'):
            load_str = f'best-layer_{aggregation}_' \
                       f'roi-{roi}_' \
                       f'{source_model}{d_randnetw[randnetw]}_' \
                       f'{target}_{value_of_interest}.csv'
        else:
            raise ValueError(f'aggregation {aggregation} not recognized')
        
        df = pd.read_csv(
            join(RESULTDIR_ROOT, source_model, 'outputs',
                 load_str))
        df.rename(columns={'Unnamed: 0': 'subj_idx'}, inplace=True)
        # Sometimes subj_idx already exists, so if it is repeated, drop it
        df = df.loc[:, ~df.columns.duplicated()]
        
        df_lst.append(df)
    
    df_all = pd.concat(df_lst)

    lst_df_stat = []
    for model1 in tqdm(models1):
        for model2 in models2:
            if model1 == model2:
                continue

            df_stat = pairwise_model_comparison_subject_boostrap(df_all=df_all,
                                                                 model1=model1, model2=model2,
                                                                 roi=roi,
                                                                 value_of_interest=value_of_interest,
                                                                 n_bootstrap=10000)
            lst_df_stat.append(df_stat)
            
    # Additional logging
    df_stat_all = pd.concat(lst_df_stat)
    df_stat_all['value_of_interest'] = value_of_interest
    df_stat_all['target'] = target
    df_stat_all['aggregation'] = aggregation
    df_stat_all['randnetw'] = randnetw
    df_stat_all['save_str'] = save_str
    df_stat_all['datetag'] = datetag

    if save:
        df_stat_all.to_csv(join(STATSDIR_CENTRALIZED,
                                f'{save_str}_'
                                f'{aggregation}_'
                                f'{target}_'
                                f'{value_of_interest}{d_randnetw[randnetw]}_stats.csv'),
                                index=False)


def pairwise_model_comparison_subject_boostrap(df_all,
                                               model1,
                                               model2, roi=None,
                                               value_of_interest='median_r2_test',
                                               n_bootstrap=10000,
                                               show_distrib=False,):
    """Compare the best layer r2 test scores of two models. The loaded file (per model) has a value per subject.
     That means that per model, we have e.g., 8 values (if CVsplits-nit approach, these are meaned across iterations.)

    generate distribution with replacement from model2's e.g. 8 subject values -->
    take mean over sampled values for each iteration --> get distribution and compare to true value from model1.

    Intended for use for neural data.
    """
    np.random.seed(0) # Set seed for reproducibility

    model1_val = df_all.query(f'source_model == "{model1}"')[value_of_interest].values
    model2_val = df_all.query(f'source_model == "{model2}"')[value_of_interest].values
    
    # generate distribution with replacement from model2 -->
    # take mean over sampled values for each iteration --> get distribution and compare to true value from model1
    model2_boostrapped_distrib = []
    for i in range(n_bootstrap):
        model2_val_sample = np.random.choice(model2_val, size=len(model2_val)) # sample with replacement
        model2_boostrapped_distrib.append(np.mean(model2_val_sample))
        
    model1_mean_val = np.mean(model1_val)
    
    if show_distrib:
        plt.hist(model2_boostrapped_distrib, bins=int(n_bootstrap / 100))
        plt.axvline(x=model1_mean_val, color='red')
        plt.title(f'{model1} (true value, red line) vs {model2} (distrib)', fontsize=10)
        plt.show()
    
    # Get p-value
    p_value = np.sum(np.array(model2_boostrapped_distrib) > model1_mean_val) / n_bootstrap
    
    # Package into a df
    df_stats = pd.DataFrame({'model1': model1,
                             'model2': model2,
                             'model1_mean_across_subj': np.mean(model1_val),
                             'model2_mean_across_subj': np.mean(model2_val),
                             'roi': roi,
                             'model2_bootstrap_distrib_mean': np.mean(model2_boostrapped_distrib),
                             'p_value': p_value,
                             'n_bootstrap': n_bootstrap}, index=[0])
    
    return df_stats


#### MISC DATAFRAME FUNCTIONS ####
def repeat_row_in_df(output_grouped, rowname, reps_layers):
    reps = [len(reps_layers[rowname]) if val == rowname else 1 for val in output_grouped.index]
    output_reindex = output_grouped.loc[np.repeat(output_grouped.index.values, reps)]
    output_reindex1 = output_reindex.loc[output_reindex.index == rowname]  # only keep rowname rows, drop others
    output_reindex1.index = reps_layers[rowname]
    
    return output_reindex1


def add_identity(axes, *line_args, **line_kwargs):
    """From https://stackoverflow.com/questions/22104256/does-matplotlib-have-a-function-for-drawing-diagonal-lines-in-axis-coordinates """
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


def add_one_hot_roi_col(df,
                        col='roi_label_general',):
    """Create columns with the unique values in col (default "roi_label_general") with 1 if the value is in the row and 0 if not

    :param df: dataframe
    :param col: column to create one hot columns from
    """

    vals = df[col].unique()
    # If not nan, create a new col with that name and 1 if the value is in the row and 0 if not
    for val in vals:
        if not pd.isna(val):
            df[val] = np.where(df[col] == val, 1, 0)

    return df