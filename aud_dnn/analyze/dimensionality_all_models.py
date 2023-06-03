"""
Computes pca on the activations to the 165 sound set for all models 
and computes the effective dimensionality. 
"""
import sys
sys.path.append('/om2/user/jfeather/projects/auditory_brain_dnn')
print(sys.path)

import numpy as np
import torch as ch
from pathlib import Path
import pickle
import matplotlib.pylab as plt
import argparse
import h5py
import os
from aud_dnn.resources import d_layer_reindex, d_sound_category_colors, sound_category_order, d_model_colors, d_model_names, source_layer_map
from sklearn.preprocessing import StandardScaler
import sys # TODO: this is a bit hacky TODO: remove
sys.path.append('../../') # TODO: remove
from aud_dnn.utils import get_source_features

DATADIR = (Path(os.getcwd()) / '..' / 'data').resolve()
RESULTDIR = (Path(os.getcwd()) / '..' / 'results').resolve()
CACHEDIR = (Path(os.getcwd()) / '..' / 'model_actv').resolve().as_posix()

# TODO: move this into utils?
## Stimuli (original indexing, activations are extracted in this order) ##
sound_meta = np.load(os.path.join(
    DATADIR, f'neural/NH2015/neural_stim_meta.npy'))

# Extract wav names in order (this is how the neural data is ordered --> order all activations in the same way)
stimuli_IDs = []
for i in sound_meta:
    stimuli_IDs.append(i[0][:-4].decode("utf-8")) # remove .wav

def run_pca_ed_on_all_models_and_all_layers(randnetw='False',
                                         save_name_base=None,
                                         mean_subtract=False,
                                         std_divide=False,
                                         ):
    """
    Computes pca on the activations to the 165 sound set for all models
    and computes the effective dimensionality from the eigenvalues.
    """
    effective_dim_dict = {}
    for model, layers in d_layer_reindex.items():
        if (model == 'spectemp') and randnetw=='True':
            continue # No random network for spectemp
        effective_dim_dict[model] = {'all_eff_dim_layer_list':[]}
        for layer in layers:
            eff_dim = run_pca_ed_on_saved_stat_dictionary(model, layer, 
                                                       randnetw=randnetw, 
                                                       mean_subtract=mean_subtract, 
                                                       std_divide=std_divide,
                                                       )
            effective_dim_dict[model]['all_eff_dim_layer_list'].append(eff_dim)
            effective_dim_dict[model][layer] = eff_dim

    if save_name_base is not None:
        if not os.path.isdir(save_name_base):
            os.mkdir(save_name_base)

        if randnetw=='False':
            save_pckl_name = 'all_trained_models_'
        else:
            save_pckl_name = 'all_random_models_'

        if mean_subtract:
            if std_divide:
                save_pckl_name+='mean_subtract_std_divide_pca_'
            else:
                save_pckl_name+='mean_subtract_pca_'

        save_pckl_name+='effective_dimensionality_165_sounds.pckl'

        save_eff_dim_pckl = os.path.join(save_name_base, save_pckl_name)

        with open(save_eff_dim_pckl, 'wb') as f:
            pickle.dump(effective_dim_dict, f)

    return effective_dim_dict

def run_pca_ed_on_saved_stat_dictionary(model_name, 
                                     model_layer,
                                     randnetw='False',
                                     save_name_base=None,
                                     mean_subtract=False,
                                     std_divide=False,
                                     ):
    """
    Loads the model's activations for the layer, runs PCA, and 
    returns the effective dimensionality.
    """
    A = get_source_features(source_model=model_name,
                            source_layer=model_layer,
                            source_layer_map=source_layer_map,
                            stimuli_IDs=stimuli_IDs,
                            randnetw=randnetw,
                            CACHEDIR=CACHEDIR)

    if mean_subtract:
        scaler_x = StandardScaler(with_std=std_divide).fit(A)
        A = scaler_x.transform(A)
        if save_name_base is not None:
            save_name_base = os.path.join(save_name_base,
                                          'mean_subtract_pca')

    effective_dim = run_pca_ed_on_layer_output(A, model_layer)
    return effective_dim

def run_pca_ed_on_layer_output(features, feature_name):
    """
    Runs PCA on the layer activation matrix and returns 
    the effective dimensionality.
    """
    try:
        A = features
        if (np.isnan(A).sum(axis=1)>0).sum()>0:
            raise ValueError('Found Nans for some of the sounds!')
        S = np.linalg.eigvals(np.cov(A))
        S = np.sort(S)[::-1] 

        dim_eff = effective_dimensionality(S)
        print('Layer %s, Effective Dim %f, Layer Shape %s'%(feature_name, dim_eff, A.shape))

        return dim_eff

    except ValueError:
        print('Unable to parse layer %s, trying as dictionary'%feature_name) 
        return {layer_key: run_pca_ed_on_layer_output(layer_item, feature_name+'_'+layer_key) for layer_key, layer_item in features.items()}

def effective_dimensionality(S):
    """
    Computes the effective dimensionality using the eigenvalues S. 
    """
    numerator = (S.sum())**2
    demoninator = (S**2).sum()
    return numerator/demoninator

def main(raw_args=None):
    #########PARSE THE ARGUMENTS FOR THE FUNCTION#########
    parser = argparse.ArgumentParser(description='Use a saved matrix of statistics values and compute PCA on the statistics.')
    parser.add_argument('-S', '--save_name_base', type=str, default=None, help='base name (typically directory) for saving the effective dimensionality pickle with values for all models')
    parser.add_argument('-R', '--random_networks', type=str, default='False', help="If 'True' computes the effective dimensionality of the random networks")
    parser.add_argument('-M', '--mean_subtract', type=str, default='False', help="If 'True' subtracts the mean across sounds before running PCA")
    parser.add_argument('-D', '--std_divide', type=str, default='False', help="If 'True' divides by the standard deviation across sounds before running PCA")

    args=parser.parse_args(raw_args)

    run_pca_ed_on_all_models_and_all_layers(randnetw=args.random_networks,
                                            save_name_base=args.save_name_base, 
                                            mean_subtract=(args.mean_subtract=='True'),
                                            std_divide=(args.std_divide=='True'), 
                                            )

if __name__ == '__main__':
    main()
