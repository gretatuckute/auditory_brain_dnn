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
from plotting_specs import d_layer_reindex
from sklearn.preprocessing import StandardScaler
import sys # TODO: this is a bit hacky TODO: remove
sys.path.append('../../') # TODO: remove
from aud_dnn.utils import get_source_features

DATADIR = (Path(os.getcwd()) / '../..' / 'data').resolve()
RESULTDIR = (Path(os.getcwd()) / '../..' / 'results').resolve()

# DATADIR = (Path(os.getcwd()) / '..' / 'data').resolve()
# RESULTDIR = (Path(os.getcwd()) / '..' / 'results').resolve()

# TODO: move this into utils? 
## Stimuli (original indexing, activations are extracted in this order) ##
sound_meta = np.load(os.path.join(DATADIR, f'neural/NH2015/neural_stim_meta.npy')) # TODO: set path appropriately... couldn't find the right file in repo.
# sound_meta = np.load('/om/user/jfeather/neural_predictions/natsounddata/neural_stim_meta.npy') # TODO: remove

# Extract wav names in order (this is how the neural data is ordered --> order all activations in the same way)
stimuli_IDs = []
for i in sound_meta:
    stimuli_IDs.append(i[0][:-4].decode("utf-8")) # remove .wav

def run_pca_on_all_models_and_all_layers(randnetw='False', save_name_base=None, save_all_pca_base=None,
                                         mean_subtract=False, std_divide=False, eig_type='cov'): # TODO: change default cachedir
    effective_dim_dict = {}
    for model, layers in d_layer_reindex.items():
        if (model in ['Kell2018init', 'ResNet50init', 'wav2vecpower','spectemp']) and randnetw=='True': # These models don't have random activations saved. 
            continue
        effective_dim_dict[model] = {'all_eff_dim_layer_list':[]}
        for layer in layers:
            eff_dim = run_pca_on_saved_stat_dictionary(model, layer, randnetw=randnetw, save_name_base=save_all_pca_base, mean_subtract=mean_subtract, std_divide=std_divide, eig_type=eig_type)
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

        save_pckl_name+=eig_type + '_'

        save_eff_dim_pckl = os.path.join(save_name_base, save_pckl_name+'effective_dimensionality_165_sounds.pckl')

        with open(save_eff_dim_pckl, 'wb') as f:
            pickle.dump(effective_dim_dict, f)

    return effective_dim_dict

def run_pca_on_saved_stat_dictionary(model_name, model_layer, randnetw='False', save_name_base=None, mean_subtract=False, std_divide=False, eig_type='cov'):
    A=get_source_features(model_name, model_layer, randnetw=randnetw, stimuli_IDs=stimuli_IDs)

    if mean_subtract:
        scaler_x = StandardScaler(with_std=std_divide).fit(A) # fit demeaning transform on train data only
        A = scaler_x.transform(A) # demeans column-wise
        if save_name_base is not None:
            save_name_base = os.path.join(save_name_base, 'mean_subtract_pca')

    effective_dim = run_pca_on_layer_output(A, model_layer, save_name_base, eig_type)
    return effective_dim

def run_pca_on_layer_output(features, feature_name, save_name_base, eig_type):
    try:
        A = ch.tensor(features)

        # Check for nans -- if there is a nan in the texture stats remove the sound.
        A = A[ch.isnan(A).sum(axis=1)==0]

        if (ch.isnan(A).sum(axis=1)>0).sum()>0:
            print('REMOVING SOUND(S)!!! There were nans in row(s) ', (ch.isnan(A).sum(axis=1)>0).nonzero(as_tuple=True))

        if eig_type=='cov':
            A = A.detach().cpu().numpy()
            U = None
            Vh = None
            S = np.linalg.eigvals(np.cov(A))
            S = np.sort(S)[::-1]
        elif eig_type=='corr':
            A = A.detach().cpu().numpy()            
            U = None
            Vh = None
            S = np.linalg.eigvals(np.corrcoef(A))
            S = np.sort(S)[::-1]
        elif eig_type=='svd':
            U, S, Vh = ch.linalg.svd(A, full_matrices=False)
            # Change the singular values to eigenvectors of A.T A 
            S = S**2
            A.detach().cpu().numpy()
            U.detach().cpu().numpy()
            S.detach().cpu().numpy()
            Vh.detach().cpu().numpy()
        else:
            raise ValueError(f'eig_type {eig_type} is not supported')

        dim_eff = effective_dimensionality(S)
        print('Layer %s, Effective Dim %f, Layer Shape %s'%(feature_name, dim_eff, A.shape))

        if save_name_base is not None: 
            if not os.path.isdir(save_name_base):
                os.mkdir(save_name_base)
            save_svd_name = os.path.join(save_name_base, '_'.join(feature_name.split('/')) + '.pckl')
            stuff_to_pickle = {'Vh': Vh,
                           'S':S,
                           'U':U,
                           'effective_dimensionality':dim_eff}
            with open(save_svd_name, 'wb') as f:
                pickle.dump(stuff_to_pickle, f)

        return dim_eff

    except ValueError:
        print('Unable to parse layer %s, trying as dictionary'%feature_name) 
        return {layer_key: run_pca_on_layer_output(layer_item, feature_name+'_'+layer_key, save_name_base, eig_type) for layer_key, layer_item in features.items()}

def effective_dimensionality(S):
    numerator = (S.sum())**2
    demoninator = (S**2).sum()
    return numerator/demoninator

def main(raw_args=None):
    #########PARSE THE ARGUMENTS FOR THE FUNCTION#########
    parser = argparse.ArgumentParser(description='Use a saved matrix of statistics values and compute PCA on the statistics.')
    parser.add_argument('-S', '--save_name_base', type=str, default=None, help='base name (typically directory) for saving the effective dimensionality pickle with values for all models')
    parser.add_argument('-P', '--save_all_pca_base', type=str, default=None, help='base name (typically directory) for saving all of the pca output. If None do not save.')
    parser.add_argument('-E', '--eig_type', type=str, default='svd', help='How to calculate the eigenvalues for the effective dimensionality.')
    parser.add_argument('-R', '--random_networks', type=str, default='False', help="If 'True' computes the effective dimensionality of the random networks")
    parser.add_argument('-M', '--mean_subtract', type=str, default='False', help="If 'True' subtracts the mean across sounds before running PCA")
    parser.add_argument('-D', '--std_divide', type=str, default='False', help="If 'True' divides by the standard deviation across sounds before running PCA")

    args=parser.parse_args(raw_args)

    run_pca_on_all_models_and_all_layers(randnetw=args.random_networks, save_name_base=args.save_name_base, 
                                         save_all_pca_base=args.save_all_pca_base, mean_subtract=(args.mean_subtract=='True'),
                                         std_divide=(args.std_divide=='True'), eig_type=args.eig_type)

if __name__ == '__main__':
    main()
