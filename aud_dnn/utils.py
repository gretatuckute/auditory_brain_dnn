import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.special import softmax
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
import pandas as pd
import numpy as np
import pickle
import copy
import warnings
import os
import timeit
from pathlib import Path
import sklearn
from scipy.linalg import norm
from scipy.io import savemat
from scipy.io import loadmat
from scipy import stats
from pathlib import Path
import argparse
import getpass
import datetime
import sys
import h5py
import xarray as xr

# Set path to current directory
CURRENT_DIR = Path(__file__).parent.absolute()
os.chdir(CURRENT_DIR)


# Set random seed
np.random.seed(0)
random.seed(0)


##### TARGET (NEURAL/COMPONENT) FUNCTIONS #####
def get_target(target,
                stimuli_IDs,
                DATADIR):
    """
    Loads the target data (neural or component) and returns it as a numpy array.

    Parameters
        target (str): name of the target data (options are 'NH2015', 'NH2015comp', 'B2021')
        stimuli_IDs (list): list of stimulus IDs (e.g., ['stim5_alarm_clock', 'stim7_applause'])
        DATADIR (str): path to the data directory

    Returns
        voxel_data (np.array): target data as a numpy array. Rows are stimuli, columns are voxels or components.
    """

    if target == 'NH2015':
        voxel_data_all = np.load(os.path.join(DATADIR, f'neural/{target}/voxel_features_array.npy'))
        voxel_meta_all = np.load(os.path.join(DATADIR, f'neural/{target}/voxel_features_meta.npy'))
        is_3 = voxel_meta_all['n_reps'] == 3  # only get the voxels with 3 repetitions (sessions)
        voxel_meta, voxel_data = voxel_meta_all[is_3], voxel_data_all[:, is_3, :]
        voxel_id = voxel_meta['voxel_id']

    elif target == 'NH2015comp':
        comp_data = loadmat(os.path.join(DATADIR, f'neural/{target}/components.mat'))
        comp_stimuli_IDs = comp_data['stim_names']
        comp_stimuli_IDs = [x[0] for x in comp_stimuli_IDs[0]]

        comp_names = comp_data['component_names']
        comp_names = [x[0][0] for x in comp_names]

        comp = comp_data['R']  # R is the response matrix

        # Create a df with stimuli IDs as index and component names as columns
        df_comp = pd.DataFrame(comp, index=comp_stimuli_IDs, columns=comp_names)

        # Reindex so it is indexed the same way as the neural data and activations (stimuli_IDs)
        voxel_data = df_comp.reindex(stimuli_IDs)  # for consistency, let's call it voxel_data
        voxel_id = np.arange(voxel_data.shape[1]) # 0 through 5 -- the component numbers, we will later assert that they match the metadata

    elif target == 'B2021':
        sound_meta_mat = loadmat(os.path.join(DATADIR, f'neural/{target}/stim_info_v4'))['stim_info']

        stimuli_IDs_B2021 = []
        for i in sound_meta_mat['stim_names'][0][0]:
            stimuli_IDs_B2021.append((i[0][0].astype(str)))

        assert (stimuli_IDs == stimuli_IDs_B2021[:165])  # same order as NH2015

        voxel_data = np.load(os.path.join(DATADIR, f'neural/{target}/voxel_features_array.npy'))
        voxel_id = np.load(os.path.join(DATADIR, f'neural/{target}/voxel_features_meta.npy')) # contains the voxel_id as the only field

        # Truncate to only run the first 165 sound set
        voxel_data = voxel_data[:len(stimuli_IDs), :, :]

    else:
        raise LookupError(f'Target dataset of interest {target} not found!')

    return voxel_data, voxel_id


##### SOURCE MODEL ACTIVATIONS FUNCTIONS #####

class PytorchWrapper:

    def __init__(self,
                 model_identifier,
                 CACHEDIR='model-actv-control/',
                 randnetw='False'):
        self.model_identifier = model_identifier
        self.randnetw = randnetw
        self.CACHEDIR = CACHEDIR

        """
        Wrapper class for loading pkl files with DNN model activations as dictionaries.
        
        The pkl files have to live in /CACHE_DIR/model_identifier/ and have to be named as follows:
            {ID}_activations.pkl, e.g.: stim5_alarm_clock_activations.pkl
        If randnetw is "True", the pkl files have to be named as follows:
            {ID}_activations_randnetw.pkl, e.g.: stim5_alarm_clock_activations_randnetw.pkl 
            (will load activations from permuted network)
        """
    
    def compile_activations(self,
                            ID_order,
                            source_layer_of_interest,):
        """
        Compile activations from a folder that contains dictionaries of activations (key=string name, value=1D ndarray with activations)
        The naming convention of each pkl file is: {ID}_activations.pkl, e.g.: stim5_alarm_clock_activations.pkl
        If it is permuted, the naming convention is: {ID}_activations_randnetw.pkl, e.g.: stim5_alarm_clock_activations_randnetw.pkl

        Parameters
            ID_order (list): list of stimulus IDs (e.g., ['stim5_alarm_clock', 'stim7_applause'])
            source_layer_of_interest (str): name of the source layer of interest (e.g., 'conv1')
        """

        # Load activations
        ACTVDIR = os.path.join(self.CACHEDIR, self.model_identifier)
        # Each single pkl file contains the activations for one stimulus (across layers)
        activation_files = [f for f in os.listdir(ACTVDIR) if os.path.isfile(os.path.join(ACTVDIR, f))]
        activation_files = [f for f in activation_files if not f.startswith('.')]

        if self.randnetw == 'True':
            activation_files = [f for f in activation_files if f.endswith('randnetw.pkl')]
            print('Loading permuted activations')
        else:
            activation_files = [f for f in activation_files if f.endswith('activations.pkl')]
        print(activation_files)
        
        if not os.path.exists(ACTVDIR):
            warnings.warn(
                f'Cache does not exist! \nModel identifier passed in: {self.model_identifier} \n')
            return
        
        assert (len(ID_order) == len(activation_files))
        
        actv_dict_all = {}  # key: stimulus ID, value: activations array
        for file in activation_files:
            ID = file.split('_activations')[0]
            activations_file = os.path.join(self.CACHEDIR, self.model_identifier, f'{file}')
            
            with open(activations_file, 'rb') as f:
                actv_dict = pickle.load(f)
            
            # print min values
            for k, v in actv_dict.items():
                print(f'{k}: min: {v.min()}')

            # for k, v in actv_dict.items():
            #     print(f'{k}: shape: {v.shape}')
            
            # extract layer
            actv_layer = actv_dict[source_layer_of_interest]
            
            # store
            actv_dict_all[ID] = actv_layer
        
        df_actv = pd.DataFrame.from_dict(actv_dict_all, orient='index')
        ID_order = [f'{f}' for f in ID_order]
        df_actv = df_actv.reindex(ID_order)
        
        self.activations = df_actv
        self.source_layer = source_layer_of_interest

        # self.plot_activations_across_sounds(df_actv,vmax=0.05)
        
        return df_actv
    
    def plot_activations_across_sounds(self, df_actv, vmin=0, vmax=0.05):
        """Plot activations across sounds"""
        plt.imshow(df_actv, aspect='auto', vmin=vmin, vmax=vmax, interpolation='none')
        plt.title(f'{self.source_layer}')
        plt.xlabel('Units')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    
    def PCA_activations(self, n_components=3):
        """Compute PCA of activations"""
        actv_np = self.activations.values
        pca = PCA(n_components=n_components)
        
        PCs = pca.fit_transform(actv_np)  # transform same input data into the new basis. Sklearn mean centers.
        explained_var = pca.explained_variance_ratio_
        
        self.activations_PCs = PCs
        self.PCs_explained_var = explained_var


def get_source_features(source_model,
                        source_layer,
                        source_layer_map,
                        randnetw,
                        stimuli_IDs,
                        CACHEDIR):
    """
    Gets the activations for a given DNN source model and layer.
    Handles the in-house and external models appropriately.

    Parameters
        source_model (str): name of the source model (e.g., 'VGGish')
        source_layer (str): name of the source layer (e.g., 'conv1')
        source_layer_map (dict): dictionary with keys as the name of the source model and value a dictionary with the
            keys as the 'pretty', interpretable name of the layer, and the value of the layer in Pytorch (see resources.py)
        randnetw (str): whether to load activations from permuted network.
        stimuli_IDs (list): list of stimulus IDs (e.g., ['stim5_alarm_clock', 'stim7_applause']). Only used for external models.
        CACHEDIR (str): path to the directory where the activations are stored

    Returns
        source_features (ndarray): activations of the source layer
    """
    d_randnetw = {'True': '_randnetw',
                  'False': ''}

    ## External models. Has to be loaded using the PytorchWrapper class.
    if source_model in ['AST', 'ASTSL01', 'ASTSL10', 'ASTSL01-from-datadir',
                        'DCASE2020',
                        'DS2',
                        'metricGAN', 'metricGANSL01', 'metricGANSL10',
                        'sepformer', 'sepformerSL01', 'sepformerSL10',
                        'S2T',
                        'VGGish', 'VGGishSL01', 'VGGishSL10', 'VGGishSL100',
                        'wav2vec', 'wav2vecpower',
                        'ZeroSpeech2020', ]:
        model = PytorchWrapper(model_identifier=source_model, CACHEDIR=CACHEDIR, randnetw=randnetw)
        source_features = model.compile_activations(ID_order=stimuli_IDs, source_layer_of_interest=source_layer_map[source_model][source_layer])
        source_features = source_features.to_numpy()

    ## In-house models. Activations are already compiled across stimuli (sounds).
    elif source_model.startswith('ResNet50') or source_model.startswith('Kell2018') or source_model.startswith('spectemp'):
        filename = CACHEDIR + f'/{source_model}/natsound_activations{d_randnetw[randnetw]}.h5'

        h5file = h5py.File(filename, 'r')
        layers = h5file['layer_list']
        layer_lst = []
        for l in layers:
            layer_lst.append(l.decode('utf-8'))

        # take care of different naming of 'final' layer (not used for the paper, because we do not include the final layers
        # in the in-house models to ensure similar number of layers between single-task and multitask models)
        if source_layer == 'final' and source_model.endswith('speaker'):
            source_features = np.asarray(h5file['final/signal/speaker_int'])
        elif source_layer == 'final' and source_model.endswith('audioset'):
            source_features = np.asarray(h5file['final/noise/labels_binary_via_int'])
        # take care of resnet50music having one layer named differently than other resnets
        elif source_layer == 'conv1_relu1' and source_model == ('ResNet50music'):
            source_features = np.asarray(h5file['relu1']) # named relu1 and not conv1_relu1. taken care of, such that going forward all resnets have the same layer names.
        # take care of the multitask networks having 3 final layers
        elif source_layer == 'final_word' and source_model.endswith('multitask'):
            source_features = np.asarray(h5file['final/signal/word_int'])
        elif source_layer == 'final_speaker' and source_model.endswith('multitask'):
            source_features = np.asarray(h5file['final/signal/speaker_int'])
        elif source_layer == 'final_audioset' and source_model.endswith('multitask'):
            source_features = np.asarray(h5file['final/noise/labels_binary_via_int'])
        else:
            source_features = np.asarray(h5file[source_layer])

    else:
        print(f'Source model {source_model} does not exist yet!')
        raise LookupError()

    print(f'Shape of layer {source_layer}: {np.shape(source_features)} and min value is {np.min(source_features)}')

    return source_features


##### MAPPING / REGRESSION FUNCTIONS #####

def ridgeRegressSplits(source_features,
                       y,
                       is_train_data,
                       is_test_data,
                       possible_alphas,
                       voxel_idx=None,
                       split_idx=None,
                       ridge_normalize=False,
                       ridge_fit_intercept=True,
                       store_cv=False,
                       verbose=True,
                       save_plot=False,
                       demean_x=True,
                       demean_y=True):
    """
    Performs CV ridge regression, using a specified set of x and y.

    Parameters:
        source_features: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples, )
        is_train_data: numpy array of shape (n_samples, ) with True/False indicating whether the sample is in the training set
        is_test_data: numpy array of shape (n_samples, ) with True/False indicating whether the sample is in the test set
        possible_alphas: list of alphas to try for the ridge regression
        voxel_idx: index of the voxel used for the regression (for printing purposes)
        split_idx: index of which CV split is being run (only for printing purposes)
        ridge_normalize: whether to normalize using scikit learn's default RidgeCV function
        ridge_fit_intercept: whether to fit an intercept for the regression
        store_cv: whether to store the cross-validation scores for each alpha (using scikit learn's default RidgeCV function)
        verbose: whether to print statements
        save_plot: whether to save the plot of predicted y vs actual y
        demean_x: whether to demean the source features (X) before regression (based on the train set)
        demean_y: whether to demean the y (based on the train set)

    Returns:
            r_prior_zero_manipulation: R prior to setting values to 0 if std(Y_pred_test)=0 or R<0 (on the test set)
            r: R after setting values to 0 if std(Y_pred_test)=0 or R<0 (on the test set)
            r2: R^2 (on the test set)
            r2_train: R^2 of Y_train and Y_train_pred
            y_pred_test: Y_pred of the test set
            y_pred_test_rescaled: Y_pred of the test set, rescaled to the original range of Y (i.e., not demeaned using the
                mean of the train set). Note that the regression still uses demeaned Y if demean_y=True, but we rescale the
                predictions to the original range of Y post-hoc using scaler_y.
            alpha: Regularization parameter
            y_pred_test_std: Std of y_pred_test
            y_pred_train_std: Std of y_pred_train
            Y_test_std: Std of y_test (neural)
            Y_train_std: Std of y_train (neural)
            warning_constant_flag: 1 if a constant warning occurred, 0 otherwise
            warning_alpha_limit: 1 if a warning occurred for upper alpha bound, 2 if lower bound, 0 otherwise
    
    Base function from Jenelle Feather.
    """
    ## Define regression ##
    ridgeCrossVal = RidgeCV(alphas=possible_alphas, normalize=ridge_normalize,
                                                     fit_intercept=ridge_fit_intercept,
                                                     store_cv_values=store_cv)
    
    ## Define train/test splits ##
    X_train = source_features[is_train_data, :]
    Y_train = y[is_train_data]

    Y_train_std = np.std(Y_train)
    
    X_test = source_features[is_test_data, :]
    Y_test = y[is_test_data]

    Y_test_std = np.std(Y_test)
    
    ## Demean train splits ##
    if demean_x:
        scaler_x = StandardScaler(with_std=False).fit(X_train) # fit demeaning transform on train data only
        X_train = scaler_x.transform(X_train) # demeans column-wise
    
    if demean_y:
        scaler_y = StandardScaler(with_std=False).fit(Y_train) # fit demeaning transform on train data only
        Y_train = scaler_y.transform(Y_train)
        
    ## Fit model ##
    ridgeCrossVal.fit(X_train, Y_train)
    alpha = ridgeCrossVal.alpha_ # View Best Model’s Alpha Value
    
    ## Demean test splits ##
    if demean_y:
        Y_test = scaler_y.transform(Y_test) # Use same demeaning transform as estimated from the training set
    
    if demean_x:
        X_test = scaler_x.transform(X_test)
    
    ## Get training R (set to 0 if std of y pred train is 0 and if r train is negative) ##
    y_pred_train = ridgeCrossVal.predict(X_train)
    y_pred_train_std = np.std(y_pred_train)
    if not y_pred_train_std == 0:
        r_train = np.corrcoef(y_pred_train.ravel(), Y_train.ravel())[1,0]
        if r_train < 0:
            r_train = 0
    else:
        r_train = 0
        
    r2_train = r_train ** 2

    ## Predict test set ##
    y_pred_test = ridgeCrossVal.predict(X_test)
    y_pred_test_std = np.std(y_pred_test)

    ## Obtain a y_pred where the demean transformation is taken out again (to make the magnitude of values comparable to the original y) ##
    if demean_y:
        y_pred_test_rescaled = y_pred_test + scaler_y.mean_
    else:
        y_pred_test_rescaled = y_pred_test
    
    ## Look into CV splits to find best alpha value ##
    if store_cv:
        # additional storing / testing
        ridgeCV_vals = ridgeCrossVal.cv_values_.squeeze()
        ridgeCV_vals_meaned = np.mean(ridgeCV_vals, 0) # mean across splits (len of train)
        best_alpha_index = np.argmin(ridgeCV_vals_meaned) # we want the lowest mean squared error
        best_alpha = possible_alphas[best_alpha_index]

        print(f'Best CV ridge negative mean squared error: {np.min(ridgeCV_vals_meaned):3} with argmin {best_alpha_index} and alpha {best_alpha}')
        assert(alpha == best_alpha)
    
    ## Check the alpha boundary hits ##
    if verbose:
        if (alpha == possible_alphas[0]) or (alpha == possible_alphas[-1]):
            print(f'WARNING: BEST ALPHA {alpha} IS AT THE EDGE OF THE SEARCH SPACE FOR VOXEL {voxel_idx}, SPLIT {split_idx}')
    
    warning_alpha_limit = 0
    if (alpha == possible_alphas[0]):
        warning_alpha_limit = 1 # set to 1 if upper hit
    if (alpha == possible_alphas[-1]):
        warning_alpha_limit = 2 # set to 2 if lower hit
    
    ## Look into constant predictions ##
    warning_constant_flag = 0
    if not np.std(y_pred_test) == 0: # if the std of predicted responses is not zero
        r = np.corrcoef(y_pred_test.ravel(), Y_test.ravel())[1,0]
        r_prior_zero_manipulation = copy.deepcopy(r)

        if r < 0:
            r = 0  # ir negative r value, set to 0.
            
        if norm(y_pred_test - np.mean(y_pred_test)) < 1e-13 * abs(np.mean(y_pred_test)): # constant prediction as defined by scikit pearsonr
            warning_constant_flag = 1 # warn and track, but leave the r-value as is
            if verbose:
                print(f'Nearly constant array for voxel {voxel_idx}') # tested for the y test array, does not happen.
            
    else: # if the std of predicted responses is zero
        if verbose:
            print(f'WARNING: VOXEL {voxel_idx} is predicted as only the expected value. Setting correlation to zero.')
        r = np.corrcoef(y_pred_test.ravel(), Y_test.ravel())[1,0] # compute because of storing the r prior manipulation
        r_prior_zero_manipulation = copy.deepcopy(r)
        
        r = 0
        warning_constant_flag = 1 # set warning flag if std = 0
    
    r2 = r ** 2
    
    if voxel_idx % 500 == 1:
        if save_plot:  # plot y hat versus y true versus mean
            diagnostic_plot_during_regression(y_pred_test, Y_test, y_pred_train, Y_train,
                                              save_plot, warning_constant_flag, voxel_idx, split_idx, alpha, r2, r2_train)


    return r_prior_zero_manipulation, r, r2, \
           r2_train, \
           y_pred_test, y_pred_test_rescaled, \
           alpha, \
           y_pred_test_std, y_pred_train_std, Y_test_std, Y_train_std, \
           warning_constant_flag, warning_alpha_limit


def ridgeCV_correctedR2(source_features,
                        voxel_data,
                        voxel_idx,
                        split_idx,
                        is_train_data,
                        is_test_data,
                        possible_alphas,
                        save_plot=False):
    """
    Wrapper for ridge regression with cross-validation and additional voxel-wise noice correction (Spearman-Brown correction).

    Parameters
        source_features: numpy array of shape (n_samples, n_features), source features to use for regression
        voxel_data: numpy array of shape (n_samples, n_voxels, repetition), voxel data to use for regression.
            Repetition refers to the session.
        voxel_idx: int, index of voxel to use for regression.
        split_idx: int, CV split index
        is_train_data: numpy array of shape (n_samples, ) with True/False indicating whether the sample is in the training set
        is_test_data: numpy array of shape (n_samples, ) with True/False indicating whether the sample is in the test set
        possible_alphas: numpy array of shape (n_alphas, ) with possible alphas to use for regression
        save_plot: boolean, whether to save the diagnostic plot

    Returns
        r_prior_zero_manipulation_mean: float, R prior to setting values to 0 if std(Y_pred)=0 or R<0
            (on test set, derived from using the mean of the three neural repetitions)
        r_mean: float, R (on test set, derived from using the mean of the three neural repetitions)
        r2_mean: float, R^2 (on test set, derived from using the mean of the three neural repetitions)
        corrected_r2: float, corrected R^2 (Spearman-Brown correction) (on test set, derived from using the mean of the three neural repetitions,
            but the correction uses the mean of two pairwise neural repetitions)
        r2_train_mean: float, R^2 (on training set, derived from using the mean of the three neural repetitions)
        y_pred_test_mean: numpy array of shape (n_samples_test, ), predicted y values (on test set,
            derived from using the mean of the three neural repetitions)
        y_pred_test_rescaled_mean: numpy array of shape (n_samples_test, ), predicted y values (on test set,
            derived from using the mean of the three neural repetitions) rescaled to the mean of the test set (i.e.
            removing the demeaning transformation that was fitted on the training set)
        alpha: float, alpha used for regression (on test set, derived from using the mean of the three neural repetitions)
        y_pred_test_std_mean: float, standard deviation of predicted y values (on test set, derived from using the mean of the three neural repetitions)
        y_pred_train_std_mean: float, standard deviation of predicted y values (on training set, derived from using the mean of the three neural repetitions)
        Y_test_std_mean: float, standard deviation of y values (on test set, derived from using the mean of the three neural repetitions)
        Y_train_std_mean: float, standard deviation of y values (on training set, derived from using the mean of the three neural repetitions)
        warning_constant_count_mean: int, 1 if a constant warning occurred, 0 otherwise (on test set, derived from using the mean of the three neural repetitions)
        warning_constant_count_splits_sum: int, sum of constant warnings across splits (in the correction phase, using pairwise repetitions)
        warning_alpha_mean: int, 1 if upper alpha limit was hit, 2 if lower alpha limit was hit, 0 otherwise (on test set, derived from using the mean of the three neural repetitions)

    Base function from Jenelle Feather.
    """
    
    ## Run RidgeCV and return uncorrected r2 ##
    # (mean refers to the fact that the model is fitted on the average of neural responses)
    
    r_prior_zero_manipulation_mean, r_mean, r2_mean, \
    r2_train_mean, \
    y_pred_test_mean, y_pred_test_rescaled_mean, \
    alpha_mean, \
    y_pred_test_std_mean, y_pred_train_std_mean, Y_test_std_mean, Y_train_std_mean, \
    warning_constant_count_mean, warning_alpha_mean = ridgeRegressSplits(
                                                        source_features=source_features,
                                                        y=np.mean(voxel_data, 2)[:, voxel_idx][:, None], # mean over reps
                                                        is_train_data=is_train_data,
                                                        is_test_data=is_test_data,
                                                        possible_alphas=possible_alphas,
                                                        voxel_idx=voxel_idx,
                                                        split_idx=split_idx,
                                                        verbose=True,
                                                        save_plot=save_plot)



    ## Get predicted responses (y hat) for models trained on data using just a single repetition ####
    warning_constant_count_splits = [] # store if a warning happened during the estimation of corrected response
    split_y_pred = []
    for split_idx_rep, split in enumerate([0, 1, 2]):
        _, _, _, _, y_pred_test_split, \
        _, _, _, _, _, _, warning_constant_count_split, _ = ridgeRegressSplits(source_features=source_features,
                                             y=voxel_data[:, voxel_idx, split][:, None], # take out one repetition, and fit model on that (last part is to reshape into samples)
                                             is_train_data=is_train_data,
                                             is_test_data=is_test_data,
                                             possible_alphas=possible_alphas,
                                             voxel_idx=voxel_idx,
                                             split_idx=split_idx, # The CV split index is the same for all repetitions
                                             verbose=False, # Do not print out alpha warnings in the rv hat estimation phase
                                             save_plot=False) # Do not save plots within correction estimation phase
        split_y_pred.append(y_pred_test_split)
        warning_constant_count_splits.append(warning_constant_count_split)
        warning_constant_count_splits_sum = np.sum(warning_constant_count_splits)

    
    ## Correlate neural data and predictions across pairs of repetitions (splits) ##
    rvs = []
    rv_hats = []
    for split_idx_rep, split in enumerate([[0, 1], [1, 2], [0, 2]]):
        # corr of neural data
        split_r = np.corrcoef(voxel_data[is_train_data, voxel_idx, split[0]], # compute correlation between e.g. rep 0 responses and rep 1 responses
                                         voxel_data[is_train_data, voxel_idx, split[1]])[1,0]
        rvs.append(split_r)
        
        # corr of predicted responses
        split_r_hat = np.corrcoef(split_y_pred[split[0]].ravel(), split_y_pred[split[1]].ravel())[1,0]
        rv_hats.append(split_r_hat)
    
    ## VOXEL-WISE NOISE CORRECTION ##
    rv_med = np.nanmedian(rvs)
    rv = np.nanmax([3 * rv_med / (1 + 2 * rv_med), 0.182]) # Spearman-Brown correction
    rv_hat_med = np.nanmedian(rv_hats)
    rv_hat = np.nanmax([3 * rv_hat_med / (1 + 2 * rv_hat_med), 0.183]) # Spearman-Brown correction on predicted data
    
    corrected_r2 = r2_mean / (rv * rv_hat) # Test-retest correction

    return r_prior_zero_manipulation_mean, r_mean, r2_mean, corrected_r2, \
           r2_train_mean, \
           y_pred_test_mean, y_pred_test_rescaled_mean, \
           alpha_mean, \
           y_pred_test_std_mean, y_pred_train_std_mean, Y_test_std_mean, Y_train_std_mean, \
           warning_constant_count_mean, warning_constant_count_splits_sum, warning_alpha_mean


def diagnostic_plot_during_regression(y_pred_test, Y_test, y_pred_train, Y_train,
                                      save_plot, warning_constant_flag, voxel_idx, split_idx, alpha, r2, r2_train):
    """Scatter plots of predicted versus actual (neural) response for the test set (plot 1), and the train set (plot 2)
    """
    
    plt.figure()
    plot_save_str_test = f'warning-{warning_constant_flag}_test_voxel-{voxel_idx}_split-{split_idx}.png'
    plt.plot(y_pred_test, label='Y pred')
    plt.plot(Y_test, label='Y test (true)')
    plt.plot([np.mean(Y_test)] * 83, label='Mean of Y test (true)')
    plt.legend()
    plt.xlabel('Number of trials (sounds)')
    try:
        plt.title(f'Test: voxel {voxel_idx}, split {split_idx}, best alpha {alpha:.2e}, r2 {r2:.2}')
    except:
        plt.title(f'Test: voxel {voxel_idx}, split {split_idx}, best alpha {alpha}, r2 {r2}')
    plt.savefig(save_plot / plot_save_str_test)
    plt.show()

    plt.figure()
    plot_save_str_train = f'warning-{warning_constant_flag}_train_voxel-{voxel_idx}_split-{split_idx}.png'
    plt.plot(y_pred_train, label='Y pred train')
    plt.plot(Y_train, label='Y train (true)')
    plt.plot([np.mean(Y_train)] * 83, label='Mean of Y train (true)')
    plt.legend()
    plt.xlabel('Number of trials (sounds)')
    try:
        plt.title(f'Train: voxel {voxel_idx}, split {split_idx}, best alpha {alpha:.2e}, r2 {r2:.2}')
    except:
        plt.title(f'Train: voxel {voxel_idx}, split {split_idx}, best alpha {alpha}, r2 {r2}')
    plt.savefig(save_plot / plot_save_str_train)
    plt.show()


##### MISC FUNCTIONS #####

def find_duplicate(nums):
    """Checks for consecutive duplicate in a list (nums)"""
    duplicates = []
    for i in range(len(nums)-1):
        if nums[i] == nums[i+1]:
            duplicates.append(nums[i])
            # print('found dup')
    if duplicates:
        return True
    else:
        return False

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def convert_num(df, exceptions=['sentence']):
    """
    Convert object columns to numeric

    :param df: dataframe
    :param exceptions: list of string values of column names to not convert
    :return: df with numeric columns
    """
    obj_type = list(df.select_dtypes(include=['object']).columns)
    # print(f'Object types: {obj_type}')
    df_num = df.copy()
    for c in obj_type:
        if c in exceptions:
            pass
        else:
            df_num[c] = pd.to_numeric(df_num[c], errors='coerce')

    return df_num

def pos_neg(x):
    if x < 0:
        y = 0
    else:
        y = 1
    return y

def get_dataset_keys(f):
    """For seeing what is available in H5 file, from:
    https://stackoverflow.com/questions/44883175/how-to-list-all-datasets-in-h5py-file
    """
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys
