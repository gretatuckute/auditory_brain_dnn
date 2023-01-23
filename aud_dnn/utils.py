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
# from model_specs import source_layer_map, d_randnetw # TODO: Do we want these here? They are used for getting the source layer. GT: yes, I previously imported them in AUD_main.py as the source features were fetched there

# Set random seed
np.random.seed(0)
random.seed(0)

##### SOURCE MODEL ACTIVATIONS FUNCTIONS #####

class PytorchWrapper:
    """
    Wrapper class for loading pkl files with model activations as dictionaries (key=string, name, value=1D ndarray with activations)
    """
    
    def __init__(self, model_identifier='DS2', CACHEDIR='model-actv-control/', randnetw='False'):
        self.model_identifier = model_identifier
        self.randnetw = randnetw
        self.CACHEDIR = CACHEDIR
        self.activations = None
        self.activations_PCs = None
        self.PCs_explained_var = None
        self.source_layer = None
    
    def get_source_layer(self, source_layer):
        """Extract layer of interest from dictionary"""
    
    def compile_activations(self, ID_order,
                            source_layer_of_interest='Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5))--0',
                            warning=True):
        # load activations
        ACTVDIR = os.path.join(self.CACHEDIR, self.model_identifier)
        activation_files = [f for f in os.listdir(ACTVDIR) if os.path.isfile(os.path.join(ACTVDIR, f))]
        activation_files = [f for f in activation_files if not f.startswith('.')]

        if self.randnetw == 'True':
            activation_files = [f for f in activation_files if f.endswith('randnetw.pkl')]
            print('Loading permuted activations')
        else:
            activation_files = [f for f in activation_files if f.endswith('activations.pkl')]
        print(activation_files)
        
        if not os.path.exists(ACTVDIR):
            if warning:
                warnings.warn(
                    f'Cache does not exist! \nModel identifier passed in: {self.model_identifier} \n')
                return
        
        assert (len(ID_order) == len(activation_files))
        
        actv_dict_all = {}  # key: ID, value: activation array
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
    
    def plot_conv_vs_relu(self, actv_dict):
        """Plot ReLU versus Conv layer for one sound (one actv_dict). For checking whether they are the same"""
        plt.plot(actv_dict['Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))--0'],
                 label='Conv2d_1')
        plt.plot(actv_dict['ReLU()--0'], label='ReLu_1')
        plt.legend()
        plt.show()
    
        plt.plot(actv_dict['Linear(in_features=4096, out_features=128, bias=True)--0'],
                 label='Linear 3', alpha=0.5)
        plt.plot(actv_dict['ReLU()--8'], label='ReLu_9', alpha=0.5)
        plt.legend()
        plt.show()
    
    def PCA_activations(self, n_components=3):
        """Compute PCA of activations"""
        actv_np = self.activations.values
        pca = PCA(n_components=n_components)
        
        PCs = pca.fit_transform(actv_np)  # transform same input data into the new basis. Sklearn mean centers.
        explained_var = pca.explained_variance_ratio_
        
        self.activations_PCs = PCs
        self.PCs_explained_var = explained_var
    
    def plot_2D_PCA(self):
        """Plot 2D PCA of activations"""
        labels = {
            str(i): f"PC {i + 1} ({var:.1f}%)"
            for i, var in enumerate(self.PCs_explained_var * 100)
        }
        fig = px.scatter_matrix(
            self.activations_PCs,
            labels=labels,
            opacity=0.4,
            dimensions=range(np.shape(self.activations_PCs)[1]),
            symbol_sequence=[2, 0],
            hover_name=self.activations.index)
        fig.update_layout(title=f'VGGish - {self.source_layer}')
        fig.update_traces(diagonal_visible=False)
        fig.update_layout()
        fig.show()
    
    def plot_3D_PCA(self):
        """Plot 3D PCA of activations"""
        labels = {
            str(i): f"PC {i + 1} ({var:.1f}%)"
            for i, var in enumerate(self.PCs_explained_var * 100)
        }
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=self.activations_PCs[:, 0], y=self.activations_PCs[:, 1], z=self.activations_PCs[:, 2],
                marker=dict(
                    size=4,
                    colorscale='Inferno'),
                hovertext=self.activations.index, mode='markers', legendgroup='markers',
                name=None))  # the name here refers to the individual point
        fig.update_layout(title=f'VGGish - {self.source_layer}')
        fig.update_layout(scene=dict(xaxis_title=labels['0'],
                                     yaxis_title=labels['1'],
                                     zaxis_title=labels['2']))
        fig.show()


def get_source_features(source_model,
                        source_layer,
                        randnetw='False',
                        stimuli_IDs=None,
                        CACHEDIR='/om2/user/gretatu/model-actv/'): # TODO: change cachedir path
    """
    Gets the activations for a given model and layer. Handles the in-house and external models appropriately.
    (same code as in AUD_main under the "Source" section)
    """

    if source_model in ['DCASE2020', 'DS2', 'VGGish', 'AST', 'ZeroSpeech2020', 'wav2vec', 'wav2vecpower', 'sepformer', 'metricGAN', 'S2T']:
        model = PytorchWrapper(model_identifier=source_model, CACHEDIR=CACHEDIR, randnetw=randnetw)
        source_features = model.compile_activations(ID_order=stimuli_IDs, source_layer_of_interest=source_layer_map[source_model][source_layer])
        source_features = source_features.to_numpy()
        print(f'Shape of layer {source_layer}: {np.shape(source_features)}')

    elif source_model.startswith('ResNet50') or source_model.startswith('Kell2018') or source_model.startswith('spectemp'):
        filename = CACHEDIR + f'{source_model}/natsound_activations{d_randnetw[randnetw]}.h5'

        h5file = h5py.File(filename, 'r')
        layers = h5file['layer_list']
        layer_lst = []
        for l in layers:
            layer_lst.append(l.decode('utf-8'))

        # take care of different naming of 'final' layer
        if source_layer == 'final' and source_model.endswith('speaker'):
            source_features = np.asarray(h5file['final/signal/speaker_int'])
        elif source_layer == 'final' and source_model.endswith('audioset'):
            source_features = np.asarray(h5file['final/noise/labels_binary_via_int'])
        # take care of resnet50music having one layer named differently than other resnets
        elif source_layer == 'conv1_relu1' and source_model == ('ResNet50music'):
            source_features = np.asarray(h5file['relu1']) # named relu1 and not conv1_relu1
        # take care of the multitask networks having 3 final layers
        elif source_layer == 'final_word' and source_model.endswith('multitask'):
            source_features = np.asarray(h5file['final/signal/word_int'])
        elif source_layer == 'final_speaker' and source_model.endswith('multitask'):
            source_features = np.asarray(h5file['final/signal/speaker_int'])
        elif source_layer == 'final_audioset' and source_model.endswith('multitask'):
            source_features = np.asarray(h5file['final/noise/labels_binary_via_int'])
        else:
            source_features = np.asarray(h5file[source_layer])

        # print(f'Shape of layer {source_layer}: {np.shape(source_features)} and min value is {np.min(source_features)}')

    else:
        print(f'Source model {source_model} does not exist yet!')
        raise LookupError()

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
            r_prior_zero_manipulation: R prior to setting values to 0 if std(Y_pred) or R<0
            r: R after setting values to 0 if std(Y_pred) == 0 or R<0
            r2: R^2
            r2_train: R^2 of Y_train and Y_train_pred
            y_pred_test: Y_pred of the test set
            alpha: Regularization parameter
            y_pred_test_std: Std of y_pred_test
            y_pred_train_std: Std of y_pred_train
            Y_test_std: Std of y_test (neural)
            Y_train_std: Std of y_train (neural)
            warning_constant_flag: 1 if a constant warning occurred, 0 otherwise
            warning_alpha_limit: 1 if a constant warning occurred for upper alpha bound, 2 if lower bound, 0 otherwise
    
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
            
    else:
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


    return r_prior_zero_manipulation, r, r2, r2_train, y_pred_test, alpha, \
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

    :return: r2 mean (not corrected), corrected r2, best alpha value for the model fit on the meaned data across repetitions

    Base function from Jenelle Feather.
    """
    
    ## Run RidgeCV and return uncorrected r2 ##
    # (mean refers to the fact that the model is fitted on the average of neural responses)
    
    r_prior_zero_manipulation_mean, r_mean, r2_mean, r2_train_mean, \
    y_pred_test_mean, alpha_mean, \
    y_pred_test_std_mean, y_pred_train_std_mean, Y_test_std_mean, Y_train_std_mean, \
    warning_constant_count_mean, warning_alpha_mean = ridgeRegressSplits(source_features,
                                                        np.mean(voxel_data, 2)[:, voxel_idx][:, None], # mean over reps
                                                        is_train_data,
                                                        is_test_data,
                                                        possible_alphas,
                                                        voxel_idx=voxel_idx,
                                                        split_idx=split_idx,
                                                        verbose=True,
                                                        save_plot=save_plot)
                                                        
    split_y_pred = []
    
    ## Get predicted responses (y hat) for models trained on data using just a single repetition ####
    warning_constant_count_splits = [] # store if a warning happened during the estimation of corrected response
    for split_idx, split in enumerate([0, 1, 2]):
        _, _, _, _, y_pred_test_split, _, y_pred_test_std_splits, \
        _, _, _, warning_constant_count_split, _ = ridgeRegressSplits(source_features,
                                             voxel_data[:, voxel_idx, split][:, None], # take out one repetition, and fit model on that (last part is to reshape into samples)
                                             is_train_data,
                                             is_test_data,
                                             possible_alphas,
                                             voxel_idx=voxel_idx,
                                             split_idx=split_idx,
                                             verbose=False, # I am not plotting alpha warnings in the rv hat estimation phase
                                             save_plot=False) # dont save plots within correction estimation phase
        split_y_pred.append(y_pred_test_split)
        warning_constant_count_splits.append(warning_constant_count_split)
        warning_constant_count_splits_sum = np.sum(warning_constant_count_splits)
        
        # store std of y pred of the models fitted on just two splits
        if split_idx == 0:
            y_pred_std_split1 = y_pred_test_std_splits
        elif split_idx == 1:
            y_pred_std_split2 = y_pred_test_std_splits
        else:
            y_pred_std_split3 = y_pred_test_std_splits
    
    ## Correlate neural data and predictions across pairs of repetitions (splits) ##
    rvs = []
    rv_hats = []
    for split_idx, split in enumerate([[0, 1], [1, 2], [0, 2]]):
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
    # if corrected_r2 > 1:
    #     print('>1')
    
    return r_prior_zero_manipulation_mean, r_mean, r2_mean, corrected_r2, r2_train_mean, alpha_mean, \
           y_pred_test_std_mean, y_pred_train_std_mean, Y_test_std_mean, Y_train_std_mean, \
           y_pred_std_split1, y_pred_std_split2, y_pred_std_split3, \
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
