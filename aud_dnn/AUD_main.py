from utils import *

DATADIR = (Path(os.getcwd()) / '..' / 'data').resolve()
RESULTDIR = (Path(os.getcwd()) / '..' / 'results_post-changes1-v2').resolve()
CACHEDIR = (Path(os.getcwd()) / '..' / 'model_actv').resolve().as_posix()

from resources import source_layer_map, d_randnetw

# Set random seed
import random
np.random.seed(0)
random.seed(0)

date = datetime.datetime.now().strftime("%m%d%Y-%T")

"""
Fits a ridge regression model from source (DNN unit activations) to target (brain or component data).
"""


def main(raw_args=None):
    parser = argparse.ArgumentParser(description='Fit voxel-wise models in the auditory domain')
    parser.add_argument('--source_model', default='ResNet50multitask',
                        type=str, help='Name of source DNN model.'
                                       'Name of the DNN model from which activations will be used as regressors.'
                                       'Should match the folder name in the CACHEDIR which contains the DNN model activations.')
    parser.add_argument('--source_layer', default='layer3',
                        type=str, help='The source DNN layer from which activations will be used as regressors.')
    parser.add_argument('--target', default='NH2015',
                        type=str, help='Target, i.e. neural data or component data. '
                                       'Options are "NH2015", "B2021", "NH2015comp"')
    parser.add_argument('--alphalimit', default=50,
                        type=str, help='Which limit to use for possible alphas in ridge regression.'
                                       'Will be range: 10 ** x for x in range(-alphalimit, alphalimit)')
    parser.add_argument('--randnetw', default='False',
                        type=str, help='If "True": extract DNN model activations from a permuted model. '
                                       'Activations should be suffixed with _randnetw.'
                                       'If "False": extract DNN model activations from the original model (no suffix).')
    parser.add_argument('--save_plot', default='False', type=str, help='Whether to save diagnostic plots')
    parser.add_argument('--save_preds', default='True',
                        type=str, help='Whether to save predictions for each single sound in each cross-validation fold.')
    parser.add_argument('--verbose', default=False,
                        type=bool, help='If True, print progress to console.'
                                        'If False, direct print output to log file.')
    args = parser.parse_args(raw_args)

    print('*' * 40)
    print(vars(args))
    print('*' * 40)

    ##### Identifiers, result folders and logging #####
    identifier = f'AUD-MAPPING-Ridge_' \
                 f'TARGET-{args.target}_' \
                 f'SOURCE-{args.source_model}-{args.source_layer}_' \
                 f'RANDNETW-{args.randnetw}_' \
                 f'ALPHALIMIT-{args.alphalimit}'

    RESULTFOLDER = (Path(RESULTDIR / args.source_model / identifier))
    (Path(RESULTFOLDER)).mkdir(exist_ok=True, parents=True)
    
    PLOTFOLDER = False
    if args.save_plot == 'True': # generate folder for plots
        PLOTFOLDER = (Path(RESULTFOLDER / 'diagnostic_plots'))
        (Path(PLOTFOLDER)).mkdir()

    # Logging
    if not args.verbose:
        sys.stdout = open(os.path.join(RESULTFOLDER,
                                       f'out-{date}.log'), 'a+')


    ##### Stimuli (sounds) #####
    sound_meta = np.load(os.path.join(DATADIR,
                                      f'neural/NH2015/neural_stim_meta.npy')) # (original indexing, neural data is extracted in this order)

    # Extract wav names in order (this is how the neural data is ordered --> order all activations in the same way)
    stimuli_IDs = []
    for i in sound_meta:
        stimuli_IDs.append(i[0][:-4].decode("utf-8")) # remove .wav


    ##### Load target (neural data or components) #####
    voxel_data, voxel_id = get_target(target=args.target,
                             stimuli_IDs=stimuli_IDs,
                             DATADIR=DATADIR)
    n_stim = voxel_data.shape[0]
    n_vox = voxel_data.shape[1]

        
    ##### SOURCE (DNN unit activations) #####
    source_features = get_source_features(source_model=args.source_model,
                                            source_layer=args.source_layer,
                                            source_layer_map=source_layer_map,
                                            stimuli_IDs=stimuli_IDs,
                                            randnetw=args.randnetw,
                                            CACHEDIR=CACHEDIR)
        
    ##### Cross-validation #####

    ## Setup splits ##
    n_CV_splits = 10
    possible_alphas = [10 ** x for x in range(-args.alphalimit, args.alphalimit)]
    possible_alphas = possible_alphas[::-1]
    n_for_train = 83
    n_for_test = 82
    
    ## Setup logging arrays ##
    
    # store r-values for test and train (and alpha values)
    r_voxels_test_prior_zero_manipulation = np.zeros([n_vox, n_CV_splits]) # save r value without setting it to zero if it is negative or std = 0
    r_voxels_test = np.zeros([n_vox, n_CV_splits])
    r2_voxels_test = np.zeros([n_vox, n_CV_splits])
    r2_voxels_test_corrected = np.zeros([n_vox, n_CV_splits])
    r2_voxels_train = np.zeros([n_vox, n_CV_splits])
    alphas = np.zeros([n_vox, n_CV_splits])

    # store std of predicted responses
    y_pred_test_std_mean = np.zeros([n_vox, n_CV_splits]) # predicted
    y_pred_train_std_mean = np.zeros([n_vox, n_CV_splits]) # predicted
    y_test_std_mean = np.zeros([n_vox, n_CV_splits]) # neural
    y_train_std_mean = np.zeros([n_vox, n_CV_splits]) # neural
    y_pred_std_split1 = np.zeros([n_vox, n_CV_splits])
    y_pred_std_split2 = np.zeros([n_vox, n_CV_splits])
    y_pred_std_split3 = np.zeros([n_vox, n_CV_splits])
    
    # store warnings
    warning_constant_mean = np.zeros([n_vox, n_CV_splits]) # whether there is a warning for a constant prediction when fitting on the mean of the three repetitions
    warning_constant_splits = np.zeros([n_vox, n_CV_splits]) # whether there is a warning for a constant prediction when fitting on just two repetitions (for reliability estimation)
    warning_alphas = np.zeros([n_vox, n_CV_splits])
    
    # store predictions
    if args.save_preds == 'True':
        # store the predictions for each sound in the test set
        y_preds_test = np.zeros([n_stim, n_CV_splits, n_vox])
        y_preds_test[:] = np.nan # fill with nan, because we wont have every value for every sound (given that some will be in the train set)
    
    
    all_train_idxs = np.zeros([n_stim, n_CV_splits]) # same split across all voxels
    all_test_idxs = np.zeros([n_stim, n_CV_splits])

    for split_idx in range(n_CV_splits):
        print(f'Running split {split_idx}\n')

        # Randomly pick train/test indices
        train_data_idxs = np.random.choice(n_stim, size=n_for_train, replace=False)
        set_of_possible_test_idxs = set(np.arange(n_stim)) - set(train_data_idxs)
        test_data_idxs = np.random.choice(list(set_of_possible_test_idxs), size=n_for_test, replace=False)
        is_train_data, is_test_data = np.zeros((n_stim), dtype=bool), np.zeros((n_stim), dtype=bool)
        is_train_data[train_data_idxs], is_test_data[test_data_idxs] = True, True

        all_train_idxs[:, split_idx] = is_train_data.copy() # columns: splits --> denotes whether stim was used as train data
        all_test_idxs[:, split_idx] = is_test_data.copy()

        for track_vox_idx, voxel_idx in enumerate(range(n_vox)):
            if voxel_idx % 100 == 1:
                print(f'Running voxel number: {voxel_idx}')
            
            # Run components: not possible to obtain corrected value
            if args.target == 'NH2015comp':
                r_voxels_test_prior_zero_manipulation[track_vox_idx, split_idx],\
                r_voxels_test[track_vox_idx, split_idx], \
                r2_voxels_test[track_vox_idx, split_idx],\
                r2_voxels_train[track_vox_idx, split_idx],\
                y_pred_test, \
                alphas[track_vox_idx, split_idx],\
                y_pred_test_std_mean[track_vox_idx, split_idx], \
                y_pred_train_std_mean[track_vox_idx, split_idx], \
                y_test_std_mean[track_vox_idx, split_idx],\
                y_train_std_mean[track_vox_idx, split_idx], \
                warning_constant_mean[track_vox_idx, split_idx], \
                warning_alphas[track_vox_idx, split_idx] = ridgeRegressSplits(source_features=source_features,
                                                                              y=voxel_data.to_numpy()[:, voxel_idx][:, None],
                                                                              is_train_data=is_train_data,
                                                                              is_test_data=is_test_data,
                                                                              possible_alphas=possible_alphas,
                                                                              voxel_idx=voxel_idx,
                                                                              split_idx=split_idx,)
                if args.save_preds == 'True':
                    # Append y_pred_test in the test indices, for the correct voxel, for the correct split
                    y_preds_test[is_test_data, split_idx, track_vox_idx] = y_pred_test.ravel()
                
            # Run neural data with correction
            else:
                r_voxels_test_prior_zero_manipulation[track_vox_idx, split_idx],\
                r_voxels_test[track_vox_idx, split_idx], \
                r2_voxels_test[track_vox_idx, split_idx],\
                r2_voxels_test_corrected[track_vox_idx, split_idx],\
                r2_voxels_train[track_vox_idx, split_idx],\
                alphas[track_vox_idx, split_idx],\
                y_pred_test_std_mean[track_vox_idx, split_idx], \
                y_pred_train_std_mean[track_vox_idx, split_idx], \
                y_test_std_mean[track_vox_idx, split_idx],\
                y_train_std_mean[track_vox_idx, split_idx], \
                y_pred_std_split1[track_vox_idx, split_idx],\
                y_pred_std_split2[track_vox_idx, split_idx],\
                y_pred_std_split3[track_vox_idx, split_idx],\
                warning_constant_mean[track_vox_idx, split_idx], \
                warning_constant_splits[track_vox_idx, split_idx], \
                warning_alphas[track_vox_idx, split_idx] = ridgeCV_correctedR2(source_features=source_features,
                                                                              voxel_data=voxel_data,
                                                                              voxel_idx=voxel_idx,
                                                                              split_idx=split_idx,
                                                                              is_train_data=is_train_data,
                                                                              is_test_data=is_test_data,
                                                                              possible_alphas=possible_alphas,
                                                                              save_plot=PLOTFOLDER)
                
            sys.stdout.flush()
            
            # if track_vox_idx == 5:
            #     break
    
    ## LOAD METADATA ##
    df_roi_meta = pd.read_pickle(os.path.join(DATADIR, f'neural/{args.target}/df_roi_meta.pkl'))
    
    # if args.target == 'B2021': # chunked data
    #     df_roi_meta = df_roi_meta[df_roi_meta.index.isin(np.arange(d_chunk_size[args.target_chunk_count][0], d_chunk_size[args.target_chunk_count][1]))]
    #
    ## CV SPLITS ##
    dict_splits = {'all_train_idxs': all_train_idxs,
                    'all_test_idxs': all_test_idxs}
    
    ## SAVE RESULTS ACROSS CV SPLITS ##
    # if args.target == 'B2021': # chunked data
    #     vox_idx_coord = np.arange(d_chunk_size[args.target_chunk_count][0], d_chunk_size[args.target_chunk_count][1])
    # else:
    #     vox_idx_coord = np.arange(n_vox)

    vox_idx_coord = np.arange(n_vox)

    
    ds = xr.Dataset(
        {"r_prior_zero": (("vox_idx", "splits"), r_voxels_test_prior_zero_manipulation), # todo unify splits split_idx
         "r_test": (("vox_idx", "split_idx"), r_voxels_test),
         "r2_test": (("vox_idx", "split_idx"), r2_voxels_test),
         "r2_test_c": (("vox_idx", "splits"), r2_voxels_test_corrected),
         "r2_train": (("vox_idx", "split_idx"), r2_voxels_train),
         "alphas": (("vox_idx", "splits"), alphas),
         "y_pred_test_std_mean": (("vox_idx", "splits"), y_pred_test_std_mean),
         "y_pred_train_std_mean": (("vox_idx", "splits"), y_pred_train_std_mean),
         "y_test_std_mean": (("vox_idx", "splits"), y_test_std_mean),
         "y_train_std_mean": (("vox_idx", "splits"), y_train_std_mean),
         "y_pred_std_split1": (("vox_idx", "splits"), y_pred_std_split1),
         "y_pred_std_split2": (("vox_idx", "splits"), y_pred_std_split2),
         "y_pred_std_split3": (("vox_idx", "splits"), y_pred_std_split3),
         "warning_constant_mean": (("vox_idx", "split_idx"), warning_constant_mean),
         "warning_constant_splits": (("vox_idx", "splits"), warning_constant_splits),
         "warning_alphas": (("vox_idx", "splits"), warning_alphas),
         },
        coords={
            "vox_idx_coord": vox_idx_coord,
            "voxel_id": voxel_id,
            "split_idx_coord": np.arange(10),
            "source_model": ("vox_idx_coord", [str(args.source_model)]*n_vox),
            "source_layer": ("vox_idx_coord", [str(args.source_layer)]*n_vox),
            "randnetw": ("vox_idx_coord", [str(args.randnetw)]*n_vox)},
        attrs=dict_splits)
    
    
    ## SAVE MEANED RESULTS ##
    # Assert that no nans exist in r_voxels_test, r2_voxels_test, r2_voxels_test_corrected
    assert np.isnan(r_voxels_test).sum() == 0
    assert np.isnan(r2_voxels_test).sum() == 0
    assert np.isnan(r2_voxels_test_corrected).sum() == 0

    # Create a df with num voxels as rows, and columns with metrics of interest
    df_results = pd.DataFrame({'mean_r_prior_zero_test': np.nanmean(r_voxels_test_prior_zero_manipulation, 1),
                               'median_r_prior_zero_test': np.nanmedian(r_voxels_test_prior_zero_manipulation, 1),
                               'mean_r_test': np.mean(r_voxels_test, 1),
                               'median_r_test': np.median(r_voxels_test, 1),
                               'mean_r2_test': np.mean(r2_voxels_test, 1),
                               'median_r2_test': np.median(r2_voxels_test, 1),
                               'std_r2_test': np.std(r2_voxels_test, 1),
                               'mean_r2_test_c': np.mean(r2_voxels_test_corrected, 1),
                               'median_r2_test_c': np.median(r2_voxels_test_corrected, 1),
                               'std_r2_test_c': np.std(r2_voxels_test_corrected, 1),
                               'mean_r2_train': np.mean(r2_voxels_train, 1),
                               'median_r2_train': np.median(r2_voxels_train, 1),
                               'std_r2_train': np.std(r2_voxels_train, 1),
                               'median_alpha': np.median(alphas, 1),
                               'mean_alpha': np.mean(alphas, 1),
                               })

    # Use the voxel_id as index
    df_results['voxel_id'] = voxel_id
    df_results = df_results.set_index('voxel_id', drop=False, inplace=False)
    df_results['vox_idx_coord'] = vox_idx_coord

    # concatenate with masks and meta
    if args.target == 'B2021':
        df_results.set_index(vox_idx_coord, inplace=True) # set index to match which voxel was run based on the chunk

    assert(voxel_id == df_roi_meta.voxel_id.values).all() # Assert that the meta and the results are in the same order (voxel_id)

    df_roi_meta = df_roi_meta.set_index('voxel_id', drop=True, inplace=False) # otherwise we end up with two identical voxel_id columns
    assert(df_results.index == df_roi_meta.index).all() # Once more assertion that the meta and the results are in the same order (voxel_id)

    df_output = pd.concat([df_results, df_roi_meta], axis=1)

    ## Log ##
    df_output['source_model'] = args.source_model
    df_output['source_layer'] = args.source_layer
    df_output['randnetw'] = args.randnetw
    df_output['target'] = args.target

    
    if args.target == 'B2021': # store with chunk name
        df_output.to_pickle(os.path.join(RESULTFOLDER, f'df_output_{args.target_chunk_count}.pkl'))
        pickle.dump(ds, open(os.path.join(RESULTFOLDER, f'ds_{args.target_chunk_count}.pkl'), 'wb'))
    else:
        df_output.to_pickle(os.path.join(RESULTFOLDER, 'df_output.pkl'))
        pickle.dump(ds, open(os.path.join(RESULTFOLDER, 'ds.pkl'), 'wb'))
        if args.save_preds == 'True':
            pickle.dump(y_preds_test, open(os.path.join(RESULTFOLDER, 'y_preds_test.pkl'), 'wb'))
            # Indexed according to stimuli_IDs, and vox order is the same as in the df_roi_metadata

    print(f'Saved results to {RESULTFOLDER}')

if __name__ == '__main__':
    main()






