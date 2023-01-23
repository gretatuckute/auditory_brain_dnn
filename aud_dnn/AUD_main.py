from utils import *

DATADIR = (Path(os.getcwd()) / '..' / 'data').resolve()
RESULTDIR = (Path(os.getcwd()) / '..' / 'results').resolve()

from model_specs import source_layer_map, d_randnetw, d_chunk_size

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
    parser.add_argument('--source_model', default='ResNet50multitask', type=str, help='Source model')
    parser.add_argument('--source_layer', default='input_after_preproc', type=str, help='Model layer of interest')
    parser.add_argument('--target', default='B2021', type=str, help='Target')
    parser.add_argument('--target_chunk_count', default='0', type=str, help='Which target chunk to run')
    parser.add_argument('--alphalimit', default=50, type=str, help='Which limit to use for possible alphas in ridge regression')
    parser.add_argument('--randemb', default='False', type=str, help='If True: run a random embedding of the same size of the source layer')
    parser.add_argument('--randnetw', default='False', type=str, help='If True: extract model activations from a random network')
    parser.add_argument('--save_plot', default='False', type=str, help='Whether to save diagnostic plots')
    parser.add_argument('--save_preds', default='True', type=str, help='Whether to save predictions for each single sound')
    args = parser.parse_args(raw_args)

    print('*' * 40)
    print(vars(args))
    print('*' * 40)
    warnings.warn(f'Args: {vars(args)}')
    
    identifier = f'AUD-MAPPING-Ridge_' \
                 f'TARGET-{args.target}_' \
                 f'SOURCE-{args.source_model}-{args.source_layer}_' \
                 f'RANDEMB-{args.randemb}_RANDNETW-{args.randnetw}_' \
                 f'ALPHALIMIT-{args.alphalimit}'

    user = getpass.getuser()
    if user == 'gt':
        CACHEDIR = 'model-actv/'
        RESULTFOLDER = (Path(RESULTDIR / identifier))
    else:
        CACHEDIR = f'/om2/user/{user}/model-actv/'
        RESULTFOLDER = (Path(Path(f'/om2/user/{user}/results/AUD/20221213_post_reviews6') / args.source_model / identifier))

    (Path(RESULTFOLDER)).mkdir(exist_ok=True, parents=True)
    
    PLOTFOLDER = False
    if args.save_plot == 'True': # generate folder for plots
        PLOTFOLDER = (Path(RESULTFOLDER / 'diagnostic_plots'))
        (Path(PLOTFOLDER)).mkdir()

    # Logging
    if user != 'gt':
        sys.stdout = open(os.path.join(RESULTFOLDER, f'out-{date}.log'), 'a+')

    ## Stimuli (original indexing, activations are extracted in this order) ##
    sound_meta = np.load(os.path.join(DATADIR, f'neural/NH2015/neural_stim_meta.npy'))

    # Extract wav names in order (this is how the neural data is ordered --> order all activations in the same way)
    stimuli_IDs = []
    for i in sound_meta:
        stimuli_IDs.append(i[0][:-4].decode("utf-8")) # remove .wav

    ## TARGET - brain / components ##
    if args.target == 'NH2015':
        # Where the voxel data lives, and only get the voxels with 3 runs.
        voxel_data_all = np.load(os.path.join(DATADIR, f'neural/{args.target}/voxel_features_array.npy'))
        voxel_meta_all = np.load(os.path.join(DATADIR, f'neural/{args.target}/voxel_features_meta.npy'))
        is_3 = voxel_meta_all['n_reps'] == 3
        voxel_meta, voxel_data = voxel_meta_all[is_3], voxel_data_all[:, is_3, :]
        
        n_stim = voxel_data.shape[0]
        n_vox = voxel_data.shape[1]
        
    elif args.target == 'NH2015comp':
        comp_data = loadmat(os.path.join(DATADIR, f'neural/{args.target}/components.mat'))
        comp_stimuli_IDs = comp_data['stim_names']
        comp_stimuli_IDs = [x[0] for x in comp_stimuli_IDs[0]]

        comp_names = comp_data['component_names']
        comp_names = [x[0][0] for x in comp_names]

        comp = comp_data['R'] # R is the response matrix
        
        # add to df with index stimuli IDs and component names as columns
        df_comp = pd.DataFrame(comp, index=comp_stimuli_IDs, columns=comp_names)
        
        # reindex so it is indexed the same way as the neural data and activations (stimuli_IDs)
        voxel_data = df_comp.reindex(stimuli_IDs)
        
        n_stim = voxel_data.shape[0]
        n_vox = voxel_data.shape[1]

    elif args.target == 'B2021':
        sound_meta_mat = loadmat(os.path.join(DATADIR, f'neural/{args.target}/stim_info_v4'))['stim_info']

        stimuli_IDs_B2021 = []
        for i in sound_meta_mat['stim_names'][0][0]:
            stimuli_IDs_B2021.append((i[0][0].astype(str)))

        assert(stimuli_IDs == stimuli_IDs_B2021[:165]) # same order as NH2021
        
        voxel_data = np.load(os.path.join(DATADIR, f'neural/{args.target}/voxel_features_array.npy'))
        
        # Truncate to only run the first 165 sound set
        voxel_data = voxel_data[:len(stimuli_IDs), :, :]
        
        # Only run the chunk of interest (given that there are ~27K voxels, we chunk it into four parts and run each separately)
        voxel_data = voxel_data[:, d_chunk_size[args.target_chunk_count][0]:d_chunk_size[args.target_chunk_count][1], :] # todo: make this more robust
        
        n_stim = voxel_data.shape[0]
        n_vox = voxel_data.shape[1]
        
        assert(n_vox == 6698) # due to chunking
        
    else:
        raise LookupError(f'Target dataset of interest {args.target} not found!')

        
    ### SOURCE (DNN unit activations) ###
    # TODO: migrate to using the get_source_features function from utils (basically a wrapper from the code chunk below)
    if args.source_model in ['DCASE2020', 'DS2', 'VGGish', 'AST', 'ZeroSpeech2020', 'wav2vec', 'wav2vecpower', 'sepformer', 'metricGAN', 'S2T']:
        model = PytorchWrapper(model_identifier=args.source_model, CACHEDIR=CACHEDIR, randnetw=args.randnetw)
        source_features = model.compile_activations(ID_order=stimuli_IDs, source_layer_of_interest=source_layer_map[args.source_model][args.source_layer])
        source_features = source_features.to_numpy()
        print(f'Shape of layer {args.source_layer}: {np.shape(source_features)}')

    elif args.source_model.startswith('ResNet50') or args.source_model.startswith('Kell2018') or args.source_model.startswith('spectemp'):
        filename = CACHEDIR + f'{args.source_model}/natsound_activations{d_randnetw[args.randnetw]}.h5'

        h5file = h5py.File(filename, 'r')
        layers = h5file['layer_list']
        layer_lst = []
        for l in layers:
            layer_lst.append(l.decode('utf-8'))
        
        # take care of different naming of 'final' layer
        if args.source_layer == 'final' and args.source_model.endswith('speaker'):
            source_features = np.asarray(h5file['final/signal/speaker_int'])
        elif args.source_layer == 'final' and args.source_model.endswith('audioset'):
            source_features = np.asarray(h5file['final/noise/labels_binary_via_int'])
        # take care of resnet50music having one layer named differently than other resnets
        elif args.source_layer == 'conv1_relu1' and args.source_model == ('ResNet50music'):
            source_features = np.asarray(h5file['relu1']) # named relu1 and not conv1_relu1
        # take care of the multitask networks having 3 final layers
        elif args.source_layer == 'final_word' and args.source_model.endswith('multitask'):
            source_features = np.asarray(h5file['final/signal/word_int'])
        elif args.source_layer == 'final_speaker' and args.source_model.endswith('multitask'):
            source_features = np.asarray(h5file['final/signal/speaker_int'])
        elif args.source_layer == 'final_audioset' and args.source_model.endswith('multitask'):
            source_features = np.asarray(h5file['final/noise/labels_binary_via_int'])
        else:
            source_features = np.asarray(h5file[args.source_layer])

        print(f'Shape of layer {args.source_layer}: {np.shape(source_features)} and min value is {np.min(source_features)}')

    else:
        print(f'Source model {args.source_model} does not exist yet!')
        raise LookupError()
        
    if args.randemb == 'True':
        source_features = np.random.rand(n_stim, source_features.shape[1])

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
        y_preds_test[:] = np.nan # fill with nan, because we wont have every value for every sound (given that some will be in the test set)
    
    
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
                warning_alphas[track_vox_idx, split_idx] = ridgeRegressSplits(source_features,
                                                                                voxel_data.to_numpy()[:, voxel_idx][:, None],
                                                                                is_train_data,
                                                                                is_test_data,
                                                                                possible_alphas,
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
                warning_alphas[track_vox_idx, split_idx] = ridgeCV_correctedR2(source_features,
                                                      voxel_data,
                                                      voxel_idx,
                                                      split_idx,
                                                      is_train_data,
                                                      is_test_data,
                                                      possible_alphas,
                                                      save_plot=PLOTFOLDER)
                
            sys.stdout.flush()
            
            # if track_vox_idx == 3:
            #     break
    
    ## LOAD METADATA ##
    df_roi_meta = pd.read_pickle(os.path.join(DATADIR, f'neural/{args.target}/df_roi_meta.pkl'))
    
    if args.target == 'B2021': # chunked data
        df_roi_meta = df_roi_meta[df_roi_meta.index.isin(np.arange(d_chunk_size[args.target_chunk_count][0], d_chunk_size[args.target_chunk_count][1]))]
    
    ## CV SPLITS ##
    dict_splits = {'all_train_idxs': all_train_idxs,
                    'all_test_idxs': all_test_idxs}
    
    ## SAVE RESULTS ACROSS CV SPLITS ##
    if args.target == 'B2021': # chunked data
        vox_idx_coord = np.arange(d_chunk_size[args.target_chunk_count][0], d_chunk_size[args.target_chunk_count][1])
    else:
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
            "vox_idx_coord": vox_idx_coord, # todo add voxel_id
            "split_idx_coord": np.arange(10),
            "source_model": ("vox_idx_coord", [str(args.source_model)]*n_vox),
            "source_layer": ("vox_idx_coord", [str(args.source_layer)]*n_vox),
            "randemb": ("vox_idx_coord", [str(args.randemb)]*n_vox),
            "randnetw": ("vox_idx_coord", [str(args.randnetw)]*n_vox)},
        attrs=dict_splits)
    
    
    ## SAVE MEANED RESULTS ##
    # Create a df with num voxels as rows, and columns with metrics of interest
    
    # Compute a version of the corrected r2 test where the nans are set to zero first.
    # (should not be relevant anymore, but can be used for assertions)
    r2_voxels_test_corrected_no_nan = copy.deepcopy(r2_voxels_test_corrected)
    nan_idx = np.isnan(r2_voxels_test_corrected_no_nan)
    r2_voxels_test_corrected_no_nan[nan_idx] = 0
    
    df_results = pd.DataFrame({'mean_r_prior_zero_test': np.nanmean(r_voxels_test_prior_zero_manipulation, 1),
                               'median_r_prior_zero_test': np.nanmedian(r_voxels_test_prior_zero_manipulation, 1),
                               'mean_r_test': np.nanmean(r_voxels_test,1),
                               'median_r_test': np.nanmedian(r_voxels_test, 1),
                               'mean_r2_test': np.nanmean(r2_voxels_test,1),
                               'median_r2_test': np.nanmedian(r2_voxels_test,1),
                               'std_r2_test': np.nanstd(r2_voxels_test,1),
                               'mean_r2_test_c': np.nanmean(r2_voxels_test_corrected, 1),
                               'mean_r2_test_c_no_nan': np.mean(r2_voxels_test_corrected_no_nan, 1),
                               'median_r2_test_c': np.nanmedian(r2_voxels_test_corrected, 1),
                               'median_r2_test_c_no_nan': np.median(r2_voxels_test_corrected_no_nan, 1),
                               'std_r2_test_c': np.nanstd(r2_voxels_test_corrected, 1),
                               'std_r2_test_c_no_nan': np.std(r2_voxels_test_corrected_no_nan, 1),
                               'mean_r2_train': np.nanmean(r2_voxels_train, 1),
                               'median_r2_train': np.median(r2_voxels_train, 1),
                               'std_r2_train': np.std(r2_voxels_train, 1),
                               'median_alpha': np.median(alphas,1),
                               'mean_alpha': np.mean(alphas, 1),
                               })

    # concatenate with masks and meta
    if args.target == 'B2021':
        df_results.set_index(vox_idx_coord, inplace=True) # set index to match which voxel was run based on the chunk
        
    df_output = pd.concat([df_results, df_roi_meta], axis=1)

    ## Log ##
    df_output['source_model'] = args.source_model
    df_output['source_layer'] = args.source_layer
    df_output['randemb'] = args.randemb
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

if __name__ == '__main__':
    main()






