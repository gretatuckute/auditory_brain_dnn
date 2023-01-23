import numpy as np
from dimensionality_all_models import *
import pandas as pd
from scipy.io import loadmat

DATADIR='/om/user/gretatu/aud-dnn/data'
RESULTSDIR='/om2/user/jfeather/projects/aud-dnn/results/dimensionality_output_2022_08_18_Ssquare/'
mean_subtract = True
eig_type = 'corr'

for std_divide in [False, True]:
    for target in ['NH2015', 'B2021']:
        save_name_base = os.path.join(RESULTSDIR, f'{target}_ms{mean_subtract}_sd{std_divide}_eigtype{eig_type}')
        print('Dimensionality on %s'%target)
        print('Mean Subtract: %s | Std Divide: %s | Eig Type %s'%(mean_subtract, std_divide, eig_type))
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

        # Average across all runs? Not sure if this is the right comparison.
        A = voxel_data.mean(2)

        if mean_subtract:
            scaler_x = StandardScaler(with_std=std_divide).fit(A) # fit demeaning transform on train data only
            A = scaler_x.transform(A) # demeans column-wise

        effective_dim = run_pca_on_layer_output(A, target, save_name_base, eig_type)

