from pathlib import Path
import pandas as pd
import numpy as np
import os
import pickle
from scipy.linalg import norm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import scipy.io

DATADIR = (Path(os.getcwd()) / '..' / '..' ).resolve()
store_new = False

## Package stimuli as neural_stim_meta.npy ##

# Load NH2015 for comparison
target = 'NH2015'
sound_meta = np.load(os.path.join(DATADIR, f'neural/{target}/neural_stim_meta.npy'))
stimuli_IDs = []
for i in sound_meta:
	stimuli_IDs.append(i[0][:-4].decode("utf-8"))  # remove .wav

voxel_data_all = np.load(os.path.join(DATADIR, f'neural/{target}/voxel_features_array.npy'))
voxel_meta_all = np.load(os.path.join(DATADIR, f'neural/{target}/voxel_features_meta.npy'))

# packaged meta
df_roi_meta = pd.read_pickle(os.path.join(DATADIR, f'neural/{target}/df_roi_meta.pkl'))

## Load B2021 mat files ##
mat = scipy.io.loadmat('voxel_responses.mat')

# Store data as a nd array, 3d
voxel_data_all_B2021 = mat['vals']
if store_new:
	np.save('voxel_features_array.npy', voxel_data_all_B2021)

n_stim = voxel_data_all_B2021.shape[0]
n_vox = voxel_data_all_B2021.shape[1]
n_reps = voxel_data_all_B2021.shape[2]

#### Create meta df ####

## Obtain more reliability metrics
split_std_median = [] # std of each repetitions response across sounds, and then median of those three values
std_across_splits = [] # mean across repetitions, and take std of response across sounds
split_r_median = [] # pearson r correlation across repetitions, ie pairwise pearson correlation across pairs of repetitions
for i in range(0, (n_vox)):
	v = voxel_data_all_B2021[:, i, :]
	std_across_splits.append(np.std(np.mean(v, axis=1)))
	split_voxelwise = []
	split_voxelwise_r_median = []
	
	# compute std of each repetition separately
	for split in range(n_reps):
		split_std = np.std(v[:, split])
		split_voxelwise.append(split_std)
	
	# compute pairwise correlations across repetitions
	for split_idx, split in enumerate([[0, 1], [1, 2], [0, 2]]):
		split_r = np.corrcoef(v[:, split[0]], v[:, split[1]])[1, 0]
		split_voxelwise_r_median.append(split_r)
		
	split_std_median.append(np.median(split_voxelwise))
	split_r_median.append(np.median(split_voxelwise_r_median))

meta = pd.DataFrame({'subj_idx':mat['subj'].ravel(),
					   'hemi': ['rh' if i == 1 else 'lh' for i in mat['hemi'].ravel()],
					  'x_ras': mat['x_ras'].ravel(),
					  'y_ras': mat['y_ras'].ravel(),
					  'voxel_id': np.arange(len(mat['subj'].ravel())),
					  'kell_r_reliability': mat['rel'].ravel() / 100, # Dana's version of it, cf email
					  'pearson_r_reliability':split_r_median,
					  'voxel_variability_mean_reps':std_across_splits, # mean across sessions first, then std of neural response across sounds
					  'voxel_variability_std':split_std_median # std of each repetition's response, and then median thereof
					  })


## Compute how many subjects share a given voxel

# find unique coordinates
str_ras_coords = [f"{(meta['x_ras'].values[i])}_{meta['y_ras'].values[i]}" for i in range(len(meta))]
unique_coords, count_coords = np.unique(str_ras_coords, return_counts=True)
counts_coords = dict(zip(unique_coords, count_coords))
print(counts_coords)
print(np.max(count_coords))  # some coordinates are repeated across all 20 subjects

# count of counts, ie how many overlapping voxels are there
u, c = np.unique(count_coords, return_counts=True)
count2 = dict(zip(u, c)) # for each coord, how many subjects share this
print(count2)

# Add information to the meta df
meta['coord_id'] = str_ras_coords
meta['shared_by'] = np.nan
for coord_val in unique_coords:
	c = coord_val.split('_')
	x = int(c[0])
	y = int(c[1])
	match_row_idx = meta.loc[np.logical_and(meta['x_ras'] == x, meta['y_ras'] == y)].index.values
	# look up how many times this coord combo was found across subjects:
	shared_by_count = counts_coords[coord_val]
	# append to shared by column
	meta.loc[match_row_idx, 'shared_by'] = int(shared_by_count)

if store_new:
	meta.to_pickle('df_roi_meta.pkl')


