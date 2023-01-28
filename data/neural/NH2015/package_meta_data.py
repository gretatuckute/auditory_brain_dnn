"""
Script for storing a meta dataframe for NH2015 and looking into voxel reliability.

This script has three main sections:
1. Compiling metadata on ROIs and loading voxel metadata & looking into the Kell reliability and Pearson R reliability metrics (April 2021)
(as reported in Methods section: "Voxelwise modeling: Correcting for reliability of the measured voxel response"
2. Counting how many voxels are shared across a given coordinate (relevant for plotting) (April 2021)
3. Adding information on which voxels are part of *any* roi (August 2021)

The three sections stores new arrays that are the df_meta_roi for NH2015.
"""
from data.neural.reliability_utils import *
DATADIR = (Path(os.getcwd()) / '..').resolve()

## Parameters
save_new = False # whether to store new versions of meta dataframes
vox_reliability = True # whether to compute Kell/Pearson-based reliability of voxel (test-retest reliability across individual scans)
corrected_vox_reliability = True # whether to compute estimated Spearman-Brown corrected voxel reliability
std_neural = True # whether to look into standard deviation of actual neural data and how that correlates to reliability
package_meta_df = True # whether to package all the metadata information into a dataframe; requires most of above parameters to be True

#### Load neural data and metadata ####

### TARGET - Neural ###
# Where the voxel data lives, and only get the voxels with 3 runs.
voxel_data_all = np.load(os.path.join(DATADIR, 'NH2015', 'voxel_features_array.npy'))
voxel_meta_all = np.load(os.path.join(DATADIR, 'NH2015', 'voxel_features_meta.npy'))
is_3 = voxel_meta_all['n_reps'] == 3
voxel_meta, voxel_data = voxel_meta_all[is_3], voxel_data_all[:, is_3, :]

# See metadata column labels
print((voxel_meta_all[0]).dtype)

n_stim = voxel_data.shape[0]
n_vox = voxel_data.shape[1]
n_reps = voxel_data.shape[2]

# Format metadata associated with each voxel
# create meta df for voxels
d_meta = {'subj_idx':voxel_meta['subj_idx'],
    'hemi':[(v['hemi'].decode('utf-8')) for v in voxel_meta],
    'x_ras':voxel_meta['x_ras'],
    'y_ras':voxel_meta['y_ras'],
     'voxel_id':voxel_meta['voxel_id']}

df_meta = pd.DataFrame(d_meta)

# ROI masks (tonotopic, pitch, music, speech) -- these are already excluding voxels from subjects with only 2 runs
roi_masks = pickle.load(open(os.path.join(DATADIR, 'NH2015', 'roi_masks.cpy'), 'rb'), encoding='latin1')

d_roi_meta = {}
all_roi_idx = []  # for testing overlaps
for roi_idx in range(0, 4):
	measure_voxels = np.nonzero(roi_masks['masks'][:, roi_idx])[0]
	all_roi_idx.append(measure_voxels)
	# create array of length voxel_data
	empty_voxel_col = np.zeros([n_vox])
	empty_voxel_col[measure_voxels] = 1
	d_roi_meta[roi_masks['mask_names'][roi_idx]] = empty_voxel_col

all_roi_idx = [item for sublist in all_roi_idx for item in sublist]
df_roi_meta = pd.DataFrame(d_roi_meta)

### VOXEL RELIABILITY METRICS ####

if vox_reliability:
	# Reliability (test-retest), Kell 2018 / Haignere 2015
	rs = []
	rs_dot = []
	for i in range(0, (n_vox)):
		v = voxel_data[:, i, :]
		v12 = np.mean(v[:, :2], axis=1)
		v3 = v[:, 2]
		proj = ((np.sum(v3.T * v12) / (norm(v3) ** 2))) * v3
		proj_dot = (np.dot(v3, v12) / (norm(v3) ** 2)) * v3
		# proj_kell = (((v3.T) / norm(v3)) * v12) * v3 # the kell eq (original)
		r = 1 - ((norm(v12 - proj)) / (norm(v12))) # ORIG! mine
		r_dot = 1 - ((norm(v12 - proj_dot)) / (norm(v12)))
		rs.append(r)
		rs_dot.append(r_dot)
	
	np.array_equal(np.round(rs, 3), np.round(rs_dot, 3))
	
	# Pearson based correlation
	split_r_all = []
	split_r_median = []
	
	# Lists for storing correlations of individual pairs of scans (like Kell S2a)
	split1_2 = []
	split2_3 = []
	split1_3 = []
	
	for i in range(0, (n_vox)):
		v = voxel_data[:, i, :]
		split_voxelwise = []
		for split_idx, split in enumerate([[0, 1], [1, 2], [0, 2]]):
			split_r  = np.corrcoef(v[:, split[0]], v[:, split[1]])[1,0]
			split_voxelwise.append(split_r)
			split_r_all.append(split_r)
			# store split r depending on which split it is
			if split_idx == 0:
				split1_2.append(split_r)
			elif split_idx == 1:
				split2_3.append(split_r)
			elif split_idx == 2:
				split1_3.append(split_r)
			else:
				raise LookupError()
		split_r_median.append(np.median(split_voxelwise)) # Kell also seemed to use median across all three pairs of scans
	
	# corr between metrics of estimating voxel reliability
	np.corrcoef(rs, split_r_median)
	spearmanr(rs, split_r_median)
	
	# Obtain the median across test-retest variablity
	print(f'Median test-retest reliability was {np.median(split_r_median):.2}, which replicated Kell, p e11, section "Voxelwise modeling: Correcting for reliability of the measured voxel response')
	
	# Plot histogram of the test-retest values
	plt.hist(split_r_median, bins=30, color='orange')
	plt.title(f'Histogram of voxel reliability of individual scans (Kell, S2a)')
	plt.xlabel('Voxel reliability of each individual scan')
	plt.ylabel('Number of voxels')
	plt.show()
	
	# Plot scatter of correlations among different pairs of scans
	plot_sessionwise_split_r(split1_2, split1_3, 'Pearson R: scan 1 vs scan 2', 'Pearson R: scan 1 vs scan 3')
	plot_sessionwise_split_r(split1_2, split2_3, 'Pearson R: scan 1 vs scan 2', 'Pearson R: scan 2 vs scan 3')
	plot_sessionwise_split_r(split1_3, split2_3, 'Pearson R: scan 1 vs scan 3', 'Pearson R: scan 2 vs scan 3')
	
	# How reliable is the reliability computation of individual scan correlations (median):
	print(f'Correlation across voxels of single-scan voxel reliability is '
		  f'{np.median([np.corrcoef(split1_2, split1_3)[1,0], np.corrcoef(split1_2, split2_3)[1,0], np.corrcoef(split1_3, split2_3)[1,0]]):.2}')

### Find spearman-brown corrected reliability ###
if corrected_vox_reliability:
	# These are the actual neural responses correlated across pairs of scan, corrected Spearman-Brown
	split_rv_median_lst = []
	corrected_rv_lst = []
	
	for i in range(0, (n_vox)):
		rvs = []
		v = voxel_data[:, i, :] # all sounds for a given vox across all sessions (reps)
		for split_idx, split in enumerate([[0, 1], [1, 2], [0, 2]]):
			# corr of neural data
			split_rv = np.corrcoef(v[:, split[0]], v[:, split[1]])[1,0]
			rvs.append(split_rv) # append the three corr values across splits, and then take the median of those:
		split_rv_median = np.median(rvs)
		split_rv_median_lst.append(split_rv_median)
		# corrected_rv = max(3 * split_rv_median / (1 + 2 * split_rv_median), 0.128) # k-corrected
		corrected_rv = 3 * split_rv_median / (1 + 2 * split_rv_median)
		corrected_rv_lst.append(corrected_rv)
	
	print(f'The median estimated test-retest reliability of the average response across all three scans (estimated via the Spearman-Brown correction) is {np.median(corrected_rv_lst):.2}')
	
	# Plot histogram of the test-retest values (corrected)
	plt.hist(corrected_rv_lst, bins=35, color='orange')
	plt.title(f'Histogram of estimated reliability of \naveraged voxel response across three scans (Kell, S2a)')
	plt.xlabel('Estimated voxel reliability of average response across all three scans')
	plt.ylabel('Number of voxels')
	plt.show()
	
	
	### Obtain spearman-brown estimates based on just two sessions ###
	# Do different pairs of scans (split half reliability)
	corrected_rv_lst_scans12 = corrected_reliability(voxel_data, [0,1])
	corrected_rv_lst_scans13 = corrected_reliability(voxel_data, [0,2])
	corrected_rv_lst_scans23 = corrected_reliability(voxel_data, [1,2])
	
	# Plot scatter of correlations among different pairs of scans (corrected)
	plot_sessionwise_split_r(corrected_rv_lst_scans12, corrected_rv_lst_scans13,
							 'Scan 1 vs scan 2', 'Scan 1 vs scan 3', min=-0.5, alpha=0.2)
	plot_sessionwise_split_r(corrected_rv_lst_scans12, corrected_rv_lst_scans23,
							 'Scan 1 vs scan 2', 'Scan 2 vs scan 3', min=-0.5, alpha=0.2)
	plot_sessionwise_split_r(corrected_rv_lst_scans13, corrected_rv_lst_scans23,
							 'Scan 1 vs scan 3', 'Scan 2 vs scan 3', min=-0.5, alpha=0.2)
	
	
	# How reliable is the reliability computation of the estimated reliability across different pairs of scans (median):
	np.median([np.corrcoef(corrected_rv_lst_scans12, corrected_rv_lst_scans13)[1,0],
			   np.corrcoef(corrected_rv_lst_scans12, corrected_rv_lst_scans23)[1,0],
			   np.corrcoef(corrected_rv_lst_scans13, corrected_rv_lst_scans23)[1,0]])
	
	# OBS: not reproduced! Kell says 0.80.
	

### VARIABILITY OF THE ACTUAL NEURAL RESPONSE ###
if std_neural:
	split_std_all = []
	split_std_median = []
	std_across_splits = []
	for i in range(0, (n_vox)):
		v = voxel_data[:, i, :]
		std_across_splits.append(np.std(np.mean(v,axis=1)))
		split_voxelwise = []
		for split in range(n_reps):
			split_std = np.std(v[:,split])
			split_voxelwise.append(split_std)
			split_std_all.append(split_std)
		split_std_median.append(np.median(split_voxelwise))
	
	# corr between neural variability and kell/pearson
	np.corrcoef(rs_dot, split_std_median)
	spearmanr(rs_dot, split_std_median)
	
	np.corrcoef(split_r_median, split_std_median)
	spearmanr(split_r_median, split_std_median)
	
	### PLOTS ###
	alpha = 0.4
	plt.figure()
	plt.scatter(std_across_splits, split_std_median, s=3, alpha=alpha)
	plt.ylabel('Median(std of neural response for each repetition)')
	plt.xlabel('Std(mean of neural responses across all repetitions)')
	plt.title('Two different ways of computing variability of neural responses')
	plt.tight_layout()
	if save_new:
		plt.savefig('std_neural_vs_std_mean_neural.pdf',dpi=180)
	plt.tight_layout()
	plt.show()
	
	alpha = 0.8
	plt.figure(figsize=(10,4))
	plt.plot(split_std_median, lw=0.5, alpha=alpha)
	plt.ylabel('Median std of neural response across repetitions')
	plt.xlabel('Voxel number')
	plt.tight_layout()
	plt.show()
	
	alpha = 0.4
	plt.figure(figsize=(15,5))
	plt.scatter(np.arange(len(split_r_median)), split_r_median, s=3, alpha=alpha, label='Pearson R')
	plt.scatter(np.arange(len(split_r_median)), rs, s=3, alpha=alpha, label='Kell reliability')
	plt.axhline(y=0.3,label='y=0.3',color='red')
	plt.ylabel('Median Pearson R across repetitions')
	plt.xlabel('Voxel number')
	plt.legend()
	plt.tight_layout()
	if save_new:
		plt.savefig('kell_vs_pearson.pdf',dpi=180)
	plt.show()
	
	if save_new:
		np.save('kell_r_voxels.npy', rs_dot)
		np.save('pearson_r_voxels.npy', split_r_median)

### PACKAGE RELIABILITY METRICS ###
if package_meta_df:
	d_reliability = {'kell_r_reliability':rs_dot,
					 'pearson_r_reliability':split_r_median,
					 'voxel_variability_mean_reps':std_across_splits, # mean across sessions first, then std of neural response across sounds
					 'voxel_variability_std':split_std_median} # std of each repetition's response, and then median thereof
	
	df_reliability = pd.DataFrame(d_reliability)
	
	### CONCAT ALL DFS ###
	df_full = pd.concat([df_meta, df_roi_meta, df_reliability], axis=1)
	if save_new:
		df_full.to_pickle(os.path.join(DATADIR, 'NH2015', 'df_voxel_features_meta.pkl'))
	
	######### ADD SHARED BY METADATA ###########
	# Based on the unique x and y ras coordinates, add information about how many subjects share a given voxel
	meta = df_full.copy(deep=True)
	
	# find unique coordinates
	str_ras_coords = [f"{(meta['x_ras'].values[i])}_{meta['y_ras'].values[i]}" for i in range(len(meta))]
	unique_coords, count_coords = np.unique(str_ras_coords, return_counts=True)
	counts_coords = dict(zip(unique_coords, count_coords))
	print(counts_coords)
	print(np.max(count_coords)) # some coordinates are repeated across all 8 subjects
	
	# count of counts, ie how many overlapping voxels are there
	u, c = np.unique(count_coords, return_counts=True)
	count2 = dict(zip(u, c))
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
		
	# In Aug 2021, one column was added, according to the code below:
	
	### ADD ANY ROI COLUMN TO META ###
	# Add a column that denotes that a given voxel is in *any* roi
	any_roi_array = np.max(np.vstack([meta.tonotopic.values, meta.pitch.values,
									  meta.music.values, meta.speech.values]), axis=0)
	
	assert(int(np.sum(any_roi_array)) == len(np.unique(all_roi_idx)))
	
	meta['any_roi'] = any_roi_array
	
	# reorganize columns
	cols = meta.columns
	fixed_cols = ['subj_idx', 'hemi', 'x_ras', 'y_ras', 'voxel_id', 'tonotopic', 'pitch',
		   'music', 'speech', 'any_roi', 'kell_r_reliability', 'pearson_r_reliability',
		   'voxel_variability_mean_reps', 'voxel_variability_std', 'coord_id',
		   'shared_by']
	
	meta = meta[fixed_cols]
	if save_new:
		meta.to_pickle(os.path.join(DATADIR, 'NH2015', 'df_roi_meta_wo_anat_labels.pkl'))
