"""This script computes the reliability metrics for B2021 as reported in Kell 2018 Methods section:
Voxelwise modeling: Correcting for reliability of the measured voxel response

I use all 192 sounds.

Note that the metadata that was ultimately used for the B2021 dataset was derived from Dana Boebinger's MATLAB metadata
(packaged in /GitHub/aud-dnn/data/neural/B2021/package_data_as_python.py), i.e. this script does not store df_roi_meta.pkl.

"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from reliability_utils import *

DATADIR = (Path(os.getcwd()) / '..' / '..' / 'data').resolve()
RESULTDIR = (Path(os.getcwd()) / '..' / 'results').resolve()

## Parameters
save_new = False # whether to store new versions of meta dataframes
vox_reliability = True # whether to compute Kell/Pearson-based reliability of voxel (test-retest reliability across individual scans)
corrected_vox_reliability = True # whether to compute estimated Spearman-Brown corrected voxel reliability

#### Load neural data and metadata ####

### TARGET - Neural ###
voxel_data = np.load(os.path.join(DATADIR, f'neural/B2021/voxel_features_array.npy'))

n_stim = voxel_data.shape[0]
n_vox = voxel_data.shape[1]
n_reps = voxel_data.shape[2]

### VOXEL RELIABILITY METRICS ####

if vox_reliability:
	# Reliability (test-retest), Kell 2018 / Haignere 2015
	rs = []
	rs_dot = []
	rs_dana = []
	for i in range(0, (n_vox)):
		v = voxel_data[:, i, :]
		v12 = np.mean(v[:, :2], axis=1)
		v3 = v[:, 2]
		proj = ((np.sum(v3.T * v12) / (norm(v3) ** 2))) * v3
		proj_dot = (np.dot(v3, v12) / (norm(v3) ** 2)) * v3
		# proj_kell = (((v3.T) / norm(v3)) * v12) * v3
		r = 1 - ((norm(v12 - proj)) / (norm(v12)))
		r_dot = 1 - ((norm(v12 - proj_dot)) / (norm(v12)))
		r_dana = 1 - ((norm(v12 - proj)**2) / (norm(v12)**2)) # squares in both the numerator and denominator!
		rs.append(r)
		rs_dot.append(r_dot)
		rs_dana.append(r_dana)
	
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
	print(f'Median test-retest reliability was {np.median(split_r_median):.2}')
	
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
	
