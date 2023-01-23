from pathlib import Path
import pandas as pd
import numpy as np
import os
import pickle
from scipy.linalg import norm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import scipy.io
from os.path import join

DATADIR = (Path(os.getcwd()) / '..' / '..' / 'data').resolve()
store_new = False
target = 'B2021'

## Package MATLAB ROI labels ##
d_label = {24: ('A1', 'Primary'),
		   174: ('LBelt', 'Primary'),
		   107: ('TA2', 'Anterior'),
		   178: ('PI', 'Anterior'),
		   124: ('Pbelt', 'Lateral'),
		   175: ('A4', 'Lateral'),
		   125: ('A5', 'Lateral'),
		   25: ('PSL', 'Posterior'),
		   104: ('RI', 'Posterior'),
		   105: ('PFcm', 'Posterior'),
		   }

roi_label_lst = []
for label_id, v in d_label.items():
	fname = f'HCP_MMP1_{label_id}_roi.mat'
	mat = scipy.io.loadmat(join(DATADIR, 'fsavg_surf', 'labels_glasser2016', 'hand-audctx_2mm', fname))
	rh = mat['s']['rh'][0][0]
	lh = mat['s']['lh'][0][0]
	
	df_rh = pd.DataFrame(rh, columns=['x_ras', 'y_ras'])
	df_rh['hemi_label'] = 'rh'
	
	df_lh = pd.DataFrame(lh, columns=['x_ras', 'y_ras'])
	df_lh['hemi_label'] = 'lh'
	
	df = pd.concat([df_rh, df_lh])
	
	df['roi_label_fname'] = fname
	df['roi_label_specific'] = v[0]
	df['roi_label_general'] = v[1]
	
	roi_label_lst.append(df)

df_all = pd.concat(roi_label_lst)
df_all['coord_id'] = [f"{(df_all['x_ras'].values[i])}_{df_all['y_ras'].values[i]}" for i in range(len(df_all))]

# check overlaps
u, c = np.unique(df_all.coord_id, return_counts=True)
dup = u[c > 1]

count = 0
coord_ids_to_exlude = [] # list of coord_ids to excluded because they project to several ROIs
for idx, i in enumerate(dup):
	df_check = df_all.loc[df_all.coord_id == i]
	assert (len(df_check.roi_label_specific.unique()) > 1) # the rest are just overlaps between sub-ROIs! These are fine, ultimately they end up in the same general ROI
	if not len(df_check.roi_label_general.unique()) == 1: # If the downsampling causes a given voxel to be in multiple ROIs, do not assign it to a ROI
		print(f'Overlap for ROI labels {df_check.roi_label_general.unique()}, hemi: {df_check.hemi_label.values}')
		count += 1
		coord_ids_to_exlude.append(i)

## IF EXCLUDING BASED ON GENERAL ROI OVERLAP ##
# # Drop the 40 coord ids that are in multiple ROIs
# df_all = df_all.loc[~df_all.coord_id.isin(coord_ids_to_exlude)]
#
# # Assert that all unique coord ids are assigned to a single general ROI
# for idx, i in enumerate(df_all.coord_id.values):
# 	df_check = df_all.loc[df_all.coord_id == i]
# 	assert(len(df_check.roi_label_general.unique()) == 1)

## IF EXCLUDING BASED ON ALL ROIS THAT OVERLAP (THAT PROJECT TO MULTIPLE LABELS) ##
df_all = df_all.drop_duplicates(subset=['coord_id'], keep=False)
df_all['roi_anat_hemi'] = [f'{x.roi_label_general}_{x.hemi_label}' for x in df_all.itertuples()]

df_meta_roi = pd.read_pickle(join(DATADIR, 'neural', target, 'df_roi_meta_wo_anat_labels.pkl'))
print(f'Overlap with {target} unique coord ids: {len(np.intersect1d(df_all.coord_id.values, df_meta_roi.coord_id.values))} out of {len(df_all)}')

## Count how many voxels in each ROI
u, c = np.unique(df_all.roi_anat_hemi, return_counts=True)
print(dict(zip(u,c)))

## Append to the df meta roi
lst_roi_label_fname = []
lst_roi_label_specific = []
lst_roi_label_general = []
lst_roi_anat_hemi = []

for row in df_meta_roi.itertuples():
	specific_coord_id = row.coord_id
	match_row = df_all.loc[df_all.coord_id == specific_coord_id] # in roi label df
	
	if match_row.empty:
		lst_roi_label_fname.append(np.nan)
		lst_roi_label_specific.append(np.nan)
		lst_roi_label_general.append(np.nan)
		lst_roi_anat_hemi.append(np.nan)
	else:
		assert(match_row.hemi_label.values == row.hemi)
		lst_roi_label_fname.append(match_row.roi_label_fname.values[0])
		lst_roi_label_specific.append(match_row.roi_label_specific.values[0])
		lst_roi_label_general.append(match_row.roi_label_general.values[0])
		lst_roi_anat_hemi.append(match_row.roi_anat_hemi.values[0])

		
new_df_meta_roi = df_meta_roi.copy(deep=True)
new_df_meta_roi['roi_label_fname'] = lst_roi_label_fname
new_df_meta_roi['roi_label_specific'] = lst_roi_label_specific
new_df_meta_roi['roi_label_general'] = lst_roi_label_general
new_df_meta_roi['roi_anat_hemi'] = lst_roi_anat_hemi

# Count how many voxels matched each ROI in the specific target dataset
u, c = np.unique(new_df_meta_roi.dropna().roi_anat_hemi, return_counts=True)
print(dict(zip(u,c)))

new_df_meta_roi.to_pickle(join(DATADIR, 'neural', target, 'df_roi_meta.pkl'))

