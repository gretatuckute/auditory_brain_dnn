"""
Quick script for comparing dimensionality (as computed in dimensionality_all_models.py) vs scores across layers, across all models

TODO: should be nicely integrated

"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import argparse
import os
import pickle
from plot_utils_AUD import load_score_across_layers_across_models
from plotting_specs import *
from scipy.stats import pearsonr, spearmanr
import matplotlib.patches as mpatches
import itertools

### GENERAL SETTINGS ###
DATADIR = (Path(os.getcwd()) / '..' / '..' / 'data').resolve()
save = True

###### LOAD ED DATA ######

def compile_dim_data(randnetw,
					demean_ED):
	"""
	Function for compiling ED data across models into a dict with key = model and value a df corresponding to the ED val
	
	:param randnetw (str): whether to use random network or not
	:param demean_ED (str): whether to demean ED or not
	:return:
	"""
	
	if randnetw == 'False':
		if demean_ED == 'True':
			fname_ed = 'all_trained_models_mean_subtract_pca_effective_dimensionality_165_sounds.pckl'
		else:
			fname_ed = 'all_trained_models_effective_dimensionality_165_sounds.pckl'
	elif randnetw == 'True':
		if demean_ED == 'True':
			fname_ed = 'all_random_models_mean_subtract_pca_effective_dimensionality_165_sounds.pckl'
		else:
			fname_ed = 'all_random_models_effective_dimensionality_165_sounds.pckl'
	else:
		raise ValueError('randnetw must be either False or True')
	
	print(f'Loading {fname_ed}')
	
	with open(os.path.join(DATADIR, 'source_models_ED', fname_ed), 'rb') as f:
		ed_dict = pickle.load(f)
	
	d_across_models_ED = {}
	for model, layers in d_layer_reindex.items():
		if (model in ['Kell2018init', 'ResNet50init', 'wav2vecpower',
					  'spectemp']) and randnetw == 'True':  # These models don't have random activations saved.
			continue
		
		# Compile the ED values for each model into a dataframe
		ed_dict_model = ed_dict[model]
		
		# Load the version with key=layer to ensure correct ordering
		ed_dict_model_layer = {}
		for layer in layers:
			ed_dict_model_layer[layer] = ed_dict_model[layer].detach().numpy().ravel()
		
		# Package with keys = rows (layers) and columns = ED val
		df_ed = pd.DataFrame.from_dict(ed_dict_model_layer, orient='index').rename(columns={0: f'ED{d_randnetw[randnetw]}_demean-{demean_ED}'})
		
		d_across_models_ED[model] = df_ed
		
	return d_across_models_ED

# Obtain 3 combinations of demean and randnetw
d_across_models_ED = compile_dim_data(randnetw='False',
									demean_ED='False')
d_across_models_ED_demean = compile_dim_data(randnetw='False',
									demean_ED='True')
d_across_models_ED_randnetw = compile_dim_data(randnetw='True',
									demean_ED='False')
d_across_models_ED_randnetw_demean = compile_dim_data(randnetw='True',
													 demean_ED='True')


###### LOAD NEURAL PREDICTIVITY ######
target = 'NH2015'
val_of_interest = 'median_r2_test_c'
agg_method = 'mean' # across subjects

models_to_omit = ['Kell2018init', 'ResNet50init', 'wav2vecpower', 'spectemp']
d_across_models_neural = load_score_across_layers_across_models(source_models=[model for model in d_layer_reindex.keys() if model not in models_to_omit],
																target=target,
																roi=None,
																randnetw='False',
																value_of_interest=val_of_interest,
																agg_method=agg_method,
																save=False,
																RESULTDIR_ROOT=f'/Users/gt/om2/results/AUD/20210915_median_across_splits_correction/')

d_across_models_neural_randnetw = load_score_across_layers_across_models(
								source_models=[model for model in d_layer_reindex.keys() if model not in models_to_omit],
								target=target,
								roi=None,
								randnetw='True',
								value_of_interest=val_of_interest,
								agg_method=agg_method,
								save=False,
								RESULTDIR_ROOT=f'/Users/gt/om2/results/AUD/20210915_median_across_splits_correction/')

##### COMPARE (loop across d_across_models_neural.keys (i.e. models) and correlate ED with neural score) #####

# Compile the ED values and neural values across models into one big dataframe
lst_across_models = [] # list of dataframes
for source_model in d_across_models_neural.keys():
	print(f'Loading {source_model}')
	
	# Obtain values from 4 ED dicts
	df_ed = d_across_models_ED[source_model].reset_index(drop=False).rename(columns={'index': 'source_layer'})
	df_ed_demean = d_across_models_ED_demean[source_model].reset_index(drop=False).rename(columns={'index': 'source_layer'})
	df_ed_randnetw = d_across_models_ED_randnetw[source_model].reset_index(drop=False).rename(columns={'index': 'source_layer'})
	df_ed_randnetw_demean = d_across_models_ED_randnetw_demean[source_model].reset_index(drop=False).rename(columns={'index': 'source_layer'})
	
	# Obtain values from 2 neural dicts
	df_neural = d_across_models_neural[source_model].reset_index(drop=False).rename(columns={'index': 'source_layer'})
	df_neural_randnetw = d_across_models_neural_randnetw[source_model].reset_index(drop=False).rename(columns={'index': 'source_layer'})
	
	# Assert that ALL indices (i.e. layers) are the same in these 5 dicts
	assert df_ed.index.equals(df_neural.index)
	assert df_ed_demean.index.equals(df_neural.index)
	assert df_ed_randnetw.index.equals(df_neural.index)
	assert df_neural_randnetw.index.equals(df_neural.index)
	assert df_ed_randnetw_demean.index.equals(df_neural.index)
	
	# Compile
	df_concat = pd.concat([df_ed, df_ed_demean,
						   df_ed_randnetw, df_ed_randnetw_demean,
						   df_neural, df_neural_randnetw], axis=1)
	
	# Drop duplicated columns
	df_concat = df_concat.loc[:, ~df_concat.columns.duplicated()]
	
	# Add in source_model column
	df_concat['source_model'] = source_model
	
	lst_across_models.append(df_concat)
	
df_across_models = pd.concat(lst_across_models, axis=0)
if save:
	df_across_models.to_csv(f'dim_vs_neural_across_models.csv', index=False)

# find nans if they exist
if df_across_models.isnull().values.any():
	print(f'Nans in overall df: {df_across_models.isnull().sum().sum()}')
	df_across_models.dropna(inplace=True)
	
##### How correlated are the ED metrics #####
# Drop cols that start with "yerr"
# df_across_models_corr = df_across_models.copy(deep=True).drop(columns=[col for col in df_across_models.columns if col.startswith('yerr')]).corr()
#
# # Plot as heatmap
# plt.figure(figsize=(9,9))
# sns.heatmap(df_across_models_corr, annot=True,
#             cmap='RdBu_r', vmin=-1, vmax=1, square=True)
# plt.tight_layout()
# plt.title('Pearson R between ED and neural predictivity')
# if save:
#     plt.savefig(f'dim_vs_neural_across_models_corr.png')
# plt.show()
#
#
# ##### OBTAIN THE DIFFERENCE IN DEMEANED ED VALUES BETWEEN TRAINED AND RANDNETW #####
# df_across_models['ED_demean-True_DIV_ED_randnetw_demean-True'] = df_across_models['ED_demean-True'] / df_across_models['ED_randnetw_demean-True']
#
# # For neural data
# df_across_models['median_r2_test_c_agg-mean_DIV_median_r2_test_c_agg-mean_randnetw'] = df_across_models['median_r2_test_c_agg-mean'] / df_across_models['median_r2_test_c_agg-mean_randnetw']
#
# # Plot as scatter
# fig, ax = plt.subplots(figsize=(9,9))
# ax.scatter(df_across_models['ED_demean-True_DIV_ED_randnetw_demean-True'], df_across_models['median_r2_test_c_agg-mean_DIV_median_r2_test_c_agg-mean_randnetw'])
# ax.set_ylim(-1,15)
# ax.set_xlim(-1,25)
# ax.set_xlabel('ED_demean-True_DIV_ED_randnetw_demean-True')
# ax.set_ylabel('median_r2_test_c_agg-mean_DIV_median_r2_test_c_agg-mean_randnetw')
# if save:
# 	plt.savefig(f'dim_vs_neural_across_models_ratios.png')
# plt.show()


##### PLOT OVERALL SCORE ACROSS LAYERS FOR ALL MODELS #####
same_ylim =  None #(-0.025,1)
same_xlim =  None #(-1,140)

# x-vals are all columns that start with ED_
x_vals = df_across_models.columns[df_across_models.columns.str.startswith('ED_')].values

# y-vals are all columns that start with {val_of_interest}_
y_vals = df_across_models.columns[df_across_models.columns.str.startswith(f'{val_of_interest}_')].values

x_y_val_combos = [('ED_demean-False', 'median_r2_test_c_agg-mean'),
				  ('ED_demean-True', 'median_r2_test_c_agg-mean'),
				  ('ED_randnetw_demean-False', 'median_r2_test_c_agg-mean_randnetw'),
				  ('ED_randnetw_demean-True', 'median_r2_test_c_agg-mean_randnetw')]


# Drop models that start with Kell or ResNet
df_across_models = df_across_models[~df_across_models['source_model'].str.startswith('Kell')]
df_across_models = df_across_models[~df_across_models['source_model'].str.startswith('ResNet')]

for (x_val, y_val) in x_y_val_combos:
	save_str = f'across_models-n={len(df_across_models.source_model.unique())}_' \
			   f'{y_val}_vs_{x_val}_' \
			   f'target={target}_' \
			   f'same_xlim={same_xlim}_' \
			   f'same_ylim={same_ylim}'
	
	color_order = [d_model_colors[model] for model in df_across_models['source_model']]
	color_order_unique = [d_model_colors[model] for model in df_across_models['source_model'].unique()]
	color_order_unique_names = [d_model_names[model] for model in df_across_models['source_model'].unique()]
	
	# Compute r and p values
	r, p = pearsonr(df_across_models[x_val], df_across_models[y_val])
	
	# Plot scatter
	fig, ax = plt.subplots(figsize=(10,10))
	# color according to source_model with legend according to colors
	ax.scatter(df_across_models[x_val], df_across_models[y_val],
			   s=60,
			   alpha=1,
			   c=color_order,)
	ax.set_xlabel(d_dim_labels[x_val], fontsize=15)
	ax.set_ylabel(d_dim_labels[y_val], fontsize=15)
	ax.set_title(f'Across all models (n={len(df_across_models.source_model.unique())}) for {target}, r={r:.3f}, p={p:.3e}'
				 f'\n{d_dim_labels[y_val]} vs. {d_dim_labels[x_val]}')
	# fig.tight_layout(pad=40)
	
	# Legend according to scatter colors
	class_colours = color_order_unique
	recs = []
	for i in range(0, len(class_colours)):
		recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
	# Put legend outside of plot
	lgd = plt.legend(recs, color_order_unique_names, fontsize=11,
					 bbox_to_anchor=(1.35, 0.9))
	
	if same_xlim:
		ax.set_xlim(same_xlim)
	if same_ylim:
		ax.set_ylim(same_ylim)
		
	if save: # TODO
		fig.savefig(f'{save_str}.png',
					bbox_inches='tight')
	plt.show()

##### OBTAIN ARGMAX LAYER PER MODEL #####
# df_across_models_argmax_layer = df_across_models.copy(deep=True).reset_index(drop=True)
#
# # Find the highest value for each model, across layers, for median_r2_test_c_agg-mean (TRAINED)
# idx_argmax_layer = df_across_models_argmax_layer.groupby('source_model')['median_r2_test_c_agg-mean'].apply(lambda x: x.idxmax())
# # df_across_models_argmax_layer['argmax_layer'] = df_across_models_argmax_layer.iloc[idx_argmax_layer].source_layer # for checking, mostly
#
# # Only get the idx_argmax_layer for the models that have the highest value for median_r2_test_c_agg-mean
# df_across_models_argmax_layer_trained = df_across_models_argmax_layer.iloc[idx_argmax_layer]
#
# # Same for randnetw
# idx_argmax_layer_randnetw = df_across_models_argmax_layer.groupby('source_model')['median_r2_test_c_agg-mean_randnetw'].apply(lambda x: x.idxmax())
# # df_across_models_argmax_layer['argmax_layer_randnetw'] = df_across_models_argmax_layer.iloc[idx_argmax_layer_randnetw].source_layer # for checking, mostly
#
# # Only get the idx_argmax_layer for the models that have the highest value for median_r2_test_c_agg-mean_randnetw
# df_across_models_argmax_layer_randnetw = df_across_models_argmax_layer.iloc[idx_argmax_layer_randnetw]
#
# same_ylim = None  # (-0.025,1)
# same_xlim = None  # (-1,140)
#
# # Plot trained
# color_order = [d_model_colors[model] for model in df_across_models_argmax_layer_trained['source_model']]
# color_order_names = [d_model_names[model] for model in df_across_models_argmax_layer_trained['source_model']]
#
# r, p = pearsonr(df_across_models_argmax_layer_trained['ED_demean-True'], df_across_models_argmax_layer_trained['median_r2_test_c_agg-mean'])
#
# save_str = f'across_models-n={len(df_across_models.source_model.unique())}-argmax-layer-trained_' \
# 		   f'{y_val}_vs_{x_val}_' \
# 		   f'target={target}_' \
# 		   f'same_xlim={same_xlim}_' \
# 		   f'same_ylim={same_ylim}'
#
#
# # Plot argmax layer for median_r2_test_c_agg-mean
# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter(df_across_models_argmax_layer_trained['ED_demean-True'], df_across_models_argmax_layer_trained['median_r2_test_c_agg-mean'],
# 		   s=100,
# 		   alpha=1,
# 		   color=color_order)
# ax.set_xlabel(d_dim_labels['ED_demean-True'], fontsize=15)
# ax.set_ylabel(d_dim_labels['median_r2_test_c_agg-mean'], fontsize=15)
# ax.set_title(f'Across all models (n={len(df_across_models_argmax_layer_trained.source_model.unique())}), argmax layer, for {target}, r={r:.3f}, p={p:.3e}'
# 			 f'\n{d_dim_labels["median_r2_test_c_agg-mean"]} vs. {d_dim_labels["ED_demean-True"]}')
# fig.tight_layout(pad=40)
# # Legend according to scatter colors
# class_colours = color_order
# recs = []
# for i in range(0, len(class_colours)):
# 	recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
# # Put legend outside of plot
# lgd = plt.legend(recs, color_order_names, fontsize=11,
# 				 bbox_to_anchor=(1.35, 0.9))
#
# if same_xlim:
# 	ax.set_xlim(same_xlim)
# if same_ylim:
# 	ax.set_ylim(same_ylim)
#
# if save:
# 	fig.savefig(f'{save_str}.png',
# 				bbox_inches='tight')
# plt.show()
#
# # Plot randnetw
# color_order = [d_model_colors[model] for model in df_across_models_argmax_layer_randnetw['source_model']]
# color_order_names = [d_model_names[model] for model in df_across_models_argmax_layer_randnetw['source_model']]
#
# r, p = pearsonr(df_across_models_argmax_layer_randnetw['ED_demean-True'],
# 				df_across_models_argmax_layer_randnetw['median_r2_test_c_agg-mean'])
#
# save_str = f'across_models-n={len(df_across_models.source_model.unique())}-argmax-layer-randnetw_' \
# 		   f'{y_val}_vs_{x_val}_' \
# 		   f'target={target}_' \
# 		   f'same_xlim={same_xlim}_' \
# 		   f'same_ylim={same_ylim}'
#
# # Plot argmax layer for median_r2_test_c_agg-mean
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.scatter(df_across_models_argmax_layer_randnetw['ED_randnetw_demean-True'],
# 		   df_across_models_argmax_layer_randnetw['median_r2_test_c_agg-mean_randnetw'],
# 		   s=100,
# 		   alpha=1,
# 		   color=color_order)
# ax.set_xlabel(d_dim_labels['ED_randnetw_demean-True'], fontsize=15)
# ax.set_ylabel(d_dim_labels['median_r2_test_c_agg-mean_randnetw'], fontsize=15)
# ax.set_title(
# 	f'Across all models (n={len(df_across_models_argmax_layer_randnetw.source_model.unique())}), argmax layer, for {target}, r={r:.3f}, p={p:.3e}'
# 	f'\n{d_dim_labels["median_r2_test_c_agg-mean_randnetw"]} vs. {d_dim_labels["ED_randnetw_demean-True"]}')
# fig.tight_layout(pad=40)
# # Legend according to scatter colors
# class_colours = color_order
# recs = []
# for i in range(0, len(class_colours)):
# 	recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
# # Put legend outside of plot
# lgd = plt.legend(recs, color_order_names, fontsize=11,
# 				 bbox_to_anchor=(1.35, 0.9))
#
# if same_xlim:
# 	ax.set_xlim(same_xlim)
# if same_ylim:
# 	ax.set_ylim(same_ylim)
#
# if save:
# 	fig.savefig(f'{save_str}.png',
# 				bbox_inches='tight')
# plt.show()
#
#
#
#
# for (x_val, y_val) in x_y_val_combos:
# 	save_str = f'across_models-n={len(df_across_models.source_model.unique())}-argmax-layer_' \
# 			   f'{y_val}_vs_{x_val}_' \
# 			   f'target={target}_' \
# 			   f'same_xlim={same_xlim}_' \
# 			   f'same_ylim={same_ylim}'
#
# 	color_order = [d_model_colors[model] for model in df_across_models['source_model']]
# 	color_order_unique = [d_model_colors[model] for model in df_across_models['source_model'].unique()]
# 	color_order_unique_names = [d_model_names[model] for model in df_across_models['source_model'].unique()]
#
# 	# Compute r and p values
# 	r, p = pearsonr(df_across_models[x_val], df_across_models[y_val])
#
# 	# Plot scatter
# 	fig, ax = plt.subplots(figsize=(10, 10))
# 	# color according to source_model with legend according to colors
# 	ax.scatter(df_across_models[x_val], df_across_models[y_val],
# 			   s=60,
# 			   alpha=1,
# 			   c=color_order, )
# 	ax.set_xlabel(d_dim_labels[x_val], fontsize=15)
# 	ax.set_ylabel(d_dim_labels[y_val], fontsize=15)
# 	ax.set_title(
# 		f'Across all models (n={len(df_across_models.source_model.unique())}) for {target}, r={r:.3f}, p={p:.3e}'
# 		f'\n{d_dim_labels[y_val]} vs. {d_dim_labels[x_val]}')
# 	# fig.tight_layout(pad=40)
#
# 	# Legend according to scatter colors
# 	class_colours = color_order_unique
# 	recs = []
# 	for i in range(0, len(class_colours)):
# 		recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
# 	# Put legend outside of plot
# 	lgd = plt.legend(recs, color_order_unique_names, fontsize=11,
# 					 bbox_to_anchor=(1.35, 0.9))
#
# 	if same_xlim:
# 		ax.set_xlim(same_xlim)
# 	if same_ylim:
# 		ax.set_ylim(same_ylim)
#
# 	if save:  # TODO
# 		fig.savefig(f'{save_str}.png',
# 					bbox_inches='tight')
# 	plt.show()
#
#
#
# # ##### PLOT OVERALL SCORE ACROSS LAYERS PER MODEL #####
# # for source_model in df_across_models.source_model.unique():
# # 	df_per_model = df_across_models[df_across_models['source_model'] == source_model]
# #
# # 	for (x_val, y_val) in x_y_val_combos:
# #
# # 		save_str = f'per_model={source_model}_' \
# # 					   f'{y_val}_vs_{x_val}_' \
# # 					   f'target={target}'
# #
# # 		# Compute r and p values
# # 		r, p = pearsonr(df_per_model[x_val], df_per_model[y_val])
# #
# # 		# Plot scatter
# # 		fig, ax = plt.subplots(figsize=(10,10))
# # 		# color according to source_model
# # 		ax.scatter(df_per_model[x_val], df_per_model[y_val],
# # 				   s=80,
# # 				   alpha=1,
# # 				   c='black')
# # 		ax.set_xlabel(d_dim_labels[x_val], fontsize=15)
# # 		ax.set_ylabel(d_dim_labels[y_val], fontsize=15)
# # 		ax.set_title(f'Source model {d_model_names[source_model]} for {target}, r={r:.3f}, p={p:.3e}'
# # 					 f'\n{d_dim_labels[y_val]} vs.\n{d_dim_labels[x_val]}')
# # 		fig.tight_layout(pad=4)
# # 		if save:
# # 			fig.savefig(f'{save_str}.png')
# # 		plt.show()
#
#
	