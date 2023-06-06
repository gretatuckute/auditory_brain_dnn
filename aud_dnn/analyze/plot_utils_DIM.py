from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import argparse
import os
import pickle
from plotting_specs import *
from scipy.stats import pearsonr, spearmanr
import matplotlib.patches as mpatches
import itertools
import datetime
import getpass
import sys
from os.path import join

# Import core paths from the plot_utils_AUD.py file
from plot_utils_AUD import user, DATADIR, RESULTDIR_ROOT, \
						    SAVEDIR_CENTRALIZED, STATSDIR_CENTRALIZED
from aud_dnn.resources import d_layer_reindex, d_layer_names, d_randnetw, d_model_colors, \
	d_model_names, d_dim_labels, d_roi_colors, FIG_2_5_MODEL_LIST, FIG_7_MODEL_LIST, ALL_MODEL_LIST
from plot_utils_AUD_RSA import load_rsa_scores_across_layers_across_models, \
	package_RSA_best_layer_scores
from plot_utils_AUD import load_score_across_layers_across_models, \
	modelwise_scores_across_targets, layerwise_scores_across_targets,\
	load_scatter_anat_roi_best_layer, permuted_vs_trained_scatter

import rsa_matrix_calculation_all_models

now = datetime.datetime.now()
datetag = now.strftime("%Y%m%d")


def compile_dim_data(source_models,
					 randnetw,
					 fname_dim,
					 str_suffix,
					 EDDIR):
	"""
	Function for compiling dim data across models into a dict with key = model and value a df corresponding to the dim val

	:param randnetw (str): whether to use random network or not
	:param fname_dim (str): name of the dim file
	:param str_suffix (str): suffix to add to the dim file name
	:param EDDIR (str): path to the dim file
	:return:
	"""
	
	print(f'Loading {fname_dim} and using str suffix: {str_suffix}')
	
	with open(os.path.join(EDDIR, fname_dim), 'rb') as f:
		dim_dict = pickle.load(f)
	
	d_across_models_dim = {}

	# Iterate through d_layer_reindex to get the layers for each model in source_models

	for model, layers in d_layer_reindex.items():
		if model not in source_models:
			continue

		if randnetw == 'True' and model in ['spectemp']:
			continue
		
		# Compile the dim values for each model into a dataframe
		dim_dict_model = dim_dict[model]
		
		# Load the version with key=layer to ensure correct ordering
		dim_dict_model_layer = {}
		for layer in layers:
			try:
				dim_dict_model_layer[layer] = dim_dict_model[layer].detach().numpy().ravel()
			except:
				dim_dict_model_layer[layer] = np.real(dim_dict_model[layer].ravel())
		
		# Package with keys = rows (layers) and columns = dim val
		df_dim = pd.DataFrame.from_dict(dim_dict_model_layer, orient='index').rename(columns={0: f'dim{str_suffix}'})
		
		d_across_models_dim[model] = df_dim
	
	return d_across_models_dim


def compile_dim_and_neural_data(source_models,
								target,
								d_across_models_dim,
								d_across_models_dim_randnetw,
								d_across_models_neural,
								d_across_models_neural_randnetw,
								val_of_interest='median_r2_test_c',
								drop_nans=False,
								roi=None,
								save=False,
								add_savestr='_pca'):
	"""
	Compile the dim values and neural values across models into one big dataframe

	:param source_models (list): list of models to compare
	:param d_across_models_dim (dict): dictionary of dim values across models, using whichever preprocessing.
	:param d_across_models_dim_randnetw (dict): dictionary of dim values across models, randnetw, using whichever preprocessing.
	:param d_across_models_neural (dict): dictionary of neural values across layers across models
	:param d_across_models_neural_randnetw (dict): dictionary of neural values across layers across models, randnetw
	:param target (str): target brain data
	:param save (bool or str): whether to save the dataframe. If not False, save to this path
	:return:
	"""
	
	lst_across_models = []  # list of dataframes
	for source_model in source_models:
		
		# Obtain values from 2 ED dicts, one is trained, other one is permuted
		df_dim_demean = d_across_models_dim[source_model].reset_index(drop=False).rename(
			columns={'index': 'source_layer'})
		
		if source_model == 'spectemp':
			# spectemp has no randnetw dimensionality, so fill with nan
			df_dim_randnetw_demean = df_dim_demean.copy(deep=True).rename(columns={f'dim{add_savestr}': f'dim_randnetw{add_savestr}'})
			df_dim_randnetw_demean[f'dim_randnetw{add_savestr}'] = np.nan
		else:
			df_dim_randnetw_demean = d_across_models_dim_randnetw[source_model].reset_index(drop=False).rename(
				columns={'index': 'source_layer'})
		
		# Obtain values from 2 neural dicts
		df_neural = d_across_models_neural[source_model].reset_index(drop=False).rename(
			columns={'index': 'source_layer'})
		df_neural_randnetw = d_across_models_neural_randnetw[source_model].reset_index(drop=False).rename(
			columns={'index': 'source_layer'})
		
		# Assert that ALL indices (i.e. layers) are the same in these dicts
		assert df_dim_demean.index.equals(df_neural.index)
		assert df_neural_randnetw.index.equals(df_neural.index)
		assert df_dim_randnetw_demean.index.equals(df_neural.index)
		
		# Compile
		df_concat = pd.concat([df_dim_demean,
							   df_dim_randnetw_demean,
							   df_neural, df_neural_randnetw], axis=1)
		
		# Drop duplicated columns
		df_concat = df_concat.loc[:, ~df_concat.columns.duplicated()]
		
		# Add in source_model column
		df_concat['source_model'] = source_model
		
		lst_across_models.append(df_concat)
	
	df_across_models = pd.concat(lst_across_models, axis=0)
	
	if drop_nans:
		# find nans if they exist (do not take into account the "roi" column)
		df_across_models['roi'] = str(roi)
		if df_across_models.isnull().values.any():
			print(f'Nans in overall df: {df_across_models.isnull().sum().sum()}')
			# df_across_models.dropna(inplace=True)
			df_across_models.fillna(0, inplace=True)
	
	n_layers = len(df_across_models.index)
	
	if save:
		savestr = f'dim_vs_{target}_across_layers-n={n_layers}-models_' \
				  f'roi-{roi}_' \
				  f'{val_of_interest}{add_savestr}_{datetag}.csv'
		df_across_models.to_csv(join(save, savestr), index=False)

	return df_across_models


def corr_heatmap_dim_neural(df_across_models,
							target,
							val_of_interest='median_r2_test_c',
							roi=None,
							save=False,
							add_savestr=''):
	"""
	Plot correlation heatmap

	Args:
		df_across_models: dataframe with dim and neural scores across models
		target: target neural data
		val_of_interest: which neural metric was used
		roi: which roi was used to obtain the neural data
		save: whether to save the plot, if not False, use that path
		add_savestr: string to add to the saved file name
	"""
	source_models = df_across_models['source_model'].unique()
	
	# Drop cols that start with "yerr"
	df_across_models_corr = df_across_models.copy(deep=True).drop(
		columns=[col for col in df_across_models.columns if col.startswith('yerr')]).corr()
	
	# Plot as heatmap
	plt.figure(figsize=(9, 9))
	sns.heatmap(df_across_models_corr, annot=True,
				cmap='RdBu_r', vmin=-1, vmax=1, square=True)
	plt.tight_layout()
	plt.title(f'Pearson R between dim metrics and neural ({target}) predictivity')
	
	if save:
		savestr = f'corr-heatmap_dim_vs_{target}_' \
				  f'across_models-n={len(source_models)}_' \
				  f'roi-{roi}_' \
				  f'{val_of_interest}{add_savestr}'
		plt.savefig(join(save, f'{savestr}.png'), dpi=180)
		
		# also save as csv
		df_across_models_corr.to_csv(join(save, f'{savestr}.csv'), index=False)
		
	plt.show()
	
def layerwise_scatter_dim_neural(df_across_models,
								 source_models,
								 target,
								 x_val='dim_demean-True',
								 y_val='neural_demean-True',
								 roi=None,
								 same_ylim=None,
								 same_xlim=None,
								 save=False,
								 add_savestr=''):
	""""
	Plot the dim and neural scores for each layer in one big scatter plot.
	
	Args:
		df_across_models: dataframe with dim and neural scores across models
		source_models: list of source models to plot
		target: target neural data
		roi: which roi was used to obtain the neural data
		same_ylim: whether to use set ylim for all plots
		same_xlim: whether to use set xlim for all plots
		save: whether to save the plot, if not False, use that path
		add_savestr: string to add to the saved file name
		
	"""
	df_across_models = df_across_models.copy(deep=True).query(f'source_model in @source_models')
	
	# Check for nans in columns of interest (x_val and y_val)
	if df_across_models[x_val].isnull().values.any():
		print(f'Nans in x_val column {x_val}: {df_across_models[x_val].isnull().sum().sum()}')
		df_across_models = df_across_models.dropna(subset=[x_val])
	if df_across_models[y_val].isnull().values.any():
		print(f'Nans in y_val column {y_val}: {df_across_models[y_val].isnull().sum().sum()}')
		df_across_models = df_across_models.dropna(subset=[y_val])
		
	
	color_order = [d_model_colors[model] for model in df_across_models['source_model']]
	color_order_unique = [d_model_colors[model] for model in df_across_models['source_model'].unique()]
	color_order_unique_names = [d_model_names[model] for model in df_across_models['source_model'].unique()]
	
	# Compute r and p values
	r, p = pearsonr(df_across_models[x_val], df_across_models[y_val])
	r2 = r**2
	
	# Plot scatter
	fig, ax = plt.subplots(figsize=(8, 8))
	# color according to source_model with legend according to colors
	ax.scatter(df_across_models[x_val], df_across_models[y_val],
			   s=60,
			   alpha=1,
			   c=color_order, )
	# Make x axis log scale
	# ax.set_xscale('log')
	ax.set_xlabel(d_dim_labels[x_val], fontsize=26)
	ax.set_ylabel(d_dim_labels[y_val], fontsize=26)
	ax.set_title(
		f'Across all models (n={len(df_across_models.source_model.unique())}) for {target}, r2={r2:.3f}, p={p:.3e}'
		f'\n{d_dim_labels[y_val]} vs. {d_dim_labels[x_val]}')

	# Plot r2 value in lower right corner
	ax.text(0.95, 0.02, f'$R^2$={r2:.3f}', horizontalalignment='right', verticalalignment='bottom',
			transform=ax.transAxes, size=24)
	
	# Legend according to scatter colors
	class_colours = color_order_unique
	recs = []
	for i in range(0, len(class_colours)):
		recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
	# Put legend outside of plot
	lgd = plt.legend(recs, color_order_unique_names, fontsize=11,
					 bbox_to_anchor=(1.35, 0.9))
	
	# Make ticklabels bigger
	ax.tick_params(axis='both', which='major', labelsize=20)
	
	if same_xlim:
		ax.set_xlim(same_xlim)
	if same_ylim:
		ax.set_ylim(same_ylim)
	
	n_layers = len(df_across_models.index)
	
	if save:
		savestr = f'layerwise_scatter_dim_{target}_roi-{roi}_' \
				  f'across_layers-n={n_layers}_models_' \
				  f'{y_val}_vs_{x_val}_' \
				  f'same_xlim={same_xlim}_' \
				  f'same_ylim={same_ylim}{add_savestr}'
		fig.savefig(join(save, f'{savestr}.png'), dpi=180, bbox_inches='tight')
		fig.savefig(join(save, f'{savestr}.svg'), dpi=180, bbox_inches='tight')
		
		# also save as csv
		df_across_models['r'] = r
		df_across_models['p'] = p
		df_across_models['r2'] = r2
		df_across_models.to_csv(join(save, f'{savestr}.csv'), index=False)

	plt.show()

def argmax_layer_scatter_dim_neural(df_across_models,
									randnetw,
								 source_models,
								 target,
								 val_of_interest='median_r2_test_c',
								 roi=None,
								 same_ylim=None,
								 same_xlim=None,
								 save=False,
								 add_savestr=''):
	""""
	Plot the dim and neural scores for each layer in one big scatter plot.
	Takes the argmax best layer for each model (not indepedently selected).
	Not used in paper.
	
	Args:
		df_across_models: dataframe with dim and neural scores across models
		randnetw: str, whether to use randnetw or not (for both neural and dim)
		source_models: list of source models to plot
		target: target neural data
		val_of_interest: which neural metric was used
		roi: which roi was used to obtain the neural data
		same_ylim: whether to use set ylim for all plots
		same_xlim: whether to use set xlim for all plots
		save: whether to save the plot, if not False, use that path
		add_savestr: string to add to the saved file name
		
	"""
	df_across_models = df_across_models.copy(deep=True).query(f'source_model in @source_models')
	
	df_across_models_argmax = df_across_models.copy(deep=True).reset_index(drop=True)
	
	# Find the highest value for each model, across layers, for median_r2_test_c_agg-mean (TRAINED) or median_r2_test_c_agg-mean_randnetw (PERMUTED)
	idx_argmax_layer = df_across_models_argmax.groupby('source_model')[f'{val_of_interest}_agg-mean{d_randnetw[randnetw]}'].apply(lambda x: x.idxmax())
	# df_across_models_argmax['argmax_layer'] = df_across_models_argmax.iloc[idx_argmax_layer].source_layer # for checking, mostly
	
	# Only get the idx_argmax_layer for the models that have the highest value for median_r2_test_c_agg-mean
	df_across_models_argmax_layer = df_across_models_argmax.iloc[idx_argmax_layer]
	
	# Plot
	color_order = [d_model_colors[model] for model in df_across_models_argmax_layer['source_model']]
	color_order_names = [d_model_names[model] for model in df_across_models_argmax_layer['source_model']]
	
	x_y_val_combos = [(f'dim{d_randnetw[randnetw]}_pca', f'median_r2_test_c_agg-mean{d_randnetw[randnetw]}'),
					  (f'dim{d_randnetw[randnetw]}_pca', f'median_r2_test_c_agg-mean{d_randnetw[randnetw]}'),]
	
	for (x_val, y_val) in x_y_val_combos:
	
		r, p = pearsonr(df_across_models_argmax_layer[x_val], df_across_models_argmax_layer[y_val])
		r2 = r**2

		# Plot argmax layer
		fig, ax = plt.subplots(figsize=(10,10))
		ax.scatter(df_across_models_argmax_layer[x_val], df_across_models_argmax_layer[y_val],
				   s=100,
				   alpha=1,
				   color=color_order)
		ax.set_xlabel(d_dim_labels[x_val], fontsize=15)
		ax.set_ylabel(d_dim_labels[y_val], fontsize=15)
		ax.set_title(f'Across all models (n={len(df_across_models_argmax_layer.source_model.unique())}), argmax layer, for {target}, r2={r2:.3f}, p={p:.3e}'
					 f'\n{d_dim_labels[x_val]} vs. {d_dim_labels[y_val]}')
		fig.tight_layout(pad=40)
		# Legend according to scatter colors
		class_colours = color_order
		recs = []
		for i in range(0, len(class_colours)):
			recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
		# Put legend outside of plot
		lgd = plt.legend(recs, color_order_names, fontsize=11,
						 bbox_to_anchor=(1.35, 0.9))
	
		if same_xlim:
			ax.set_xlim(same_xlim)
		if same_ylim:
			ax.set_ylim(same_ylim)
			
		# Make ticklabels bigger
		ax.tick_params(axis='both', which='major', labelsize=16)
	
		if save:
			savestr = f'argmax_layer_scatter_dim_{target}_roi-{roi}_' \
					   f'across_models-n={len(df_across_models.source_model.unique())}{d_randnetw[randnetw]}_' \
					   f'{y_val}_vs_{x_val}_' \
					   f'same_xlim={same_xlim}_' \
					   f'same_ylim={same_ylim}{add_savestr}'
			
			fig.savefig(join(save, f'{savestr}.png'), dpi=180, bbox_inches='tight')
			fig.savefig(join(save, f'{savestr}.svg'), dpi=180, bbox_inches='tight')
			
			# also save as csv
			df_across_models_argmax_layer['r'] = r
			df_across_models_argmax_layer['p'] = p
			df_across_models_argmax_layer['r2'] = r2
			df_across_models_argmax_layer.to_csv(join(save, f'{savestr}.csv'), index=False)
			
		plt.show()

def dim_across_layers(df_across_models,
					  source_models,
					  alpha=1,
					  alpha_randnetw=0.3,
					  label_rotation=45,
					  ylim=None,
					  save=False,
					  str_suffix='_pca',
					  add_savestr=''):
	"""
	Plot dim values across layers for each model in source_models.


	"""
	for source_model in source_models:
		df_model = df_across_models.copy(deep=True).query(f'source_model == "{source_model}"')

		layer_reindex = d_layer_reindex[source_model]
		layer_legend = [d_layer_names[source_model][layer] for layer in layer_reindex]

		title_str = f'{d_model_names[source_model]}, ED)'
		
		# Plot
		fig, ax = plt.subplots(figsize=(7, 5))
		ax.set_box_aspect(0.6)
		plt.errorbar(np.arange(len(layer_reindex)), df_model[f'dim{str_suffix}'],
					 alpha=alpha, lw=2,
					 color=d_roi_colors['none'])
		
		plt.errorbar(np.arange(len(layer_reindex)), df_model[f'dim_randnetw{str_suffix}'],
					 alpha=alpha_randnetw, lw=2,
					 color=d_roi_colors['none'],
					 label='Permuted network')
		
		plt.xticks(np.arange(len(layer_legend)), layer_legend, rotation=label_rotation)
		plt.ylabel(d_dim_labels[f'dim{str_suffix}'], fontsize=15)
		if ylim is not None:
			plt.ylim(ylim)
		plt.title(title_str)
		plt.legend(frameon=False)
		# Make x and y tick labels bigger
		plt.setp(ax.get_xticklabels(), fontsize=13)
		plt.setp(ax.get_yticklabels(), fontsize=13)
		plt.tight_layout(h_pad=1.6)
		if save:
			savestr = f'dim_across_layers{str_suffix}_' \
					   f'source_model-{source_model}_' \
					   f'same_ylim={ylim}{add_savestr}'
			fig.savefig(join(save, f'{savestr}.png'), dpi=180, bbox_inches='tight')
			fig.savefig(join(save, f'{savestr}.svg'), dpi=180, bbox_inches='tight')
		
		plt.show()
	
def loop_across_best_layer_regr_rsa(source_models,
									target,
									randnetw,
									primary_flag='Primary',
									non_primary_flags=['Anterior', 'Lateral', 'Posterior'],
									):
	"""Wrapper for looping over the best layer results for both regression and RSA.

	:param
	source_models: list of strings, source models to loop over
	target: string, target investigste
	randnetw: string, 'True' or 'False'
	primary_flag: string, 'Primary' if the primary ROI is denoted by that name
	non_primary_flags: list of strings, 'Anterior', 'Lateral', 'Posterior'

	"""
	
	# Obtain specs for loading of best layer values
	if randnetw == 'False':
		save_str = '_good-models'
		permuted_flag = False
		layer_exclude_flag = ''
		print(f'randnetw = {randnetw}, \n'
			  f'using save_str = {save_str}\n'
			  f'permuted_flag = {permuted_flag}\n'
			  f'layer_exclude_flag = {layer_exclude_flag}')
	
	elif randnetw == 'True':  # PERMUTED: Version used for paper n=15 models and excludes input layer for in-house
		save_str = '_good-models'  # Was stored with this suffix with the input layers excluded for in-house
		permuted_flag = True
		layer_exclude_flag = ''
		print(f'randnetw = {randnetw}, \n'
			  f'using save_str = {save_str}\n'
			  f'permuted_flag = {permuted_flag}\n'
			  f'layer_exclude_flag = {layer_exclude_flag}')
	else:
		raise ValueError()
	
	# Lists for storing regression and RSA scores
	regr_lst = []
	rsa_lst = []
	
	# Loop across ROIs and load best layer scores for all source models, both regression and RSA
	for non_primary_flag in non_primary_flags:
		#### REGRESSION #####
		df_regr = load_scatter_anat_roi_best_layer(target=target,
												   randnetw=randnetw,
												   annotate=False,
												   condition_col='roi_label_general',
												   collapse_over_val_layer='median',
												   primary_rois=['Primary'],
												   non_primary_rois=[non_primary_flag],
												   yerr_type='within_subject_sem',
												   save_str=save_str,
												   value_of_interest='median_r2_test_c',
												   layer_value_of_interest='rel_pos',
												   layers_to_exclude=layer_exclude_flag,
												   RESULTDIR_ROOT=join(RESULTDIR_ROOT,
																	   'PLOTS_across-models'))
		
		# Rename to add in the names of the primary and non-primary ROIs as well as suffix with _regr
		df_regr = df_regr.rename(columns={'Unnamed: 0': 'source_model',
										  'primary_mean': f'{primary_flag}_mean_regr',
										  # mean refers to the fact that we take the mean across the best layer preferences across participants
										  'non_primary_mean': f'{non_primary_flag}_mean_regr', })
		
		df_regr = df_regr.query(f'source_model in {source_models}')
		
		# Copy the non primary column to a generic name!
		df_regr['NonPrimary_mean_regr'] = df_regr[f'{non_primary_flag}_mean_regr']
		df_regr['ROI'] = non_primary_flag
		
		# Set source_model as index
		df_regr = df_regr.set_index('source_model', drop=False)
		
		assert (df_regr.non_primary_roi.unique() == non_primary_flag)
		regr_lst.append(df_regr[['source_model',
								 'NonPrimary_mean_regr',
								 f'{primary_flag}_mean_regr',
								 'ROI']])
		
		#### RSA ####
		df_rsa = rsa_matrix_calculation_all_models.load_all_models_best_layer_df(
			pckl_path=join(DATADIR, 'RSA', 'best_layer_rsa_analysis_dict.pckl'),
			target=target,
			roi_non_primary=non_primary_flag,
			use_all_165_sounds=True,
			permuted=permuted_flag
		)
		
		df_rsa = df_rsa.query(f'source_model in {source_models}')
		
		# Rename to suffix with _rsa and adhere to naming convention
		df_rsa = df_rsa.rename(columns={f'{primary_flag}_mean_values': f'{primary_flag}_mean_rsa',
										f'{non_primary_flag}_mean_values': f'{non_primary_flag}_mean_rsa', })
		
		# Copy the non primary column to a generic name!
		df_rsa['NonPrimary_mean_rsa'] = df_rsa[f'{non_primary_flag}_mean_rsa']
		df_rsa['ROI'] = non_primary_flag
		
		# Set source_model as index
		df_rsa = df_rsa.set_index('source_model', drop=False)
		
		# Sort in the same way as regression
		df_rsa = df_rsa.reindex(df_regr.index)
		
		rsa_lst.append(df_rsa[['source_model',
							   'NonPrimary_mean_rsa',
							   f'{primary_flag}_mean_rsa',
							   'ROI']])
	
	# Compile into one df
	df_concat_regr = pd.concat(regr_lst)
	df_concat_rsa = pd.concat(rsa_lst)
	
	# We have non primary in one column, and we have primary in the other
	# Check that the primary ones are the same for each non primary column (Lateral, Anterior, Posterior)
	
	# Regression
	assert ([df_concat_regr.query(f'ROI == "Anterior"')[f'{primary_flag}_mean_regr'].unique() ==
			 df_concat_regr.query(f'ROI == "Lateral"')[f'{primary_flag}_mean_regr'].unique()])
	assert ([df_concat_regr.query(f'ROI == "Anterior"')[f'{primary_flag}_mean_regr'].unique() ==
			 df_concat_regr.query(f'ROI == "Posterior"')[f'{primary_flag}_mean_regr'].unique()])
	
	# Append one instantiation of the primary column to the df
	df_concat_regr_primary = df_concat_regr.query(f'ROI == "Anterior"').drop(columns={f'NonPrimary_mean_regr',
																					  'ROI'})
	df_concat_regr_primary['ROI'] = 'Primary'
	
	# Concatenate
	df_regr_final = pd.concat([df_concat_regr.rename(
		columns={f'NonPrimary_mean_regr': 'mean_best_layer_regr'}).drop(columns={f'{primary_flag}_mean_regr'}),
							   df_concat_regr_primary.rename(
								   columns={f'{primary_flag}_mean_regr': 'mean_best_layer_regr'})])
	
	# RSA
	assert ([df_concat_rsa.query(f'ROI == "Anterior"')[f'{primary_flag}_mean_rsa'].unique() ==
			 df_concat_rsa.query(f'ROI == "Lateral"')[f'{primary_flag}_mean_rsa'].unique()])
	assert ([df_concat_rsa.query(f'ROI == "Anterior"')[f'{primary_flag}_mean_rsa'].unique() ==
			 df_concat_rsa.query(f'ROI == "Posterior"')[f'{primary_flag}_mean_rsa'].unique()])
	
	# Append one instantiation of the primary column to the df
	df_concat_rsa_primary = df_concat_rsa.query(f'ROI == "Anterior"').drop(columns={f'NonPrimary_mean_rsa',
																					'ROI'})
	df_concat_rsa_primary['ROI'] = 'Primary'
	
	# Concatenate
	df_rsa_final = pd.concat([df_concat_rsa.rename(columns={f'NonPrimary_mean_rsa': 'mean_best_layer_rsa'}).drop(
		columns={f'{primary_flag}_mean_rsa'}),
							  df_concat_rsa_primary.rename(
								  columns={f'{primary_flag}_mean_rsa': 'mean_best_layer_rsa'})])
	
	return df_regr_final, df_rsa_final
