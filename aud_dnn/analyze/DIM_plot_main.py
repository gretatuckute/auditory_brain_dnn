from plot_utils_DIM import *
from plot_utils_AUD_RSA import load_rsa_scores_across_layers_across_models

"""
Script for:

1. comparing dimensionality (as computed in dimensionality_all_models.py) vs scores across layers, across all models

"""

### Directories ###
# Logging
date = datetime.datetime.now().strftime("%m%d%Y-%T")
if user != 'gt':
	sys.stdout = open(join(RESULTDIR_ROOT, 'logs', f'out-{date}.log'), 'a+')

### Settings ###
source_models = ALL_MODEL_LIST

method = 'rsa' # "regr" or "rsa"
target = 'NH2015' # 'NH2015' or 'B2021'
if method == 'regr':
	val_of_interest = 'median_r2_test_c'
elif method == 'rsa':
	val_of_interest = 'all_data_rsa_for_best_layer'
agg_method = 'mean'  # across subjects
roi = None

save = True
if save:
	save = SAVEDIR_CENTRALIZED
else:
	save = False

layerwise_scatter = True
layerwise_scatter_within_architecture = False # Reviewer comment
corr_heatmap = False # sanity check, not used in paper
argmax_layer_scatter = False # sanity check, not used in paper
dim_vs_layer = False # sanity check, not used in paper


###### LOAD DIM (ED) DATA ######
if method == 'regr':
	
	# Method for computing ED: Demean "mean_subtract" using PCA covariance
	str_suffix = '_pca'
	d_across_models_dim = compile_dim_data(source_models=source_models,
											randnetw='False',
											fname_dim='all_trained_models_mean_subtract_pca_effective_dimensionality_165_sounds.pckl',
											str_suffix=str_suffix,
											EDDIR=join(RESULTDIR_ROOT, 'ED'))
	d_across_models_dim_randnetw = compile_dim_data(source_models=source_models,
													randnetw='True',
													fname_dim='all_random_models_mean_subtract_pca_effective_dimensionality_165_sounds.pckl',
													str_suffix=f'_randnetw{str_suffix}',
													EDDIR=join(RESULTDIR_ROOT, 'ED'))

elif method == 'rsa':
	
	# Method for computing ED: Std_divide (z-score) using PCA covariance
	str_suffix = '_zscore_pca'
	d_across_models_dim = compile_dim_data(source_models=source_models,
										   randnetw='False',
										   fname_dim='/Users/gt/Documents/GitHub/auditory_brain_dnn/results/ED/all_trained_models_mean_subtract_std_divide_pca_effective_dimensionality_165_sounds.pckl', #'all_trained_models_mean_subtract_std_divide_pca_cov_effective_dimensionality_165_sounds.pckl',
										   str_suffix=str_suffix,
										   EDDIR=join(RESULTDIR_ROOT, 'ED'))
	d_across_models_dim_randnetw = compile_dim_data(source_models=source_models,
													randnetw='True',
													fname_dim='all_random_models_mean_subtract_std_divide_pca_effective_dimensionality_165_sounds.pckl', #'all_random_models_mean_subtract_std_divide_pca_cov_effective_dimensionality_165_sounds.pckl',
													str_suffix=f'_randnetw{str_suffix}',
													EDDIR=join(RESULTDIR_ROOT, 'ED'))

else:
	raise ValueError('Method not recognized')


###### LOAD NEURAL PREDICTIVITY/REPRESENTATIONAL SIMILARITY ACROSS LAYERS ######
if method == 'regr':
	d_across_models_neural, d_across_layers_neural = load_score_across_layers_across_models(
		source_models=source_models,
		target=target,
		roi=roi,
		randnetw='False',
		value_of_interest=val_of_interest,
		agg_method=agg_method,
		save=False, # True or False, savedir is every model's output dir
		RESULTDIR_ROOT=RESULTDIR_ROOT)
	
	d_across_models_neural_randnetw, d_across_layers_neural_randnetw = load_score_across_layers_across_models(
		source_models=source_models,
		target=target,
		roi=roi,
		randnetw='True',
		value_of_interest=val_of_interest,
		agg_method=agg_method,
		save=False,
		RESULTDIR_ROOT=RESULTDIR_ROOT)
	
elif method == 'rsa':
	d_across_models_neural, d_across_layers_neural = load_rsa_scores_across_layers_across_models(
		source_models=source_models,
		target=target,
		roi=roi,
		randnetw='False',
		value_of_interest=val_of_interest,
		agg_method=agg_method,
		RESULTDIR_ROOT=join(RESULTDIR_ROOT, 'RSA')) # Copied over Jenelle's results to my datadir
	
	d_across_models_neural_randnetw, d_across_layers_neural_randnetw = load_rsa_scores_across_layers_across_models(
		source_models=source_models,
		target=target,
		roi=roi,
		randnetw='True',
		value_of_interest=val_of_interest,
		agg_method=agg_method,
		RESULTDIR_ROOT=join(RESULTDIR_ROOT, 'RSA')) # Copied over Jenelle's results to my datadir
	
else:
	raise ValueError('method must be "regr" or "rsa"')


###### COMPILE INTO ONE DF ######
df_across_models = compile_dim_and_neural_data(source_models=source_models,
											   target=target,
											   d_across_models_dim=d_across_models_dim,
											   d_across_models_dim_randnetw=d_across_models_dim_randnetw,
											   d_across_models_neural=d_across_models_neural,
											   d_across_models_neural_randnetw=d_across_models_neural_randnetw,
											   drop_nans=False,
											   val_of_interest=val_of_interest,
											   add_savestr=str_suffix,
											   save=save)
						

##### PLOT SCORE ACROSS LAYERS FOR ALL MODELS VS ED #####
if layerwise_scatter:
	
	x_y_val_combos = [(f'dim{str_suffix}', f'{val_of_interest}_agg-{agg_method}'),
					  (f'dim_randnetw{str_suffix}', f'{val_of_interest}_agg-{agg_method}_randnetw')]
	
	if method == 'regr': # which ylimit to use!
		ylim_flag = (-0.015,1)
	elif method == 'rsa':
		ylim_flag = (-0.015,0.8)
	else:
		raise ValueError()
	
	for x_y_val_combo in x_y_val_combos:

		for same_ylim_flag in [None, ylim_flag]:
			if same_ylim_flag is None:
				same_xlim_flag = None
			else:
				if method == 'regr':
					same_xlim_flag = (-1,70) # max val for pca cov is ~65
				elif method == 'rsa':
					same_xlim_flag = (-1, 130) # max val for pca cov (std_divide!) is ~121
				else:
					raise ValueError()
		
			layerwise_scatter_dim_neural(df_across_models=df_across_models,
										target=target,
										source_models=source_models,
										 x_val=x_y_val_combo[0],
										 y_val=x_y_val_combo[1],
										roi=roi,
										same_ylim=same_ylim_flag,
										same_xlim=same_xlim_flag,
										 add_savestr=str_suffix,
										save=save,)

#### PLOT SCORE ACROSS LAYERS FOR ALL MODELS VS ED, WITHIN ARCHITECTURE (KELL AND RESNET) #####
if layerwise_scatter_within_architecture:

	source_model_lst = [['Kell2018word', 'Kell2018speaker',
						'Kell2018music', 'Kell2018audioset',
						'Kell2018multitask',
						'Kell2018wordClean', 'Kell2018speakerClean',],
						['ResNet50word', 'ResNet50speaker',
						'ResNet50music', 'ResNet50audioset',
						'ResNet50multitask',
						'ResNet50wordClean', 'ResNet50speakerClean',]]

	x_y_val_combos = [(f'dim{str_suffix}', f'{val_of_interest}_agg-{agg_method}'),
					  (f'dim_randnetw{str_suffix}', f'{val_of_interest}_agg-{agg_method}_randnetw')]

	if method == 'regr': # which ylimit to use!
		ylim_flag = (-0.015,1)
	elif method == 'rsa':
		ylim_flag = (-0.015,0.8)
	else:
		raise ValueError()

	for source_models in source_model_lst:

		df_across_models_within_architecture = df_across_models[df_across_models['source_model'].isin(source_models)]

		for x_y_val_combo in x_y_val_combos:

			for same_ylim_flag in [None, ylim_flag]:
				if same_ylim_flag is None:
					same_xlim_flag = None
				else:
					if method == 'regr':
						same_xlim_flag = (-1,70)
					elif method == 'rsa':
						same_xlim_flag = (-1, 130)
					else:
						raise ValueError()

				layerwise_scatter_dim_neural(df_across_models=df_across_models_within_architecture,
											target=target,
											source_models=source_models,
											 x_val=x_y_val_combo[0],
											 y_val=x_y_val_combo[1],
											roi=roi,
											same_ylim=same_ylim_flag,
											same_xlim=same_xlim_flag,
											 add_savestr=str_suffix,
											save=save,)




###### CORRELATION HEATMAP (not used in paper, just a quick sanity check) ######
if corr_heatmap:
	corr_heatmap_dim_neural(df_across_models=df_across_models,
							target=target,
							val_of_interest=val_of_interest,
							roi=roi,
							save=save,)

##### OBTAIN ARGMAX LAYER PER MODEL #####
if argmax_layer_scatter:
	for randnetw_flag in ['False', 'True']:
		argmax_layer_scatter_dim_neural(df_across_models=df_across_models,
										randnetw=randnetw_flag,
									   target=target,
									   source_models=source_models,
									   val_of_interest=val_of_interest,
									   roi=roi,
									   save=save,)
	
##### PLOT DIM vs LAYER #####
if dim_vs_layer:
	ylim_flags = [(-1,155)]

	
	for ylim_flag in ylim_flags:
		dim_across_layers(df_across_models=df_across_models,
						  source_models=source_models,
						  ylim=ylim_flag,
						  str_suffix=str_suffix,
						  save=save)
