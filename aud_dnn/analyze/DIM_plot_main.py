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
source_models = [  'Kell2018word', 'Kell2018speaker',  'Kell2018music', 'Kell2018audioset', 'Kell2018multitask',
				 'ResNet50word', 'ResNet50speaker', 'ResNet50music', 'ResNet50audioset',   'ResNet50multitask',
				'AST',  'wav2vec', 'DCASE2020', 'DS2', 'VGGish',  'ZeroSpeech2020', 'S2T', 'metricGAN', 'sepformer', 'spectemp']
# source_models = [  'Kell2018word', 'Kell2018speaker',  'Kell2018music', 'Kell2018audioset', 'Kell2018multitask',
# 				 'ResNet50word', 'ResNet50speaker', 'ResNet50music', 'ResNet50audioset',   'ResNet50multitask',
# 				'AST',  'wav2vec', 'VGGish', 'S2T',  'sepformer']
# source_models = ['Kell2018word', 'Kell2018speaker',  'Kell2018music', 'Kell2018audioset', 'Kell2018multitask',
# 				'ResNet50word', 'ResNet50speaker', 'ResNet50music', 'ResNet50audioset',   'ResNet50multitask',]
# source_models = ['AST',  'wav2vec', 'DCASE2020', 'DS2', 'VGGish', 'ZeroSpeech2020', 'S2T', 'metricGAN', 'sepformer']

method = 'regr' # "regr" or "rsa"
target = 'B2021'
if method == 'regr':
	val_of_interest = 'median_r2_test_c'
elif method == 'rsa':
	val_of_interest = 'all_data_rsa_for_best_layer'
agg_method = 'mean'  # across subjects
roi = None

corr_heatmap = False
layerwise_scatter = True
argmax_layer_scatter = False
dim_vs_layer = False


###### LOAD DIM (ED) DATA ######
if method == 'regr':
	# Obtain X combinations of demean and randnetw (but only use two in the data compilation)
	
	# # Normal demean "mean_subtract"
	# d_across_models_dim = compile_dim_data(randnetw='False',
	# 											 fname_dim='all_trained_models_mean_subtract_pca_effective_dimensionality_165_sounds.pckl',
	# 											  str_suffix='_demean')
	# d_across_models_dim_randnetw = compile_dim_data(randnetw='True',
	# 											 fname_dim='all_random_models_mean_subtract_pca_effective_dimensionality_165_sounds.pckl',
	# 											  str_suffix='_randnetw_demean')
	
	# Demean "mean_subtract" with PCA cov
	str_suffix = '_pca_cov'
	d_across_models_dim = compile_dim_data(randnetw='False',
												  fname_dim='all_trained_models_mean_subtract_pca_cov_effective_dimensionality_165_sounds.pckl',
												  str_suffix=str_suffix)
	d_across_models_dim_randnetw = compile_dim_data(randnetw='True',
												   fname_dim='all_random_models_mean_subtract_pca_cov_effective_dimensionality_165_sounds.pckl',
												   str_suffix=f'_randnetw{str_suffix}')

	# # Demean "mean_subtract" with PCA corr
	# str_suffix = '_pca_corr'
	# d_across_models_dim = compile_dim_data(randnetw='False',
	# 									  fname_dim='all_trained_models_mean_subtract_pca_corr_effective_dimensionality_165_sounds.pckl',
	# 									  str_suffix=str_suffix)
	# # d_across_models_dim_randnetw = compile_dim_data(randnetw='True', # DOESNT EXIST!!!!!!!!!!!!!!
	# # 												   fname_dim='all_random_models_mean_subtract_pca_corr_effective_dimensionality_165_sounds.pckl',
	# # 												   str_suffix=f'_randnetw{str_suffix}')
	# d_across_models_dim_randnetw = compile_dim_data(randnetw='True',
	# 											 fname_dim='all_random_models_mean_subtract_pca_effective_dimensionality_165_sounds.pckl',
	# 											  str_suffix='_randnetw_demean') # MOCK!!!!!!!

elif method == 'rsa':
	
	# # Normal std_divide
	# str_suffix = '_zscore'
	# d_across_models_dim = compile_dim_data(randnetw='False',
	# 									   fname_dim='all_trained_models_mean_subtract_std_divide_pca_effective_dimensionality_165_sounds.pckl',
	# 									   str_suffix=str_suffix)
	# d_across_models_dim_randnetw = compile_dim_data(randnetw='True',
	# 									   fname_dim='all_random_models_mean_subtract_std_divide_pca_effective_dimensionality_165_sounds.pckl',
	# 										str_suffix=f'_randnetw{str_suffix}')
	
	# Std_divide with PCA cov
	str_suffix = '_zscore_pca_cov'
	d_across_models_dim = compile_dim_data(randnetw='False',
										   fname_dim='all_trained_models_mean_subtract_std_divide_pca_cov_effective_dimensionality_165_sounds.pckl',
										   str_suffix=str_suffix)
	d_across_models_dim_randnetw = compile_dim_data(randnetw='True',
										   fname_dim='all_random_models_mean_subtract_std_divide_pca_cov_effective_dimensionality_165_sounds.pckl',
											str_suffix=f'_randnetw{str_suffix}')
	#
	# Std_divide with PCA corr
	# str_suffix = '_zscore_pca_corr'
	# d_across_models_dim = compile_dim_data(randnetw='False',
	# 									   fname_dim='all_trained_models_mean_subtract_std_divide_pca_corr_effective_dimensionality_165_sounds.pckl',
	# 									   str_suffix=str_suffix)
	# # d_across_models_dim_randnetw = compile_dim_data(randnetw='True', # DOES NOT EXIST
	# # 									   fname_dim='all_random_models_mean_subtract_std_divide_pca_corr_effective_dimensionality_165_sounds.pckl',
	# # 										str_suffix=f'_randnetw{str_suffix}')
	# d_across_models_dim_randnetw = compile_dim_data(randnetw='True',
	# 									   fname_dim='all_random_models_mean_subtract_std_divide_pca_effective_dimensionality_165_sounds.pckl',
	# 										str_suffix=f'_randnetw{str_suffix}') # MOCK

else:
	raise ValueError('Method not recognized')





# if method == 'regr':
# 	# Obtain 4 combinations of demean and randnetw (but only use two in the data compilation)
# 	d_across_models_dim = compile_dim_data(randnetw='False',
# 										  demean_dim='False',
# 										   std_divide_dim='False',)
# 	d_across_models_dim_demean = compile_dim_data(randnetw='False',
# 												 demean_dim='True',
# 												  std_divide_dim='False',)
# 	d_across_models_dim_randnetw = compile_dim_data(randnetw='True',
# 												   demean_dim='False',
# 													std_divide_dim='False',)
# 	d_across_models_dim_randnetw_demean = compile_dim_data(randnetw='True',
# 														  demean_dim='True',
# 														   std_divide_dim='False',)
# elif method == 'rsa':
# 	d_across_models_dim_demean_std_divide = compile_dim_data(randnetw='False',
# 												 demean_dim='True',
# 												  std_divide_dim='True',)
# 	d_across_models_dim_randnetw_demean_std_divide = compile_dim_data(randnetw='True',
# 														  demean_dim='True',
# 														   std_divide_dim='True',)
#
# else:
# 	raise ValueError('Method not recognized')


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
		RESULTDIR_ROOT=join(DATADIR, 'RSA')) # Copied over Jenelle's results to my datadir
	
	d_across_models_neural_randnetw, d_across_layers_neural_randnetw = load_rsa_scores_across_layers_across_models(
		source_models=source_models,
		target=target,
		roi=roi,
		randnetw='True',
		value_of_interest=val_of_interest,
		agg_method=agg_method,
		RESULTDIR_ROOT=join(DATADIR, 'RSA')) # Copied over Jenelle's results to my datadir
	
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
											   save=False) # SAVEDIR_CENTRALIZED

###### CORRELATION HEATMAP ######
if corr_heatmap:
	corr_heatmap_dim_neural(df_across_models=df_across_models,
							target=target,
							val_of_interest=val_of_interest,
							roi=roi,
							save=SAVEDIR_CENTRALIZED,)
						

##### PLOT OVERALL SCORE ACROSS LAYERS FOR ALL MODELS #####
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
		
			layerwise_scatter_dim_neural(df_across_models=df_across_models,
										target=target,
										source_models=source_models,
										 x_val=x_y_val_combo[0],
										 y_val=x_y_val_combo[1],
										roi=roi,
										same_ylim=same_ylim_flag,
										same_xlim=same_xlim_flag,
										 add_savestr=str_suffix,
										save=False,) # SAVEDIR_CENTRALIZED


##### OBTAIN ARGMAX LAYER PER MODEL #####
if argmax_layer_scatter:
	for randnetw_flag in ['False', 'True']:
		argmax_layer_scatter_dim_neural(df_across_models=df_across_models,
										randnetw=randnetw_flag,
									   target=target,
									   source_models=source_models,
									   val_of_interest=val_of_interest,
									   roi=roi,
									   save=SAVEDIR_CENTRALIZED,)
	
##### PLOT DIM vs LAYER #####
if dim_vs_layer:
	ylim_flags = [(-1,155)]

	
	for demean_flag in ['False', 'True']:
		for ylim_flag in ylim_flags: # [None, (-1,140)]
			dim_across_layers(df_across_models=df_across_models,
							  source_models=source_models,
							  demean_dim=demean_flag,
							  ylim=ylim_flag,
							  save=SAVEDIR_CENTRALIZED)
