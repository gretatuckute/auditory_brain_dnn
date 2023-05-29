from plot_utils_AUD import *
np.random.seed(0)
random.seed(0)

### Directories ###
date = datetime.datetime.now().strftime("%m%d%Y")
DATADIR = (Path(os.getcwd()) / '..' / '..' / 'data').resolve()
RESULTDIR_LOCAL = (Path(os.getcwd()) / '..' / '..' / 'results').resolve()
PLOTSURFDIR = Path(f'{ROOT}/results/PLOTS_SURF_across-models/')
SURFDIR = f'{DATADIR}/fsavg_surf/'

### Settings for which plots to make ###
save = True # Whether to save any plots/csvs
concat_over_models = True

# If concat_over_models = False, we load each individual model and perform the analysis on that
if not concat_over_models:
	# Shared for neural and components
	best_layer_cv_nit = False # Basis for Figure 2 for neural, basis for Figure 5 for components; obtain best layer for each voxel based on independent CV splits across 10 iterations

	# Neural specific
	pred_across_layers = False # SI 2; predictivity for each model across all layers
	best_layer_anat_ROI = False # Basis for Figure 7 for neural; best layer for each anatomical ROI
	run_surf_argmax = False # Basis for Figure 6 for neural, dump argmax surface position to .mat file
	run_surf_argmax_merge_datsets = False # Basis for Figure 6 for neural, merge NH2015 and B2021 datasets to find argmax surface position across both datasets
	run_surf_direct_val = False # plotting arbitrary values on the surface (not used in paper)


if concat_over_models:
	# Shared for neural and components
	plot_barplot_across_models = False # Figure 2 for neural, Figure 5 for components; barplot of performance across models
	plot_scatter_across_models = False # Figure 2, Seed1 vs Seed2 scatter for neural
	plot_word_clean_models = False # Figure 9 for neural and components; barplot of performance for word models vs clean models

	# Neural specific
	plot_anat_roi_scatter = True # Figure 7 neural; scatter of performance across models for anatomical ROIs
	stats_barplot_across_models = False # Figure 2 neural; stats for barplot of performance across models
	determine_surf_colorscale = False # For figuring out which colorscale to use for Figure 6
	median_surface_across_models = False # Figure 6 neural; median surface across models for each dataset

	# Component specific
	plot_barplot_across_inhouse_models = False # Figure 8A) for components (in-house models)
	plot_scatter_comp_vs_comp = False # Figure 8B) for components (in-house models)
	plot_scatter_pred_vs_actual = False # Figure 4, scatter for components


target = 'B2021'

# Logging
date = datetime.datetime.now().strftime("%m%d%Y-%T")
if user != 'gt':
	sys.stdout = open(join(RESULTDIR_ROOT, 'logs', f'out-{date}.log'), 'a+')

# All models (n=19)
# source_models = [  'Kell2018word', 'Kell2018speaker',  'Kell2018music', 'Kell2018audioset', 'Kell2018multitask',
# 				 'ResNet50word', 'ResNet50speaker', 'ResNet50music', 'ResNet50audioset',   'ResNet50multitask',
# 				'AST',  'wav2vec', 'DCASE2020', 'DS2',  'VGGish', 'ZeroSpeech2020', 'S2T', 'metricGAN', 'sepformer']# 'spectemp']
# source_models = [ 'ResNet50audioset',   'ResNet50multitask',
# 				'AST',  'wav2vec', 'DCASE2020', 'DS2',  'VGGish', 'ZeroSpeech2020', 'S2T', 'metricGAN', 'sepformer', 'spectemp']
# Models above spectemp baseline (n=15)
source_models = [  'Kell2018word', 'Kell2018speaker',  'Kell2018music', 'Kell2018audioset', 'Kell2018multitask',
				 'ResNet50word', 'ResNet50speaker', 'ResNet50music', 'ResNet50audioset',   'ResNet50multitask',
				'AST',  'wav2vec', 'VGGish', 'S2T',  'sepformer']

# # Models below spectemp baseline (n=4)
# source_models = [  'DCASE2020', 'DS2',  'ZeroSpeech2020', 'metricGAN']

# In-house models (n=10)
# source_models = ['Kell2018word', 'Kell2018speaker',  'Kell2018music', 'Kell2018audioset', 'Kell2018multitask',
# 				'ResNet50word', 'ResNet50speaker', 'ResNet50music', 'ResNet50audioset',   'ResNet50multitask', ]
# source_models = ['ResNet50multitask', 'spectemp', 'Kell2018multitask']

# All inhouse models with music as well (which only has seed 1)
# source_models = ['Kell2018word', 'Kell2018speaker',  'Kell2018music', 'Kell2018audioset', 'Kell2018multitask',
# 				'ResNet50word', 'ResNet50speaker', 'ResNet50music', 'ResNet50audioset',   'ResNet50multitask',
# 				 'Kell2018wordSeed2', 'Kell2018speakerSeed2',  'Kell2018audiosetSeed2', 'Kell2018multitaskSeed2',
# 				'ResNet50wordSeed2', 'ResNet50speakerSeed2', 'ResNet50audiosetSeed2',  'ResNet50multitaskSeed2',]

# All inhouse models that have both seed 1 and seed 2 (ie exclusing music)
# source_models = ['Kell2018word', 'Kell2018speaker',  'Kell2018audioset', 'Kell2018multitask',
# 				'ResNet50word', 'ResNet50speaker', 'ResNet50audioset',  'ResNet50multitask',
# 				 'Kell2018wordSeed2', 'Kell2018speakerSeed2',  'Kell2018audiosetSeed2', 'Kell2018multitaskSeed2',
# 				'ResNet50wordSeed2', 'ResNet50speakerSeed2', 'ResNet50audiosetSeed2', 'ResNet50multitaskSeed2',]

# Just seed 2 models
# source_models = ['Kell2018wordSeed2', 'Kell2018speakerSeed2',  'Kell2018audiosetSeed2', 'Kell2018multitaskSeed2',
# 				'ResNet50wordSeed2', 'ResNet50speakerSeed2', 'ResNet50audiosetSeed2',  'ResNet50multitaskSeed2',]
# Seed 2 models with music
# source_models = ['Kell2018wordSeed2', 'Kell2018speakerSeed2',  'Kell2018music', 'Kell2018audiosetSeed2', 'Kell2018multitaskSeed2',
# 				'ResNet50wordSeed2', 'ResNet50speakerSeed2', 'ResNet50music', 'ResNet50audiosetSeed2',  'ResNet50multitaskSeed2',]
# source_models = ['Kell2018word','Kell2018wordClean',
# 				 'ResNet50word', 'ResNet50wordClean']
# All clean word models
# source_models = ['Kell2018wordClean', 'Kell2018wordCleanSeed2',
# 				 'ResNet50wordClean', 'ResNet50wordCleanSeed2']
# source_models = ['Kell2018word', 'Kell2018wordClean', 'Kell2018wordSeed2', 'Kell2018wordCleanSeed2',
# 				 'ResNet50word', 'ResNet50wordClean', 'ResNet50wordSeed2', 'ResNet50wordCleanSeed2']

print(f'---------- Target: {target} ----------')

if concat_over_models:  # assemble plots across models
	if target != 'NH2015-B2021':
		df_meta_roi = pd.read_pickle(join(DATADIR, 'neural', target, 'df_roi_meta.pkl'))

	# Decide whether we want to store plots
	if not save:
		SAVEDIR_CENTRALIZED = None
	
	if target == 'NH2015comp': # components

		### For plotting all models barplot (Figure 5) ###
		if plot_barplot_across_models:

			##### Best layer component predictions across models (independently selected layer) ####
			add_savestr = f''
			for sort_flag in ['performance', NH2015_all_models_performance_order]: #'performance', NH2015_all_models_performance_order
				for randnetw_flag in ['False', 'True']: # 'False', 'True'
					barplot_components_across_models(source_models=source_models,
													 target=target,
													 randnetw=randnetw_flag,
													 value_of_interest='median_r2_test',
													 yerr_type='median_r2_test_sem_over_it',
													 save=SAVEDIR_CENTRALIZED,
													 add_savestr=add_savestr,
													 include_spectemp=True,
													 sort_by=sort_flag,
													 add_in_spacing_bar=False,
													 ylim=[0,1],
													 box_aspect=0.8)

		### Scatterplot of Seed1 vs Seed2 ###
		if plot_scatter_across_models:
			add_savestr = f'_seed1-vs-seed2'

			for randnetw_flag in ['False', 'True']: # 'False', 'True'
				if randnetw_flag == 'False':
					ylim = [0.4, 1]
				else:
					ylim = [0, 1]

				scatter_components_across_models_seed(source_models=source_models,
												 target=target,
												 randnetw=randnetw_flag,
												 models1=['Kell2018word', 'Kell2018speaker',  'Kell2018audioset', 'Kell2018multitask',
														  'ResNet50word', 'ResNet50speaker', 'ResNet50audioset', 'ResNet50multitask',],
												 models2=['Kell2018wordSeed2', 'Kell2018speakerSeed2', 'Kell2018audiosetSeed2', 'Kell2018multitaskSeed2',
														'ResNet50wordSeed2', 'ResNet50speakerSeed2', 'ResNet50audiosetSeed2',  'ResNet50multitaskSeed2',],
												 value_of_interest='median_r2_test',
												 yerr_type='median_r2_test_sem_over_it',
												 save=SAVEDIR_CENTRALIZED,
												 add_savestr=add_savestr,
												 ylim=ylim,
												 xlim=ylim)



		### For plotting in-house models barplot (Figure 8A) ###
		if plot_barplot_across_inhouse_models:
			add_savestr = f'_task-grouped-ymin-0.2-empty-bar-inhouse-models_seed2'
			sort_flag_manual = ['Kell2018word', 'ResNet50word', 'Kell2018speaker', 'ResNet50speaker', 'Kell2018music', 'ResNet50music',
							   'Kell2018audioset','ResNet50audioset', 'Kell2018multitask','ResNet50multitask',]

			#### Best layer component predictions across models (independently selected layer) ####
			for sort_flag in [sort_flag_manual]: #'performance', NH2015_all_models_performance_order
				# Remember to make source_models align with sort_flag
				for randnetw_flag in ['False', 'True']: # 'False', 'True'
					barplot_components_across_models(source_models=source_models,
													 target=target,
													 randnetw=randnetw_flag,
													 value_of_interest='median_r2_test',
													 yerr_type='median_r2_test_sem_over_it',
													 save=SAVEDIR_CENTRALIZED,
													 add_savestr=add_savestr,
													 include_spectemp=True,
													 sort_by=sort_flag,
													 add_in_spacing_bar=True,
													 ylim=[0.2,1],
													 box_aspect=0.8) # 1?

		### For plotting in-house models barplot (Figure 9A) ###
		if plot_word_clean_models:
			source_model_lst = [['Kell2018word', 'Kell2018wordClean',
							 'ResNet50word', 'ResNet50wordClean'],
							 ['Kell2018wordSeed2', 'Kell2018wordCleanSeed2',
							 'ResNet50wordSeed2', 'ResNet50wordCleanSeed2']]

			#### Best layer component predictions across models (independently selected layer) ####
			for source_models in source_model_lst:
				# if all end with 'Seed2' then add savestr
				if all([x.endswith('Seed2') for x in source_models]):
					add_savestr = f'_seed2'
				else:
					add_savestr = f'_seed1'

				for sort_flag in [source_models]: #'performance'
					# Remember to make source_models align with sort_flag
					for randnetw_flag in ['False','True']: # 'False', 'True'
						barplot_components_across_models(source_models=source_models,
														 target=target,
														 randnetw=randnetw_flag,
														 value_of_interest='median_r2_test',
														 yerr_type='median_r2_test_sem_over_it',
														 save=SAVEDIR_CENTRALIZED,
														 include_spectemp=True,
														 sort_by=sort_flag,
														 add_in_spacing_bar=False,
														 ylim=[0, 1],
														 add_savestr=f'_word-clean-models{add_savestr}',
														 box_aspect=0.8)

				# Run associated stats
				compare_CV_splits_nit(source_models=source_models,
									  target=target,
									  save=save,
									  save_str=f'_word-clean-models{add_savestr}',
									  models1=source_models,
									  models2=source_models,
									  aggregation='CV-splits-nit-10',
									  randnetw=randnetw_flag,
									  )

		
		#### Scatter: comp1 vs comp2 predictivity across models (Figure 8B) ####
		if plot_scatter_comp_vs_comp:
			save_str = '_inhouse-models_symbols-no-err-seed2' # !!!!!!!!

			#### Best layer component predictions across models (independently selected layer) as scatters ####

			for randnetw_flag in ['False', 'True']: # 'False', 'True',
				if randnetw_flag == 'False':
					ylim = [0.5, 1]
				else:
					ylim = [0, 1]

				scatter_components_across_models(source_models=source_models,
												 target=target,
												 randnetw=randnetw_flag,
												 aggregation='CV-splits-nit-10',
												 save=SAVEDIR_CENTRALIZED,
												 save_str=save_str,
												 include_spectemp=False,
												 ylim=ylim,
												 value_of_interest='median_r2_test',
												 sem_of_interest='median_r2_test_sem_over_it')

				## Associated statistics - comp1 vs comp2 comparions for models of interest ##
				# compare_CV_splits_nit(source_models=source_models,
				# 					  target=target,
				# 					  save=save,
				# 					  save_str='all-models-bootstrap', # Here we just run all models such that all numbers are in the same place
				# 					  models1 = ['Kell2018word', 'Kell2018speaker',  'Kell2018music', 'Kell2018audioset', 'Kell2018multitask',
				# 	 			'ResNet50word', 'ResNet50speaker', 'ResNet50music', 'ResNet50audioset',   'ResNet50multitask',
				# 				'AST', 'wav2vec', 'DCASE2020', 'DS2', 'VGGish',  'ZeroSpeech2020', 'S2T', 'metricGAN', 'sepformer'],
				# 					  models2 = ['Kell2018word', 'Kell2018speaker',  'Kell2018music', 'Kell2018audioset', 'Kell2018multitask',
				# 	 			'ResNet50word', 'ResNet50speaker', 'ResNet50music', 'ResNet50audioset',   'ResNet50multitask',
				# 				'AST', 'wav2vec', 'DCASE2020', 'DS2', 'VGGish',  'ZeroSpeech2020', 'S2T', 'metricGAN', 'sepformer'],
				# 					  aggregation='CV-splits-nit-10',
				# 					  randnetw=randnetw_flag,)

				# For in-house (Seed2)
				compare_CV_splits_nit(source_models=source_models,
									  target=target,
									  save=save,
									  save_str=save_str,
									  models1 = source_models,
									  models2 = source_models,
									  aggregation='CV-splits-nit-10',
									  randnetw=randnetw_flag,
									  )


		#### Predicted versus actual components (independently selected layer - most frequent one) ####
		if plot_scatter_pred_vs_actual:
			savestr = ''

			for source_model in source_models:
				for randnetw_flag in ['False','True']:
					if source_model == 'spectemp' and randnetw_flag == 'True':
						continue

					scatter_NH2015comp_resp_vs_pred(source_model=source_model,
													target=target,
													randnetw=randnetw_flag,
													save=SAVEDIR_CENTRALIZED,
													add_savestr=savestr,
													generate_scatter_plot=True,
													obtain_partial_corr=True,)


		#### Analyze best layer (not independently selected, just the argmax layer) ####
		# for randnetw_flag in [ 'True']:
		# 	scatter_comp_best_layer_across_models(source_models=source_models, target=target,
		# 									 randnetw=randnetw_flag, aggregation='argmax',
		# 									 save=SAVEDIR_CENTRALIZED, save_str='_inhouse-models_symbols', ylim=[-0.02,1.02],
		# 									 value_of_interest='rel_pos',
		# 									 )

	
	elif target in ['NH2015', 'B2021']: # neural data, either Nh2015 or B2021

		# BARPLOTS ACROSS MODELS #
		if plot_barplot_across_models:
			for sort_flag in ['performance']: # 'performance' NH2015_all_models_performance_order B2021_all_models_performance_order
				for val_flag in ['median_r2_test_c', ]:
					for agg_flag in ['CV-splits-nit-10']:
						for randnetw_flag in ['False', ]: # 'False', 'True'
							barplot_across_models(source_models=source_models,
												  target=target,
												  roi=None,
												  save=SAVEDIR_CENTRALIZED,
												  randnetw=randnetw_flag,
												  aggregation=agg_flag,
												  value_of_interest=val_flag,
												  sort_by=sort_flag,
												  add_savestr=f'',
												  box_aspect=0.8)

		# SCATTER ACROSS MODELS (SEED1 VS SEED2) #
		if plot_scatter_across_models:

			for randnetw_flag in ['False', 'True']:
				if randnetw_flag == 'False':
					ylim = [0.6,0.8]
					xlim = [0.6,0.8]
				else:
					ylim = [0.0,0.6]
					xlim = [0.0,0.6]
				scatter_across_models(source_models=source_models,
									  models1=['Kell2018word', 'Kell2018speaker', 'Kell2018audioset', 'Kell2018multitask',
											   'ResNet50word', 'ResNet50speaker', 'ResNet50audioset',   'ResNet50multitask',],
									  models2=['Kell2018wordSeed2', 'Kell2018speakerSeed2', 'Kell2018audiosetSeed2', 'Kell2018multitaskSeed2',
											   'ResNet50wordSeed2', 'ResNet50speakerSeed2', 'ResNet50audiosetSeed2',  'ResNet50multitaskSeed2',],
									  target=target,
									  save=SAVEDIR_CENTRALIZED,
									  randnetw=randnetw_flag,
									  aggregation='CV-splits-nit-10',
									  value_of_interest='median_r2_test_c',
									  add_savestr=f'_seed1-vs-seed2_xl={xlim[0]}-{xlim[1]}_yl={ylim[0]}-{ylim[1]}',
									  ylim=ylim,
									  xlim=xlim,
									  )

		# STATS FOR BARPLOTS ACROSS MODELS (bootstrap across subjects)
		if stats_barplot_across_models:
			for val_flag in ['median_r2_test_c',]:
				for randnetw_flag in ['False','True']:
					compare_models_subject_bootstrap(source_models=source_models,
													 target=target,
													 save=save,
													 value_of_interest=val_flag,
													  save_str='all-models_subject-bootstrap',
													  models1=[ 'ResNet50multitask',],
													  models2=['Kell2018word', 'Kell2018speaker',  'Kell2018music', 'Kell2018audioset', 'Kell2018multitask',
															'ResNet50word', 'ResNet50speaker', 'ResNet50music', 'ResNet50audioset',   'ResNet50multitask',],
													  aggregation='CV-splits-nit-10',
													  randnetw=randnetw_flag, )

		# PERFORMANCE OF CLEAN SPEECH NETWORKS (2 SEEDS)
		if plot_word_clean_models:
			source_model_lst = [['Kell2018word', 'Kell2018wordClean',
							 'ResNet50word', 'ResNet50wordClean'],
							 ['Kell2018wordSeed2', 'Kell2018wordCleanSeed2',
							 'ResNet50wordSeed2', 'ResNet50wordCleanSeed2']]

			for source_models in source_model_lst:

				# if all the models end with Seed2, create add_savestr = '_seed2'
				if all([model.endswith('Seed2') for model in source_models]):
					add_savestr = '_seed2'
				else:
					add_savestr = '_seed1'

				for sort_flag in [source_models]: # 'performance' NH2015_all_models_performance_order B2021_all_models_performance_order
					for val_flag in ['median_r2_test_c', ]:
						for agg_flag in ['CV-splits-nit-10']:
							for randnetw_flag in ['False', 'True' ]: # 'False', 'True'

								barplot_across_models(source_models=source_models,
													  target=target,
													  roi=None,
													  save=SAVEDIR_CENTRALIZED,
													  randnetw=randnetw_flag,
													  aggregation=agg_flag,
													  value_of_interest=val_flag,
													  sort_by=sort_flag,
													  add_savestr=f'_word-clean-models{add_savestr}',
													  box_aspect=1)

				# Run according stats
				for val_flag in ['median_r2_test_c', ]:
					for randnetw_flag in ['False', 'True' ]: # 'False', 'True'
						compare_models_subject_bootstrap(source_models=source_models,
														 target=target,
														 save=save,
														 value_of_interest=val_flag,
														 save_str=f'_word-clean-models_subject-bootstrap{add_savestr}',
														 models1=source_models, # just compile everything in this csv
														 models2=source_models,
														 aggregation='CV-splits-nit-10',
														 randnetw=randnetw_flag,
														 include_spectemp=False)


		# ANATOMICAL SCATTER PLOTS
		if plot_anat_roi_scatter:
			if len(source_models) == 19:
				save_str = '_all-models'
			elif len(source_models) == 15:
				save_str = '_good-models'
			else:
				save_str = f'_{len(source_models)}-models'

			for val_flag in ['median_r2_test_c']:
				for non_primary_flag in ['Lateral','Posterior']: # ['Anterior', 'Lateral', 'Posterior'] # For other combs, run with "Posterior" and then 'Lateral','Posterior'
					for cond_flag in ['roi_label_general']:
						for collapse_flag in ['median']: # How we collapsed over the relative position value for each subject (median is the default)
							for randnetw_flag in ['True',]: # 'True', 'False'
								scatter_anat_roi_across_models(source_models=source_models,
															   target=target,
															   save=SAVEDIR_CENTRALIZED,
															   randnetw=randnetw_flag,
															   condition_col=cond_flag,
															   collapse_over_val_layer=collapse_flag,
															   primary_rois=['Anterior', ], # 'Primary',# For other combs, run with "Lateral" and then "Anterior"
															   non_primary_rois=[non_primary_flag],
															   annotate=False,
															   save_str=save_str,
															   value_of_interest=val_flag,
															   stats=True)

		
		## LOAD SCORE ACROSS LAYERS (FOR DIMENSIONALITY ANALYSIS -- migrated to DIM_plot_main)
		# load_score_across_layers_across_models(source_models=source_models,
		# 									   RESULTDIR_ROOT=RESULTDIR_ROOT,)

		# GENERATE A MEDIAN SURFACE OF ARGMAX LAYER POSITION ACROSS MODELS
		if median_surface_across_models:

			if not save:
				SURFDIR = False

			# Create median surface across models
			if len(source_models) == 15: # good models!
				for val_flag in ['median_r2_test_c']:
					for randnetw_flag in ['False', 'True']:
						df_median_model_surf = create_avg_model_surface(source_models=source_models,
																		target=target,
																		PLOTSURFDIR=PLOTSURFDIR,
																		val_of_interest=val_flag,
																		randnetw=randnetw_flag,
																		plot_val_of_interest='rel_pos', # When we take the median across models, we need to operate in the relative layer position space
																		quantize=False)

						df_median_model_surf_quantized = create_avg_model_surface(source_models=source_models,
																		target=target,
																		PLOTSURFDIR=PLOTSURFDIR,
																		val_of_interest=val_flag,
																		randnetw=randnetw_flag,
																		plot_val_of_interest='rel_pos',
																		quantize=True)

						dump_for_surface_writing_avg(median_subj=df_median_model_surf,
													 source_model='all-good-models',
													 SURFDIR=SURFDIR,
													 PLOTSURFDIR=PLOTSURFDIR,
													 randnetw=randnetw_flag,
													 subfolder_name=f'TYPE=subj-median-argmax-model-median_'
																	f'METRIC={val_flag}_PLOTVAL=rel_pos*10+1'
																	f'_{target}' )

						dump_for_surface_writing_avg(median_subj=df_median_model_surf_quantized,
													 source_model='all-good-models',
													 SURFDIR=SURFDIR,
													 PLOTSURFDIR=PLOTSURFDIR,
													 randnetw=randnetw_flag,
													 subfolder_name=f'TYPE=subj-median-argmax-model-median_'
																	f'METRIC={val_flag}_PLOTVAL=rel_pos*10+1-quantized'
																	f'_{target}' )


	elif target == 'NH2015-B2021':

		# DETERMINE COLOR SCALE FOR SURFACE MAPS
		if determine_surf_colorscale:
			for randnetw_flag in ['False']:
				determine_surf_layer_colorscale(target='NH2015-B2021',
												source_models=source_models,
												randnetw=randnetw_flag,
												save=PLOTSURFDIR)

	else:
		print(f'Target ({target}) not recognized')


if not concat_over_models:
	for source_model in source_models:
		print(f'\n######### MODEL: {source_model} ##########\n')
		#### Identifier information #####
		mapping = 'AUD-MAPPING-Ridge'
		alpha_str = '50'
		
		#### Paths (source model specific) ####
		RESULTDIR = (Path(f'{RESULTDIR_ROOT}/{source_model}/')).resolve()
		PLOTDIR = (Path(f'{RESULTDIR}/outputs/')).resolve()
		PLOTDIR.mkdir(exist_ok=True)
		if not save:
			PLOTDIR = False # if save is False, don't save

		# Load voxel meta
		df_meta_roi = pd.read_pickle(join(DATADIR, 'neural', target, 'df_roi_meta.pkl'))
		meta = df_meta_roi.copy(deep=True) # for adding plotting values

		#### LOAD DATA ####
		# Trained network
		output, output_folders = concat_dfs_modelwise(RESULTDIR=RESULTDIR,
													  mapping=mapping,
													  df_str='df_output',
													  source_model=source_model, target=target,
													  truncate=None,
													  randnetw='False')
		output_folders_paths = [join(RESULTDIR, x) for x in output_folders]
		
		# Concatenate ds for B2021 (IF NOT DONE YET)
		# concat_ds_B2021(source_model=source_model, output_folders_paths=output_folders_paths,
		# 				df_roi_meta=df_meta_roi, randnetw=randnetw)
		# assert_output_ds_match(output_folders_paths=output_folders_paths)
		
		# Ensure that r2 test corrected does exceed 1
		output['median_r2_test_c'] = output['median_r2_test_c'].clip(upper=1)
		output['mean_r2_test_c'] = output['mean_r2_test_c'].clip(upper=1)

		# Permuted network (does not exist for spectemp or init models)
		if source_model.endswith('init') or source_model == 'spectemp': #or source_model in source_models: # FOR NOW, LETS NOT PLOT RANDNETW
			output_randnetw = None
			output_folders_paths_randnetw = []
		else:
			output_randnetw, output_folders_randnetw = concat_dfs_modelwise(RESULTDIR=RESULTDIR,
																			mapping=mapping,
																			df_str='df_output',
													  						source_model=source_model, target=target,
																			truncate=None,
													  						randnetw='True')
			output_randnetw['median_r2_test_c'] = output_randnetw['median_r2_test_c'].clip(upper=1)
			output_randnetw['mean_r2_test_c'] = output_randnetw['mean_r2_test_c'].clip(upper=1)
			output_folders_paths_randnetw = [join(RESULTDIR, x) for x in output_folders_randnetw]

		##### First, check if have spectemp. Then, only aggregate the data and do not make more plots #####
		if source_model == 'spectemp':
			if target == 'NH2015comp':
				val_flags = ['median_r2_test'] # Uncorrected r2
			else:
				val_flags = ['median_r2_test_c']

			print(f'Aggregating data for {source_model} model')
			for collapse_flag in ['median']:
				for val_flag in val_flags:
					obtain_spectemp_val_CV_splits_nit(roi=None,
													  target=target,
													  df_meta_roi=df_meta_roi,
													  collapse_over_splits=collapse_flag,
													  value_of_interest=val_flag,
													  nit=10)
		else: # Move on to all other possible analyses for models

			######### FIGURE OUT WHETHER WE HAVE NEURAL OR COMPONENT TARGET DATA ########
			if target in ['NH2015', 'B2021']:
				print(f'Plotting neural data: {target}')

				######### PREDICITIVITY ACROSS ALL LAYERS (SI 2) ########
				if pred_across_layers:

					# Plot predictivity across all layers, all voxels
					plot_score_across_layers(output=output,
											 output_randnetw=output_randnetw,
											 source_model=source_model,
											 target=target,
											 ylim=[0, 1],
											 roi=None,
											 save=PLOTDIR,
											 value_of_interest='median_r2_test_c',)

					# Create one-hot cols with ROI labels for plotting
					output = add_one_hot_roi_col(df=output,
												 col='roi_label_general',)
					# Plot predictivity across all layers, for anatomical ROIs (roi_label_general)
					plot_score_across_layers(output=output,
											 output_randnetw=output_randnetw,
											 source_model=source_model,
											 target=target,
											 ylim=[0, 1],
											 roi='roi_label_general',
											 save=PLOTDIR,
											 value_of_interest='median_r2_test_c',)
					sys.stdout.flush()

				######### FIND BEST LAYER USING INDEPENDENT CV SPLITS (BASIS FOR FIGURE 2) ########
				if best_layer_cv_nit:

					# Best layer based on CV splits -- TRAINED and PERMUTED NETWORK
					for collapse_flag in ['median']:
						for val_flag in ['r2_test_c']:
							for randnetw_flag in ['False', 'True']:
								if randnetw_flag == 'True':
									if output_randnetw is not None:
										select_r2_test_CV_splits_nit(output_folders_paths=output_folders_paths_randnetw,
																	 df_meta_roi=df_meta_roi,
																	 collapse_over_splits=collapse_flag,
																	 source_model=source_model, target=target,
																	 value_of_interest=val_flag,
																	 randnetw='True', roi=None, save=PLOTDIR, nit=10)
									else:
										print('No permuted network data found')
								elif randnetw_flag == 'False':
									select_r2_test_CV_splits_nit(output_folders_paths=output_folders_paths,
																 df_meta_roi=df_meta_roi,
																 collapse_over_splits=collapse_flag,
																 source_model=source_model, target=target,
																 value_of_interest=val_flag,
																 randnetw='False', roi=None, save=PLOTDIR, nit=10)
								else:
									raise ValueError()
							sys.stdout.flush()

				######### OBTAIN BEST LAYER PER ANATOMICAL ROI (BASIS FOR FIGURE 7) ########
				if best_layer_anat_ROI:
					# Barplots of best layer for anatomical ROIs -- TRAINED and PERMUTED NETWORK
					for cond_flag in ['roi_label_general']:  # ['roi_label_general','roi_anat_hemi' ]
						for collapse_flag in ['median']:  # ['median', 'mean'] # which aggfunc to use when obtaining an aggregate over rel_pos layer index values for each subject.
							# For the paper, we take the median over rel_pos (relative position) layer index values for each subject. Then, we take the mean over subjects.
							for val_flag in ['median_r2_test_c', ]:  # ['median_r2_test', 'median_r2_test_c',]
								for randnetw_flag in ['False', 'True']:  # ['False', 'True',]

									if randnetw_flag == 'True' and output_randnetw is not None:

										barplot_best_layer_per_anat_ROI(output=output_randnetw,
																		meta=meta,
																		source_model=source_model, target=target,
																		randnetw=randnetw_flag,
																		collapse_over_val_layer=collapse_flag,
																		save=PLOTDIR, condition_col=cond_flag,
																		value_of_interest=val_flag,
																		val_layer='rel_pos')
									elif randnetw_flag == 'True' and output_randnetw is None:
										print('No permuted network data found')
									elif randnetw_flag == 'False':
										barplot_best_layer_per_anat_ROI(output=output, meta=meta,
																		source_model=source_model, target=target,
																		randnetw=randnetw_flag,
																		collapse_over_val_layer=collapse_flag,
																		save=PLOTDIR, condition_col=cond_flag,
																		value_of_interest=val_flag,
																		val_layer='rel_pos', )
									else:
										raise ValueError()

				######### SURFACE ANALYSES (BASIS FOR FIGURE 6) ########
				if run_surf_argmax:
					PLOTSURFDIR.mkdir(exist_ok=True)

					if not save:
						PLOTSURFDIR = False  # if save is False, then do not save the plots
						SURFDIR = False

					######## Surface argmax plots #########
					for randnetw_flag in ['False', 'True']:

						if randnetw_flag == 'True' and output_randnetw is not None:
							output_to_use = output_randnetw
						elif randnetw_flag == 'True' and output_randnetw is None:
							print(f'No permuted network data found for {target}, source model {source_model}')
							continue
						elif randnetw_flag == 'False':
							output_to_use = output
						else:
							raise ValueError()

						for val_flag in ['median_r2_test_c', ]:
							for plot_val_flag in ['pos', 'rel_pos']:  # pos is used for plotting in MATLAB.

								## SUBJECT-WISE ARGMAX ANALYSIS ##
								# First, obtain the argmax layer for each voxel
								df_plot, layer_names = surface_argmax(output=output_to_use,
																	  source_model=source_model,
																	  target=target,
																	  randnetw=randnetw_flag,
																	  value_of_interest=val_flag,
																	  hist=True,
																	  save=PLOTSURFDIR)

								# Second, dump subject-wise mat files
								dump_for_surface_writing(vals=df_plot[plot_val_flag],
														 meta=meta,
														 source_model=source_model,
														 SURFDIR=SURFDIR,
														 randnetw=randnetw_flag,
														 subfolder_name=f'TYPE=subj-argmax_'
																		f'METRIC={val_flag}_'
																		f'PLOTVAL={plot_val_flag}_'
																		f'{target}')

								# Third, obtain a median subject surface (the function is called avg, but default is median to retain discrete layer position values)
								median_subj = create_avg_subject_surface(df_plot=df_plot,
																		 meta=meta,
																		 source_model=source_model,
																		 save=PLOTSURFDIR,
																		 target=target,
																		 val_of_interest=val_flag,
																		 plot_val_of_interest=plot_val_flag,
																		 randnetw=randnetw_flag)

								# Dump the average (median across subjects) brain to the surface
								dump_for_surface_writing_avg(median_subj=median_subj,
															 source_model=source_model,
															 SURFDIR=SURFDIR,
															 randnetw=randnetw_flag,
															 subfolder_name=f'TYPE=subj-median-argmax_'
																			f'METRIC={val_flag}_'
																			f'PLOTVAL={plot_val_flag}_'
																			f'{target}')



				##### Generate plots by plotting certain values of interest directly on the surface (not used in paper) #####
				if run_surf_direct_val:
					PLOTSURFDIR.mkdir(exist_ok=True)

					if not save:
						PLOTSURFDIR = False  # if save is False, then do not save the plots
						SURFDIR = False

					# Not used in paper, but can come in handy
					val_flags = ['kell_r_reliability', 'roi_label_general', 'pearson_r_reliability', 'shared_by']
					val_flags = ['kell_r_reliability']

					# Transform values
					for val_flag in val_flags:
						# Transformations
						if val_flag.endswith('reliability'):
							val_flag_to_plot = f'{val_flag}*10'
						elif val_flag == 'roi_label_general':
							val_flag_to_plot = 'roi_label_general_int'
						else:
							val_flag_to_plot = val_flag

						df_plot_direct = direct_plot_val_surface(output=output,
																 df_meta_roi=df_meta_roi,
																 val=val_flag, )

						# Make sure we take the median across shared coordinates across subjects!
						df_plot_direct_median = create_avg_subject_surface(df_plot=df_plot_direct,
																		   source_model=source_model,
																		   val_of_interest=val_flag_to_plot,
																		   # does not really matter here besides for logging..
																		   meta=meta,
																		   save=PLOTSURFDIR,
																		   target=target,
																		   randnetw='False',
																		   plot_val_of_interest=val_flag_to_plot, )

						# Dump to mat file
						dump_for_surface_writing_direct(df_plot_direct=df_plot_direct_median,
														val='median_plot_val',  # val_flag_to_plot,
														source_model=source_model,
														SURFDIR=SURFDIR,
														randnetw='False',
														subfolder_name=f'TYPE=subj-median-direct_'
																	   f'PLOTVAL={val_flag_to_plot}_'
																	   f'{target}')

					sys.stdout.flush()

				######### MERGE NH2015 AND B2021 TO GET THE LIMITS OF THE COLORSCALE (FOR FIGURE 6) ########
				if run_surf_argmax_merge_datsets:
					# Run surface_argmax() for each of NH2015 and B2021, and then get a histogram of layer preference for
					# voxels across both datasets (to determine color scale for surface plotting). Otherwise, this flag is
					# not used for anything else.

					# We are only interested in doing this for trained networks.

					# This flag has to be run with target='NH2015' and then we will load 'B2021' data and merge the two.
					assert(target == 'NH2015')
					PLOTSURFDIR.mkdir(exist_ok=True)

					if not save:
						PLOTSURFDIR = False  # if save is False, then do not save the plots
						SURFDIR = False

					# Load B2021 data
					# Load voxel meta
					df_meta_roi_B2021 = pd.read_pickle(join(DATADIR, 'neural', 'B2021', 'df_roi_meta.pkl'))
					meta_B2021 = df_meta_roi_B2021.copy(deep=True)  # for adding plotting values

					#### LOAD DATA ####
					# Trained network
					output_B2021, output_folders_B2021 = concat_dfs_modelwise(RESULTDIR=RESULTDIR,
																  mapping=mapping,
																  df_str='df_output',
																  source_model=source_model, target='B2021',
																  truncate=None,
																  randnetw='False')
					output_folders_paths_B2021 = [join(RESULTDIR, x) for x in output_folders_B2021]

					# Ensure that r2 test corrected does exceed 1 (just like for NH2015)
					output_B2021['median_r2_test_c'] = output_B2021['median_r2_test_c'].clip(upper=1)
					output_B2021['mean_r2_test_c'] = output_B2021['mean_r2_test_c'].clip(upper=1)

					## Obtain the df_plot with argmax values for each dataset

					for val_flag in ['median_r2_test_c']:

						# First, obtain the argmax layer for each voxel in NH2015
						df_plot_NH2015, layer_names_NH2015 = surface_argmax(output=output,
															  source_model=source_model,
															  target=target,
															  randnetw='False',
															  value_of_interest=val_flag,
															  hist=True,
															  save=False) # We already stored this using the run_surf_argmax flag

						# Then, obtain the argmax layer for each voxel in B2021
						df_plot_B2021, layer_names_B2021 = surface_argmax(output=output_B2021,
																		  source_model=source_model,
																		  target='B2021',
																		  randnetw='False',
																		  value_of_interest=val_flag,
																		  hist=True,
																		  save=False) # We already stored this using the run_surf_argmax flag when running target='B2021'

						assert(layer_names_NH2015 == layer_names_B2021).all()

						# Merge the two datasets and store the histogram of layer preference for each voxel
						surface_argmax_hist_merge_datasets(df_plot1=df_plot_NH2015,
														   df_plot2=df_plot_B2021,
														   source_model=source_model,
														   target='NH2015-B2021',
														   layer_names=layer_names_NH2015,
														   randnetw='False',
														   save=PLOTSURFDIR,
														   save_full_idxmax_layer=True,
														   value_of_interest=val_flag)

					sys.stdout.flush()


			elif target == 'NH2015comp':  # components
				print(f'Plotting component data: {target}')

				######### FIND BEST LAYER USING INDEPENDENT CV SPLITS (BASIS FOR FIGURE 5) ########
				if best_layer_cv_nit:
					for randnetw_flag in ['False','True']: # ['False', 'True']:
						if randnetw_flag == 'True':
							if output_randnetw is not None:
								select_r2_test_CV_splits_nit(output_folders_paths=output_folders_paths_randnetw,
															 df_meta_roi=df_meta_roi,
															 source_model=source_model,
															 target=target,
															 value_of_interest='r2_test',
															 randnetw=randnetw_flag,
															 save=PLOTDIR)
							else:
								print(f'No permuted network data found for NH2015comp, source model {source_model}')
						elif randnetw_flag == 'False':
							select_r2_test_CV_splits_nit(output_folders_paths=output_folders_paths,
														 df_meta_roi=df_meta_roi,
														 source_model=source_model,
														 target=target,
														 value_of_interest='r2_test',
														 randnetw=randnetw_flag,
														 save=PLOTDIR)
						else:
							raise ValueError()

				# # Per component, find the best layer and obtain the associated r2 test score (stores the 'best-layer-argmax_per-comp_{source_model}_NH2015comp_{value_of_interest}.csv')
				# # Trained and permuted:
				# for randnetw_flag in ['False','True']:
				# 	obtain_best_layer_per_comp(source_model=source_model, target=target, randnetw=randnetw_flag,
				# 							   value_of_interest='median_r2_test', sem_of_interest='sem_r2_test', )
				# 	obtain_best_layer_per_comp(source_model=source_model, target=target, randnetw=randnetw_flag,
				# 							   value_of_interest='median_r2_train', sem_of_interest='sem_r2_train', )
				#

			else:
				raise ValueError('Target not available')

	sys.stdout.flush()

