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
save = False # Whether to save any plots
concat_over_models = True

# If concat_over_models = False, we load each individual model and perform the analysis on that
if not concat_over_models:
	pred_across_layers = True # SI 2; predictivity for each model across all layers
	best_layer_cv_nit = True # Basis for Figure 2; obtain best layer for each voxel based on independent CV splits across 10 iterations
	best_layer_anat_ROI = True # Basis for Figure 7;

# deal w these
run_surf = False
merge_surface_targets = False


if concat_over_models:
	plot_barplot_across_models = False # Figure 2; barplot of performance across models
	stats_barplot_across_models = False # Figure 2; stats for barplot of performance across models
	plot_anat_roi_scatter = True # Figure 7; scatter of performance across models for anatomical ROIs



# Logging
date = datetime.datetime.now().strftime("%m%d%Y-%T")
if user != 'gt':
	sys.stdout = open(join(RESULTDIR_ROOT, 'logs', f'out-{date}.log'), 'a+')

source_models = [  'Kell2018word', 'Kell2018speaker',  'Kell2018music', 'Kell2018audioset', 'Kell2018multitask',
				 'ResNet50word', 'ResNet50speaker', 'ResNet50music', 'ResNet50audioset',   'ResNet50multitask',
				'AST',  'wav2vec', 'DCASE2020', 'DS2',   'ZeroSpeech2020', 'S2T', 'metricGAN', 'sepformer',
				 ] # 'spectemp' # 'VGGish' was missign rand! 'VGGish',
# source_models = [  'Kell2018word', 'Kell2018speaker',  'Kell2018music', 'Kell2018audioset', 'Kell2018multitask',
# 				 'ResNet50word', 'ResNet50speaker', 'ResNet50music', 'ResNet50audioset',   'ResNet50multitask',
# 				'AST',  'wav2vec', 'VGGish', 'S2T',  'sepformer']
# source_models = ['Kell2018word', 'Kell2018speaker',  'Kell2018music', 'Kell2018audioset', 'Kell2018multitask',
# 				'ResNet50word', 'ResNet50speaker', 'ResNet50music', 'ResNet50audioset',   'ResNet50multitask',]
# source_models = ['Kell2018word', 'Kell2018speaker',  'Kell2018music', 'Kell2018audioset', 'Kell2018multitask',
# 				'ResNet50word', 'ResNet50speaker', 'ResNet50music', 'ResNet50audioset',   'ResNet50multitask',
# 				 'Kell2018wordSeed2', 'Kell2018speakerSeed2',  'Kell2018audiosetSeed2', 'Kell2018multitaskSeed2',
# 				'ResNet50wordSeed2', 'ResNet50speakerSeed2', 'ResNet50audiosetSeed2',  'ResNet50multitaskSeed2',]
# source_models = [
# 				'AST','ASTSL01', 'ASTSL10',
# 				 'VGGish', 'VGGishSL01', 'VGGishSL10',
# 				 'sepformer', 'sepformerSL01',
# 				 'metricGAN', 'metricGANSL01',
# 				'metricGANSL10',
# 'Kell2018word', 'Kell2018speaker',  'Kell2018music', 'Kell2018audioset', 'Kell2018multitask',
# 				'ResNet50word', 'ResNet50speaker', 'ResNet50music', 'ResNet50audioset',   'ResNet50multitask',
# # 	'Kell2018wordSeed2', 'Kell2018speakerSeed2', 'Kell2018audiosetSeed2', 'Kell2018multitaskSeed2',
# # 	'ResNet50wordSeed2', 'ResNet50speakerSeed2', 'ResNet50audiosetSeed2', 'ResNet50multitaskSeed2',
# 			]

target = 'NH2015'

print(f'---------- Target: {target} ----------')

if concat_over_models:  # assemble plots across models
	if target != 'NH2015-B2021':
		df_meta_roi = pd.read_pickle(join(DATADIR, 'neural', target, 'df_roi_meta.pkl'))
	
	if target == 'NH2015comp': # components
		
		### For plotting in-house models barplot (Figure 6) ###
		# save_str = f'_task-grouped-ymin-0.2-empty-bar-new-spacing-inhouse-models'
		# # #### Best layer component predictions across models (independently selected layer) ####
		# for sort_flag in [['Kell2018word', 'ResNet50word', 'Kell2018speaker', 'ResNet50speaker', 'Kell2018music',  'ResNet50music',
		# 				   'Kell2018audioset','ResNet50audioset', 'Kell2018multitask','ResNet50multitask',]]: #'performance', NH2015_all_models_performance_order
		# 	for randnetw_flag in ['False', 'True']:
		# 		barplot_components_across_models(source_models=source_models, target=target, df_meta_roi=df_meta_roi,
		# 										 randnetw=randnetw_flag, value_of_interest='median_r2_test',
		# 										 sem_of_interest='median_r2_test_sem_over_it',
		# 										 save=SAVEDIR_CENTRALIZED, save_str=save_str, include_spectemp=True,
		# 										 sort_by=sort_flag, add_in_spacing_bar=True)
				
		### For plotting all models barplot (part of Figure 6) ###
		# save_str = f'_replot-alpha-1'
		# # #### Best layer component predictions across models (independently selected layer) ####
		# for sort_flag in [NH2015_all_models_performance_order]: #'performance', NH2015_all_models_performance_order
		# 	for randnetw_flag in ['True', 'False']: # 'False', 'True'
		# 		barplot_components_across_models(source_models=source_models, target=target, df_meta_roi=df_meta_roi,
		# 										 randnetw=randnetw_flag, value_of_interest='median_r2_test',
		# 										 sem_of_interest='median_r2_test_sem_over_it',
		# 										 save=SAVEDIR_CENTRALIZED, save_str=save_str, include_spectemp=True,
		# 										 sort_by=sort_flag, add_in_spacing_bar=False)

		#### Analyze best layer (not independently selected, just the argmax layer) ####
		# for randnetw_flag in [ 'True']:
		# 	scatter_comp_best_layer_across_models(source_models=source_models, target=target,
		# 									 randnetw=randnetw_flag, aggregation='argmax',
		# 									 save=SAVEDIR_CENTRALIZED, save_str='_inhouse-models_symbols', ylim=[-0.02,1.02],
		# 									 value_of_interest='rel_pos',
		# 									 )
		
		#### Scatter: comp1 vs comp2 predictivity across models ####
		# for randnetw_flag in [ 'False', 'True',]:
		# 	if randnetw_flag == 'False':
		# 		ylim = [0.5, 1]
		# 	else:
		# 		ylim = [0, 1]
		#
		# 	scatter_components_across_models(source_models=source_models, target=target, df_meta_roi=df_meta_roi,
		# 									 randnetw=randnetw_flag, aggregation='CV-splits-nit-10',
		# 									 save=SAVEDIR_CENTRALIZED, save_str='_inhouse-models_symbols-no-err', include_spectemp=False,ylim=ylim,
		# 									 value_of_interest='median_r2_test', sem_of_interest='median_r2_test_sem_over_it')
		#
		# 	# ## Associated statistics - comp1 vs comp2 comparions for models of interest ##
		# 	compare_CV_splits_nit(source_models=source_models, target=target, df_meta_roi=df_meta_roi,
		# 						  save=True, save_str='all-models-bootstrap',
		# 						  models1 = ['Kell2018word', 'Kell2018speaker',  'Kell2018music', 'Kell2018audioset', 'Kell2018multitask',
		# 		 			'ResNet50word', 'ResNet50speaker', 'ResNet50music', 'ResNet50audioset',   'ResNet50multitask',
		# 					'AST',  'wav2vec', 'DCASE2020', 'DS2', 'VGGish',  'ZeroSpeech2020', 'S2T', 'metricGAN', 'sepformer'],
		# 						  models2 = ['Kell2018word', 'Kell2018speaker',  'Kell2018music', 'Kell2018audioset', 'Kell2018multitask',
		# 		 			'ResNet50word', 'ResNet50speaker', 'ResNet50music', 'ResNet50audioset',   'ResNet50multitask',
		# 					'AST',  'wav2vec', 'DCASE2020', 'DS2', 'VGGish',  'ZeroSpeech2020', 'S2T', 'metricGAN', 'sepformer'],
		# 						  aggregation='CV-splits-nit-10',
		# 						  randnetw=randnetw_flag,)
			# compare_CV_splits_nit(source_models=source_models, target=target, save=True, save_str='inhouse-models_CochResNet50-bootstrap',
			# 					  models1=['ResNet50word', 'ResNet50speaker', 'ResNet50multitask','ResNet50audioset', 'ResNet50music'],
			# 					  models2=['ResNet50word', 'ResNet50speaker', 'ResNet50multitask','ResNet50audioset', 'ResNet50music'],
			# 					  aggregation='CV-splits-nit-10',
			# 					  randnetw=randnetw_flag, )
		
		
		#### Predicted versus actual components (independently selected layer - most frequent one) ####
		savestr = '_with_p-vals-TEST'
		for source_model in source_models:
			for randnetw_flag in ['False','True']:
				if source_model == 'spectemp' and randnetw_flag == 'True':
					continue
				scatter_NH2015comp_resp_vs_pred(source_model=source_model, target=target, df_meta_roi=df_meta_roi,
												randnetw=randnetw_flag, save=SAVEDIR_CENTRALIZED, add_savestr=savestr)
	
	elif target == 'NH2015-B2021':
		# DETERMINE COLOR SCALE FOR SURFACE MAPS
		# if len(source_models) == 19: # All
		# 	for randnetw_flag in ['False', 'True']:
		# 		determine_surf_layer_colorscale(target='NH2015-B2021', source_models=source_models, randnetw=randnetw_flag,
		# 										save=PLOTSURFDIR)
		print('Done!')

	
	elif target in ['NH2015', 'B2021']: # neural data, either Nh2015 or B2021

		# BARPLOTS ACROSS MODELS #
		if plot_barplot_across_models:
			for sort_flag in ['performance']:
				for val_flag in ['median_r2_test_c', ]:
					for agg_flag in ['CV-splits-nit-10']:
						for randnetw_flag in ['False']: # 'False', 'True'
							barplot_across_models(source_models=source_models,
												  target=target,
												  roi=None,
												  df_meta_roi=df_meta_roi,
												  save=SAVEDIR_CENTRALIZED,
												  randnetw=randnetw_flag,
												  aggregation=agg_flag,
												  value_of_interest=val_flag,
												  sort_by=sort_flag,
												  add_savestr=f'_{datetag}')

		# STATS FOR BARPLOTS ACROSS MODELS (bootstrap across subjects)
		if stats_barplot_across_models:
			for val_flag in ['median_r2_test_c',]:
				for randnetw_flag in ['False','True']:
					compare_models_subject_bootstrap(source_models=source_models, target=target, df_meta_roi=df_meta_roi,
													 save=True, value_of_interest=val_flag,
										  save_str='all-models_subject-bootstrap',
										  models1=[ 'ResNet50multitask',],
										  models2=['Kell2018word', 'Kell2018speaker',  'Kell2018music', 'Kell2018audioset', 'Kell2018multitask',
												'ResNet50word', 'ResNet50speaker', 'ResNet50music', 'ResNet50audioset',   'ResNet50multitask',],
										  aggregation='CV-splits-nit-10',
										  randnetw=randnetw_flag, )

		# ANATOMICAL SCATTER PLOTS
		if plot_anat_roi_scatter:
			save_str = '_good-models'
			for val_flag in ['median_r2_test_c']:
				for non_primary_flag in ['Anterior', 'Lateral', 'Posterior']:
					for cond_flag in ['roi_label_general']:
						for collapse_flag in ['median']: # How we collapsed over the relative position value for each subject (median is the default)
							for randnetw_flag in ['False']: # 'True', 'False'
								scatter_anat_roi_across_models(source_models=source_models,
															   target=target,
															   save=SAVEDIR_CENTRALIZED,
															   randnetw=randnetw_flag,
															   condition_col=cond_flag,
															   collapse_over_val_layer=collapse_flag,
															   primary_rois=['Primary'],
															   non_primary_rois=[non_primary_flag],
															   annotate=False,
															   save_str=save_str,
															   value_of_interest=val_flag)

		
		## LOAD SCORE ACROSS LAYERS (FOR DIMENSIONALITY ANALYSIS -- migrated to DIM_plot_main)
		# load_score_across_layers_across_models(source_models=source_models,
		# 									   RESULTDIR_ROOT=RESULTDIR_ROOT,)
					

		# # Create median surface across models
		# if len(source_models) == 15: # good models!
		# 	for val_flag in ['median_r2_test_c', 'median_r2_test']:
		# 		for randnetw_flag in ['False','True']:
		# 			df_median_model_surf = create_avg_model_surface(source_models, target, PLOTSURFDIR,
		# 															val_of_interest=val_flag, randnetw=randnetw_flag,
		# 															plot_val_of_interest='rel_pos',
		# 															quantize=False)
		# 			df_median_model_surf_quantized = create_avg_model_surface(source_models, target, PLOTSURFDIR,
		# 															  val_of_interest=val_flag, randnetw=randnetw_flag,
		# 															  plot_val_of_interest='rel_pos',
		# 															  quantize=True)
		# 			dump_for_surface_writing_avg(df_median_model_surf, source_model='all-good-models', SURFDIR=SURFDIR,
		# 										 randnetw=randnetw_flag, subfolder_name=f'TYPE=subj-median-argmax-model-median_METRIC={val_flag}_PLOTVAL=rel_pos*10+1'
		# 																		f'_{target}' )
		# 			dump_for_surface_writing_avg(df_median_model_surf_quantized, source_model='all-good-models', SURFDIR=SURFDIR,
		# 										 randnetw=randnetw_flag, subfolder_name=f'TYPE=subj-median-argmax-model-median_METRIC={val_flag}_PLOTVAL=rel_pos*10+1-quantized'
		# 																		f'_{target}' )
	
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
		
		# spectemp_path = [f'{ROOT}/results/AUD/20210915_median_across_splits_correction/spectemp/AUD-MAPPING-Ridge_TARGET-B2021_SOURCE-spectemp-avgpool_RANDEMB-False_RANDNETW-False_ALPHALIMIT-50']
		# concat_ds_B2021(source_model='spectemp', output_folders_paths=spectemp_path,
		# 				df_roi_meta=df_meta_roi, randnetw=randnetw)

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

								if randnetw_flag == 'True':

									barplot_best_layer_per_anat_ROI(output_randnetw, meta, source_model=source_model,
																	target=target,
																	randnetw=randnetw_flag,
																	collapse_over_val_layer=collapse_flag,
																	save=PLOTDIR, condition_col=cond_flag,
																	value_of_interest=val_flag,
																	val_layer='rel_pos')
								elif randnetw_flag == 'False':
									barplot_best_layer_per_anat_ROI(output, meta, source_model=source_model, target=target,
																	randnetw=randnetw_flag,
																	collapse_over_val_layer=collapse_flag,
																	save=PLOTDIR, condition_col=cond_flag,
																	value_of_interest=val_flag,
																	val_layer='rel_pos', )
								else:
									raise ValueError()


			elif target == 'NH2015comp':  # components
				print(f'Plotting component data: {target}')

				# for randnetw_flag in ['False']:
				# 	if randnetw_flag == 'True':
				# 		select_r2_test_CV_splits_nit(output_folders_paths=output_folders_paths_randnetw, df_meta_roi=df_meta_roi, source_model=source_model,
				# 								 target=target, value_of_interest='r2_test', randnetw=randnetw_flag, save=PLOTDIR)
				# 	elif randnetw_flag == 'False':
				# 		select_r2_test_CV_splits_nit(output_folders_paths=output_folders_paths, df_meta_roi=df_meta_roi, source_model=source_model,
				# 								 target=target, value_of_interest='r2_test', randnetw=randnetw_flag, save=PLOTDIR)
				# 	else:
				# 		raise ValueError()
				
				
				# # Plot components across layers, and store 'across-layers_{source_model}_NH2015comp_{value_of_intereset}.csv'
				plot_comp_across_layers(output=output, source_model=source_model, output_randnetw=output_randnetw,
										target=target, save=PLOTDIR, value_of_interest='median_r2_test')
				plot_comp_across_layers(output=output, source_model=source_model, output_randnetw=output_randnetw,
										target=target, save=PLOTDIR, value_of_interest='median_r2_train')
				#
				# # Per component, find the best layer and obtain the associated r2 test score (stores the 'best-layer-argmax_per-comp_{source_model}_NH2015comp_{value_of_interest}.csv')
				# # Trained and permuted:
				# for randnetw_flag in ['False','True']:
				# 	obtain_best_layer_per_comp(source_model=source_model, target=target, randnetw=randnetw_flag,
				# 							   value_of_interest='median_r2_test', sem_of_interest='sem_r2_test', )
				# 	obtain_best_layer_per_comp(source_model=source_model, target=target, randnetw=randnetw_flag,
				# 							   value_of_interest='median_r2_train', sem_of_interest='sem_r2_train', )
				#
				# # Per component, select the best layer based on r2 train and get the r2 test value for that layer
				# # (stores the f'best-layer-CV-splits-train-test_per-comp_{source_model}_NH2015_{value_of_interest}.csv')
				# # Also use this info to denote which scatterplot (resp vs pred) layer is used.
				# for randnetw_flag in ['False','True']:
				# 	select_r2_test_CV_splits_train_test(source_model=source_model, target=target, randnetw=randnetw_flag, )
			

			else:
				raise ValueError('Target not available')


		######### SURFACE ANALYSES ########
		if run_surf:  # surf has to be run with randnetw False and True separately
			PLOTSURFDIR.mkdir(exist_ok=True)
			save_full_surface = True
			
			##### Generate plots by plotting certain values of interest directly on the surface #####
			val_flags = ['kell_r_reliability', 'roi_label_general', 'pearson_r_reliability', 'shared_by']
			
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
														 val=val_flag,)
				
				# Make sure we take the median across shared coordinates across subjects!
				df_plot_direct_median = create_avg_subject_surface(df_plot=df_plot_direct,
																   source_model=source_model,
																   val_of_interest=val_flag_to_plot, # does not really matter here besides for logging..
																   meta=meta,
																   save=PLOTSURFDIR,
																   target=target,
																   randnetw=randnetw,
																   plot_val_of_interest=val_flag_to_plot,
																   save_full_surface=True)
				
				# Dump to mat file
				dump_for_surface_writing_direct(df_plot_direct=df_plot_direct_median,
												val='median_plot_val',#val_flag_to_plot,
												 source_model=source_model,
												 SURFDIR=SURFDIR,
												 randnetw=randnetw,
												 subfolder_name=f'TYPE=subj-median-direct_PLOTVAL={val_flag_to_plot}_{target}')
			
			######## Surface argmax plots #########
			for val_flag in ['median_r2_test_c', 'median_r2_test']:
				for plot_val_flag in ['pos', 'rel_pos']:
					
					## SUBJECT-WISE ARGMAX ANALYSIS ##
					df_plot, layer_names = surface_argmax(output, source_model=source_model, target=target, randnetw=randnetw,
														  value_of_interest=val_flag, hist=True, save=PLOTSURFDIR)
					
					# Dump subject-wise mat files
					dump_for_surface_writing(vals=df_plot[plot_val_flag],
											 meta=meta,
											 source_model=source_model,
											 SURFDIR=False,  # SURFDIR
											 randnetw=randnetw,
											 subfolder_name=f'TYPE=subj-argmax_METRIC={val_flag}_PLOTVAL={plot_val_flag}_'
															f'{target}')
					
					## AVERAGE SUBJECT ##
					if merge_surface_targets:  # load the other target dataset
						# Load output results
						output2, _ = concat_dfs_modelwise(RESULTDIR, mapping=mapping, df_str='df_output',
														  source_model=source_model, target=d_target[target],
														  truncate=None, randemb=randemb, randnetw=randnetw)
						# Load meta
						df_meta_roi2 = pd.read_pickle(join(DATADIR, 'neural', d_target[target], 'df_roi_meta.pkl'))
						meta2 = df_meta_roi2.copy(deep=True)  # for adding plotting values
						meta2['target'] = d_target[target]
						
						if source_model == 'wav2vec':  # rename 'Logits' to Final
							output2.loc[output2.source_layer == 'Logits', 'source_layer'] = 'Final'
						
						# Ensure that r2 test corrected does exceed 1
						output2['median_r2_test_c'] = output2['median_r2_test_c'].clip(upper=1)
						
						df_plot2, layer_names2 = surface_argmax(output2, source_model=source_model, target=d_target[target],
																randnetw=randnetw, value_of_interest=val_flag, hist=True,
																save=PLOTSURFDIR)
						
						# Also save subjwise files for target 2
						dump_for_surface_writing(df_plot2[plot_val_flag], meta2, source_model, SURFDIR, randnetw=randnetw,
												 subfolder_name=f'TYPE=subj-argmax_METRIC={val_flag}_PLOTVAL={plot_val_flag}_'
																f'{d_target[target]}')
						
						# Store median subject surface map for secondary dataset
						median_subj2 = create_avg_subject_surface(df_plot=df_plot2, meta=meta2, source_model=source_model, save=PLOTSURFDIR,
																  target=d_target[target], val_of_interest=val_flag,
																  plot_val_of_interest=plot_val_flag, randnetw=randnetw, save_full_surface=True)
						dump_for_surface_writing_avg(median_subj2, source_model, SURFDIR, randnetw=randnetw,
													 subfolder_name=f'TYPE=subj-median-argmax_METRIC={val_flag}_PLOTVAL={plot_val_flag}_'
																	f'{d_target[target]}')
						
						assert (layer_names == layer_names2).all()
						
						# Median subj, based on both datasets
						meta['target'] = target
						median_subj = create_avg_subject_surface_merge_targets(df_plot1=df_plot, meta1=meta,
																			   df_plot2=df_plot2, meta2=meta2, plot_val_of_interest=plot_val_flag)
						
						# Plot histogram for layer preference across voxels from both datasets
						surface_argmax_hist_merge_datasets(df_plot1=df_plot, df_plot2=df_plot2, source_model=source_model,
														   save=PLOTSURFDIR, randnetw=randnetw, layer_names=layer_names)
						
						# Dump the average (median across subjects) brain to the surface
						dump_for_surface_writing_avg(median_subj, source_model, SURFDIR, randnetw=randnetw,
													 subfolder_name=f'TYPE=subj-median-argmax_METRIC={val_flag}_PLOTVAL={plot_val_flag}'
																	f'_{target}-{d_target[target]}')
					
					# Just use the first output to create the surface for primary target (independent of whether I am merging or not)
					median_subj1 = create_avg_subject_surface(df_plot=df_plot, meta=meta, source_model=source_model, save=PLOTSURFDIR,
																  target=target, val_of_interest=val_flag,
																  plot_val_of_interest=plot_val_flag, randnetw=randnetw, save_full_surface=True)
					
					# Dump the average (median across subjects) brain to the surface
					dump_for_surface_writing_avg(median_subj1, source_model, SURFDIR, randnetw=randnetw,
												 subfolder_name=f'TYPE=subj-median-argmax_METRIC={val_flag}_PLOTVAL={plot_val_flag}'
																f'_{target}')
					
					sys.stdout.flush()
		

	
			
			
			
	
		
	
