from plot_utils_DIM import *

"""
Script for:

1. comparing regression/RSA consistency between datasets and among methods
Runs both layer-wise comparisons and model-wise comparisons

2. Compare best layer values between regression/RSA as used in Fig 5

For permuted plots, SpectroTemporal is excluded.

"""

### Directories ###
# Logging
date = datetime.datetime.now().strftime("%m%d%Y-%T")
if user != 'gt':
	sys.stdout = open(join(RESULTDIR_ROOT, 'logs', f'out-{date}.log'), 'a+')

### Settings ###

agg_method = 'mean'  # across subjects
roi = None

regr = True
rsa = True
modelwise_scatter = True # Compare modelwise between datasets and methods
layerwise_scatter = False # Compare layerwise between datasets and methods
best_layer_scatter = False # Compare best layer Fig 7 values between methods

save = False
if not save:
	SAVEDIR_CENTRALIZED = False

if modelwise_scatter:
	if regr:
		# ##### REGRESSION #####
		val_of_interest = 'median_r2_test_c'
		
		# SCATTER OF THE BARPLOT VALUES (FIGURE 2; CV SPLITS NIT 10) ACROSS MODELS, NH2015 VS B2021
		source_models = FIG_2_5_MODEL_LIST

		for ylim in [((-0.015,1)), None]:

			df_across_models_regr = modelwise_scores_across_targets(source_models=source_models,
																	value_of_interest=val_of_interest,
																	target1='NH2015', target2='B2021',
																	target1_loadstr_suffix='_performance_sorted',
																	target2_loadstr_suffix='_performance_sorted',
																	RESULTDIR_ROOT=RESULTDIR_ROOT,
																	roi=roi,
																	aggregation='CV-splits-nit-10',
																	add_savestr='_all-models',
																	ylim=ylim,
																	randnetw='False',
																	save=False)

			df_across_models_regr_randnetw = modelwise_scores_across_targets(source_models=[source_model for source_model in source_models if source_model != 'spectemp'],
																			value_of_interest=val_of_interest,
																			target1='NH2015', target2='B2021',
																			target1_loadstr_suffix='_performance_sorted',
																			target2_loadstr_suffix='_performance_sorted',
																			RESULTDIR_ROOT=RESULTDIR_ROOT,
																			roi=roi,
																			aggregation='CV-splits-nit-10',
																			add_savestr='_all-models',
																			ylim=ylim,
																			randnetw='True',
																			save=False)

		# LOOK INTO THE CORRELATION BETWEEN TRAINED AND PERMUTED MODELS
		for target in ['NH2015', 'B2021']:
			permuted_vs_trained_scatter(df_across_models_regr=df_across_models_regr,
										df_across_models_regr_randnetw=df_across_models_regr_randnetw,
										target=target,
										val_of_interest=f'{val_of_interest}_mean',
										save=SAVEDIR_CENTRALIZED)


	
	if rsa:
		##### RSA #####
		val_of_interest = 'best_layer_rsa_median_across_splits'
	
		df_across_models_rsa = package_RSA_best_layer_scores(source_models=source_models,
									  value_of_interest=val_of_interest,
									  target1='NH2015',
									  target2='B2021',
									  roi=None,
									  randnetw='False',
									  agg_method='mean',
									  save=False,
									  RESULTDIR_ROOT=join(RESULTDIR_ROOT, 'RSA'))
	
	df_across_models_rsa_randnetw = package_RSA_best_layer_scores(source_models=[source_model for source_model in source_models if source_model != 'spectemp'],
														 value_of_interest=val_of_interest,
														 target1='NH2015',
														 target2='B2021',
														 roi=None,
														 randnetw='True',
														 agg_method='mean',
														 save=False,
														 RESULTDIR_ROOT=join(RESULTDIR_ROOT, 'RSA'))
		
		
	if regr and rsa: # Compare model-wise scores
		
		val_of_interest_regr = f'median_r2_test_c_{agg_method}'
		regr_NH2015 = df_across_models_regr.query('target == "NH2015"').rename(columns={val_of_interest_regr: f'{val_of_interest_regr}_NH2015'}).set_index('source_model')
		regr_B2021 = df_across_models_regr.query('target == "B2021"').rename(columns={val_of_interest_regr: f'{val_of_interest_regr}_B2021'}).set_index('source_model')
		regr_NH2015_randnetw = df_across_models_regr_randnetw.query('target == "NH2015"').rename(columns={val_of_interest_regr: f'{val_of_interest_regr}_NH2015_randnetw'}).set_index('source_model')
		regr_B2021_randnetw = df_across_models_regr_randnetw.query('target == "B2021"').rename(columns={val_of_interest_regr: f'{val_of_interest_regr}_B2021_randnetw'}).set_index('source_model')
		
		val_of_interest_rsa = f'best_layer_rsa_median_across_splits_agg-{agg_method}'
		rsa_NH2015 = df_across_models_rsa.query('target == "NH2015"').rename(columns={val_of_interest_rsa: f'{val_of_interest_rsa}_NH2015'}).set_index('source_model')
		rsa_B2021 = df_across_models_rsa.query('target == "B2021"').rename(columns={val_of_interest_rsa: f'{val_of_interest_rsa}_B2021'}).set_index('source_model')
		rsa_NH2015_randnetw = df_across_models_rsa_randnetw.query('target == "NH2015"').rename(columns={val_of_interest_rsa: f'{val_of_interest_rsa}_NH2015_randnetw'}).set_index('source_model')
		rsa_B2021_randnetw = df_across_models_rsa_randnetw.query('target == "B2021"').rename(columns={val_of_interest_rsa: f'{val_of_interest_rsa}_B2021_randnetw'}).set_index('source_model')
		
		# Assert whether the source_models and source_model ordering are the same, and package into one dataframe
		assert (regr_NH2015.index.values == regr_B2021.index.values).all()
		assert (regr_NH2015_randnetw.index.values == regr_B2021_randnetw.index.values).all()
		assert (rsa_NH2015.index.values == rsa_B2021.index.values).all()
		assert (rsa_NH2015_randnetw.index.values == rsa_B2021_randnetw.index.values).all()
		assert (regr_NH2015.index.values == rsa_NH2015.index.values).all()
		assert (regr_NH2015_randnetw.index.values == rsa_NH2015_randnetw.index.values).all()
		assert (regr_B2021.index.values == rsa_B2021.index.values).all()
		assert (regr_B2021_randnetw.index.values == rsa_B2021_randnetw.index.values).all()
		
		
		df_across_models_regr_rsa = pd.concat([regr_NH2015[[f'{val_of_interest_regr}_NH2015']],
											   regr_B2021[[f'{val_of_interest_regr}_B2021']],
											   regr_NH2015_randnetw[[f'{val_of_interest_regr}_NH2015_randnetw']],
											   regr_B2021_randnetw[[f'{val_of_interest_regr}_B2021_randnetw']],
											   rsa_NH2015[[f'{val_of_interest_rsa}_NH2015']],
											   rsa_B2021[[f'{val_of_interest_rsa}_B2021']],
											   rsa_NH2015_randnetw[[f'{val_of_interest_rsa}_NH2015_randnetw']],
											   rsa_B2021_randnetw[[f'{val_of_interest_rsa}_B2021_randnetw']]], axis=1)
		
		df_across_models_regr_rsa_r = df_across_models_regr_rsa.corr()
		df_across_models_regr_rsa_r2 = df_across_models_regr_rsa_r.apply(lambda x: x**2)
		df_across_models_regr_rsa_p = df_across_models_regr_rsa.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*df_across_models_regr_rsa_r.shape)
		
		# Save as csvs!
		if save:
			df_across_models_regr_rsa.to_csv(f'{STATSDIR_CENTRALIZED}/df_across_models_regr_rsa_{datetag}.csv')
			df_across_models_regr_rsa_r.to_csv(f'{STATSDIR_CENTRALIZED}/df_across_models_regr_rsa_r_{datetag}.csv')
			df_across_models_regr_rsa_r2.to_csv(f'{STATSDIR_CENTRALIZED}/df_across_models_regr_rsa_r2_{datetag}.csv')
			df_across_models_regr_rsa_p.to_csv(f'{STATSDIR_CENTRALIZED}/df_across_models_regr_rsa_p_{datetag}.csv')
	

if layerwise_scatter:
	if regr:
		###### LOAD NEURAL PREDICTIVITY ACROSS LAYERS ######
		# We don't include spectemp for permuted.

		source_models = ALL_MODEL_LIST
		
		# ##### REGRESSION #####
		val_of_interest = 'median_r2_test_c'
		
		## NH2015 ##
		d_across_models_regr_NH2015, d_across_layers_regr_NH2015 = load_score_across_layers_across_models(
			source_models=source_models,
			target='NH2015',
			roi=roi,
			randnetw='False',
			value_of_interest=val_of_interest,
			agg_method=agg_method,
			save=False, # True or False, savedir is every model's output dir
			RESULTDIR_ROOT=RESULTDIR_ROOT)
		
		d_across_models_regr_NH2015_randnetw, d_across_layers_regr_NH2015_randnetw = load_score_across_layers_across_models(
			source_models=[source_model for source_model in source_models if source_model != 'spectemp'],
			target='NH2015',
			roi=roi,
			randnetw='True',
			value_of_interest=val_of_interest,
			agg_method=agg_method,
			save=False,
			RESULTDIR_ROOT=RESULTDIR_ROOT)
		
		## B2021 ##
		d_across_models_regr_B2021, d_across_layers_regr_B2021 = load_score_across_layers_across_models(
			source_models=source_models,
			target='B2021',
			roi=roi,
			randnetw='False',
			value_of_interest=val_of_interest,
			agg_method=agg_method,
			save=False, # True or False, savedir is every model's output dir
			RESULTDIR_ROOT=RESULTDIR_ROOT)
		
		d_across_models_regr_B2021_randnetw, d_across_layers_regr_B2021_randnetw = load_score_across_layers_across_models(
			source_models=[source_model for source_model in source_models if source_model != 'spectemp'],
			target='B2021',
			roi=roi,
			randnetw='True',
			value_of_interest=val_of_interest,
			agg_method=agg_method,
			save=False,
			RESULTDIR_ROOT=RESULTDIR_ROOT)
		
		# REGRESSION: SCATTER OF THE LAYERWISE SCORES (SI 1) ACROSS MODELS, NH2015 VS B2021
		
		for ylim in [((-0.015,1)), None]:
		
			# TRAINED
			layerwise_scores_across_targets(source_models=source_models,
											value_of_interest=val_of_interest,
											target1='NH2015',
											target2='B2021',
											d_across_layers_target1=d_across_layers_regr_NH2015,
											d_across_layers_target2=d_across_layers_regr_B2021,
											randnetw='False',
											yerr_type='within_subject_sem',
											ylim=ylim,
											add_savestr='_regr',
											save=SAVEDIR_CENTRALIZED)
			
			# PERMUTED
			layerwise_scores_across_targets(source_models=[source_model for source_model in source_models if source_model != 'spectemp'],
											value_of_interest=val_of_interest,
											target1='NH2015',
											target2='B2021',
											d_across_layers_target1=d_across_layers_regr_NH2015_randnetw,
											d_across_layers_target2=d_across_layers_regr_B2021_randnetw,
											randnetw='True',
											yerr_type='within_subject_sem',
											ylim=ylim,
											add_savestr='_regr',
											save=SAVEDIR_CENTRALIZED)
	
	if rsa:
		##### RSA #####
		val_of_interest = 'all_data_rsa_for_best_layer'
		
		## NH2015 ##
		d_across_models_rsa_NH2015, d_across_layers_rsa_NH2015 = load_rsa_scores_across_layers_across_models(
			source_models=source_models,
			target='NH2015',
			roi=roi,
			randnetw='False',
			value_of_interest=val_of_interest,
			agg_method=agg_method,
			RESULTDIR_ROOT=join(RESULTDIR_ROOT, 'RSA')) # Copied over Jenelle's results to overall resultdir
		
		d_across_models_rsa_NH2015_randnetw, d_across_layers_rsa_NH2015_randnetw = load_rsa_scores_across_layers_across_models(
			source_models=[source_model for source_model in source_models if source_model != 'spectemp'],
			target='NH2015',
			roi=roi,
			randnetw='True',
			value_of_interest=val_of_interest,
			agg_method=agg_method,
			RESULTDIR_ROOT=join(RESULTDIR_ROOT, 'RSA'))
			
		
		## B2021 ##
		d_across_models_rsa_B2021, d_across_layers_rsa_B2021 = load_rsa_scores_across_layers_across_models(
			source_models=source_models,
			target='B2021',
			roi=roi,
			randnetw='False',
			value_of_interest=val_of_interest,
			agg_method=agg_method,
			RESULTDIR_ROOT=join(RESULTDIR_ROOT, 'RSA'))
		
		d_across_models_rsa_B2021_randnetw, d_across_layers_rsa_B2021_randnetw = load_rsa_scores_across_layers_across_models(
			source_models=[source_model for source_model in source_models if source_model != 'spectemp'],
			target='B2021',
			roi=roi,
			randnetw='True',
			value_of_interest=val_of_interest,
			agg_method=agg_method,
			RESULTDIR_ROOT=join(RESULTDIR_ROOT, 'RSA'))
		
		# RSA: SCATTER OF THE LAYERWISE SCORES ACROSS MODELS, NH2015 VS B2021
		
		for ylim in [((-0.015,1)), (-0.015,0.8), None]:
		
			# TRAINED
			layerwise_scores_across_targets(source_models=source_models,
											value_of_interest=val_of_interest,
											target1='NH2015',
											target2='B2021',
											d_across_layers_target1=d_across_layers_rsa_NH2015,
											d_across_layers_target2=d_across_layers_rsa_B2021,
											randnetw='False',
											yerr_type='within_subject_sem',
											ylim=ylim,
											add_savestr='_rsa',
											save=SAVEDIR_CENTRALIZED)
			
			# PERMUTED
			layerwise_scores_across_targets(source_models=[source_model for source_model in source_models if source_model != 'spectemp'],
											value_of_interest=val_of_interest,
											target1='NH2015',
											target2='B2021',
											d_across_layers_target1=d_across_layers_rsa_NH2015_randnetw,
											d_across_layers_target2=d_across_layers_rsa_B2021_randnetw,
											randnetw='True',
											yerr_type='within_subject_sem',
											ylim=ylim,
											add_savestr='_rsa',
											save=SAVEDIR_CENTRALIZED)
	
	
	if regr and rsa: # Then we can compare between methods!
		######## REGRESSION VS RSA ###########
		
		for ylim in [((-0.015,1)), None]:

			# TRAINED, NH2015
			layerwise_scores_across_targets(source_models=source_models,
											value_of_interest='score',
											target1='NH2015_rsa',
											target2='NH2015_regr',
											d_across_layers_target1=d_across_layers_rsa_NH2015,
											d_across_layers_target2=d_across_layers_regr_NH2015,
											randnetw='False',
											yerr_type='within_subject_sem',
											ylim=ylim,
											add_savestr='',
											plot_identity=False,
											save=SAVEDIR_CENTRALIZED)
		
			# TRAINED, B2021
			layerwise_scores_across_targets(source_models=source_models,
											value_of_interest='score',
											target1='B2021_rsa',
											target2='B2021_regr',
											d_across_layers_target1=d_across_layers_rsa_B2021,
											d_across_layers_target2=d_across_layers_regr_B2021,
											randnetw='False',
											yerr_type='within_subject_sem',
											ylim=ylim,
											add_savestr='',
											plot_identity=False,
											save=SAVEDIR_CENTRALIZED)
			
			# PERMUTED, NH2015
			layerwise_scores_across_targets(source_models=[source_model for source_model in source_models if source_model != 'spectemp'],
											value_of_interest='score',
											target1='NH2015_rsa',
											target2='NH2015_regr',
											d_across_layers_target1=d_across_layers_rsa_NH2015_randnetw,
											d_across_layers_target2=d_across_layers_regr_NH2015_randnetw,
											randnetw='True',
											yerr_type='within_subject_sem',
											ylim=ylim,
											add_savestr='',
											plot_identity=False,
											save=SAVEDIR_CENTRALIZED)
			
			# PERMUTED, B2021
			layerwise_scores_across_targets(source_models=[source_model for source_model in source_models if source_model != 'spectemp'],
											value_of_interest='score',
											target1='B2021_rsa',
											target2='B2021_regr',
											d_across_layers_target1=d_across_layers_rsa_B2021_randnetw,
											d_across_layers_target2=d_across_layers_regr_B2021_randnetw,
											randnetw='True',
											yerr_type='within_subject_sem',
											ylim=ylim,
											add_savestr='',
											plot_identity=False,
											save=SAVEDIR_CENTRALIZED)

if best_layer_scatter:
	save_new = False
	###### FOR COMPARING THE BEST LAYER IN ANATOMICAL ROIS AS OBTAINED BY REGRESSION OR RSA #######
	
	#### Settings ####
	source_models = FIG_7_MODEL_LIST
	
	df_lst = []
	for target in ['NH2015', 'B2021']:
		for randnetw in ['False', 'True']:
			
			
			# across-layers-n=197-models_B2021_rsa-vs-B2021_regr_randnetw_scatter_roi-None_score_ylim-(-0.015, 1)
			save_str_analyze = f'across-models-n={len(source_models)}_' \
					   f'best-layer-corr_{target}_' \
					   f'rsa-vs-regr{d_randnetw[randnetw]}.csv'
			save_str_regr = f'across-models-n={len(source_models)}_' \
					   f'best-layer_{target}_' \
					   f'regr{d_randnetw[randnetw]}.csv'
			save_str_rsa = f'across-models-n={len(source_models)}_' \
					   f'best-layer_{target}_' \
					   f'rsa{d_randnetw[randnetw]}.csv'
			
			df_regr_final, df_rsa_final = loop_across_best_layer_regr_rsa(source_models=source_models,
																		  target=target,
																		  randnetw=randnetw,
																		  primary_flag='Primary',
																		  non_primary_flags=['Anterior', 'Lateral', 'Posterior'])
			
			assert (df_regr_final.source_model == df_rsa_final.source_model).all()
			assert (df_regr_final.ROI == df_rsa_final.ROI).all()
			ROIs = df_regr_final.ROI.unique()
			
			# Store the compiled best layer files for both regression and RSA
			if save_new:
				df_regr_final.to_csv(os.path.join(STATSDIR_CENTRALIZED, save_str_regr), index=False)
				df_rsa_final.to_csv(os.path.join(STATSDIR_CENTRALIZED, save_str_rsa), index=False)
			
	
			## Analyze ##
			
			# Overall correlation among num source models (15) * num ROIs (4)
			r, p = pearsonr(df_regr_final[f'mean_best_layer_regr'], df_rsa_final[f'mean_best_layer_rsa'])
			
			df = pd.DataFrame({'source_models': [source_models],
								'ROI': [ROIs],
								'num_in_comparison': [len(df_regr_final[f'mean_best_layer_regr'])],
								'target': [target],
								'randnetw': [randnetw],
								'r': [r],
								'r2': [r**2],
								'p': [p]})
			df_lst.append(df)
			
			# Run correlations within each of the values in the ROIs
			for ROI in ROIs:
				r_roi, p_roi = pearsonr(df_regr_final[df_regr_final.ROI == ROI][f'mean_best_layer_regr'],
										df_rsa_final[df_rsa_final.ROI == ROI][f'mean_best_layer_rsa'])
				df_roi = pd.DataFrame({'source_models': [source_models],
									   'ROI': [ROI],
									   'num_in_comparison': [len(df_regr_final[df_regr_final.ROI == ROI][f'mean_best_layer_regr'])],
									   'target': [target],
									   'randnetw': [randnetw],
									   'r': [r_roi],
									   'r2': [r_roi**2],
									   'p': [p_roi]})
				
				df_lst.append(df_roi)
			
	df_all = pd.concat(df_lst)
	if save_new:
		df_all.to_csv(os.path.join(STATSDIR_CENTRALIZED, save_str_analyze), index=False)

	# Post-hoc check of the reliability, i.e. how correlated are the best layer values within method, between datasets

	# Regression
	df_regr_NH2015 = pd.read_csv(join(STATSDIR_CENTRALIZED, 'across-models-n=15_best-layer_NH2015_regr.csv')) # '/Users/gt/Documents/GitHub/aud-dnn/results/20220121/Fig7_INFO_best-layer-corr_regr_vs_rsa/across-models-n=15_best-layer_B2021_regr.csv'
	df_regr_B2021 = pd.read_csv(join(STATSDIR_CENTRALIZED, 'across-models-n=15_best-layer_B2021_regr.csv')) # '/Users/gt/Documents/GitHub/aud-dnn/results/20220121/Fig7_INFO_best-layer-corr_regr_vs_rsa/across-models-n=15_best-layer_NH2015_regr.csv'

	r_regr, p_regr = pearsonr(df_regr_B2021['mean_best_layer_regr'].values, df_regr_NH2015['mean_best_layer_regr'].values)

	print(f'r_regr = {r_regr:.2f}, r2_regr = {r_regr**2:.2f}, p_regr = {p_regr:.2f}')

	# RSA
	df_rsa_NH2015 = pd.read_csv(join(STATSDIR_CENTRALIZED, 'across-models-n=15_best-layer_NH2015_rsa.csv')) # '/Users/gt/Documents/GitHub/aud-dnn/results/20220121/Fig7_INFO_best-layer-corr_regr_vs_rsa/across-models-n=15_best-layer_B2021_rsa.csv'
	df_rsa_B2021 = pd.read_csv(join(STATSDIR_CENTRALIZED, 'across-models-n=15_best-layer_B2021_rsa.csv')) # '/Users/gt/Documents/GitHub/aud-dnn/results/20220121/Fig7_INFO_best-layer-corr_regr_vs_rsa/across-models-n=15_best-layer_NH2015_rsa.csv'

	r_rsa, p_rsa = pearsonr(df_rsa_B2021['mean_best_layer_rsa'].values, df_rsa_NH2015['mean_best_layer_rsa'].values)

	print(f'r_rsa = {r_rsa:.2f}, r2_rsa = {r_rsa**2:.2f}, p_rsa = {p_rsa:.2f}')

			
			
			
