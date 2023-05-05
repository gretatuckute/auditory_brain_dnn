from aud_dnn.resources import d_layer_reindex, d_randnetw
import numpy as np
import pickle
import pandas as pd

def load_rsa_scores_across_layers_across_models(source_models,
												target='NH2015',
												value_of_interest='all_data_rsa_for_best_layer',
												roi=None,
												randnetw='False',
												agg_method='mean',
												RESULTDIR_ROOT='/rdma/vast-rdma/vast/mcdermott/jfeather/projects/aud-dnn/results/rsa_analysis',
												add_savestr=''):
	"""
	Loads the rsa score across layers for a list of source models, for a given target, ROI, and randnetw specification.

	Loop across models and compile a dataframe for each model that aggregates (default: mean) across subjects and stores the
	dataframe in a dictionary with the key = source model.
	
	# Options for the "value of interest"

	'best_layer_rsa_median_across_splits' : Used for Figure 2. Finds the best layer with "training" data split and
	        measures the RSA value with "test" data split.
	'all_data_rsa_for_best_layer' : RSA for the layer using all 165 sounds. Used for the "best layer" analysis
	        in Figure 5 where we only look at the argmax and do not plot the value. Also can be used to show
	        the RSA value for each layer.
	'splits_median_distance_all_layers' : For each layer, uses the splits for Figure 2 and computes the median value
	        across the test split for each layer (training data is unused). Generally not used, but evaluated to
	        have a comparison with the same number of sounds used for the measurements as the regression analysis.

	
	### Example
	source_models = ['ResNet50multitask', 'Kell2018word']
	load_rsa_scores_across_layers_across_models(source_models, roi=None, value_of_interest='all_data_rsa_for_best_layer')
	
	Args:
		source_models (list): list of source models to load
		target (str): target to load
		value_of_interest: key to load from each participant rsa dictionary
		roi (str or None): ROI to load
		randnetw (str): whether to load randnetw or not
		agg_method (str): method to aggregate the data (i.e. which aggregation to perform across subjects)
		save (bool): whether to save the data or not
		RESULTROOT (str): path to the results root
		add_savestr (str): string to add to the save path

	Returns:
		d_across_models: dictionary with the dataframes for each source model (key = source model, value = df with rows as layers and
			columns with the meaned (agg_method) across subjects) if value_of_interest == 'all_data_rsa_for_best_layer'
			if value_of_interest == 'best_layer_rsa_median_across_splits' then this is a dict (key = source model, value meaned across subjects)
			
		d_df_across_models: dict of dataframes for each source model. Key = source_model. Value = df with index as the subj idx
		 	and columns are value_of_interest for each layer (if value_of_interest == 'all_data_rsa_for_best_layer')
		 	if value_of_interest == 'best_layer_rsa_median_across_splits' then we don't have layers, and hence we return
		 	a dataframe with rows as participants and columns the best layer score
	"""
	
	if roi == None:
		fname = f'{RESULTDIR_ROOT}/all_dataset_rsa_dict.pckl'
	else:  # This has the ROI specific scores
		fname = f'{RESULTDIR_ROOT}/best_layer_rsa_analysis_dict.pckl'
	
	try:
		with open(fname, 'rb') as f:
			rsa_data_dictionary = pickle.load(f)
	except FileNotFoundError:
		raise FileNotFoundError(f'RSA analysis for ROI: {roi} does not have a file {fname}')
	
	if roi == None:
		if randnetw == 'False':
			train_or_perm = 'trained'
		else:
			train_or_perm = 'permuted'
		all_layer_rsa_dict = rsa_data_dictionary[target][train_or_perm]
	else:
		if randnetw == 'False':
			train_or_perm = 'rsa_analysis_dict_all_rois'
		else:
			train_or_perm = 'rsa_analysis_dict_all_rois_permuted'
		all_layer_rsa_dict = rsa_data_dictionary[train_or_perm][target][roi]
	
	d_across_models = {}
	d_df_across_models = {}
	lst_df_across_models = [] # for saving best layer scores as a dataframe in the end
	
	for source_model in source_models:
		all_values_of_interest = []
		
		for p, p_info in all_layer_rsa_dict[source_model].items():
			model_layers_participant = p_info['model_layers']
			
			assert model_layers_participant == d_layer_reindex[
				source_model], 'Check model ordering, reordering not implemented.'
			
			all_values_of_interest.append(p_info[value_of_interest])
		
		# Array where each row is a participant, and each column is a value (if multiple)
		all_values_of_interest = np.array(all_values_of_interest)
		
		# Package into a dataframe and aggregate across participants
		
		if value_of_interest == 'all_data_rsa_for_best_layer': # We have layer-wise data!
		
			if agg_method == 'mean':
				df = pd.DataFrame(all_values_of_interest.mean(axis=0), index=model_layers_participant).rename(columns={0: f'{value_of_interest}_agg-{agg_method}{d_randnetw[randnetw]}'})
			elif agg_method == 'median':
				df = pd.DataFrame(all_values_of_interest.median(axis=0), index=model_layers_participant).rename(columns={0: f'{value_of_interest}_agg-{agg_method}{d_randnetw[randnetw]}'})
			else:
				raise ValueError(f'agg_method {agg_method} not implemented')
			
			# Log
			df['source_model'] = source_model
			df['target'] = target
			df['roi'] = roi
			df['value_of_interest'] = value_of_interest
			df['randnetw'] = randnetw
			df['agg_method'] = agg_method
			
			d_across_models[source_model] = df
			d_df_across_models[source_model] = pd.DataFrame(all_values_of_interest, columns=model_layers_participant) # add in model layer names as colnames
			
			# Log
			df['source_model'] = source_model
			df['target'] = target
			df['roi'] = roi
			df['value_of_interest'] = value_of_interest
			df['randnetw'] = randnetw
			df['agg_method'] = agg_method
			
			d_across_models[source_model] = df
			d_df_across_models[source_model] = pd.DataFrame(all_values_of_interest,
															columns=model_layers_participant)  # add in model layer names as colnames
			
		elif value_of_interest == 'best_layer_rsa_median_across_splits': # We don't have layers, only participants!
			
			if agg_method == 'mean':
				agg_across_subjects_val = pd.DataFrame({f'{value_of_interest}_agg-{agg_method}{d_randnetw[randnetw]}':
															all_values_of_interest.mean(axis=0)}, index=[0])
			elif agg_method == 'median':
				agg_across_subjects_val = pd.DataFrame({f'{value_of_interest}_agg-{agg_method}{d_randnetw[randnetw]}':
															all_values_of_interest.median(axis=0)}, index=[0])
				
			else:
				raise ValueError(f'agg_method {agg_method} not implemented')
			
			# Ultimately we want to end up with a dataframe with rows as participants and columns as the value of interest for each source model (d_df_across_models)
			d_across_models[source_model] = agg_across_subjects_val
			d_df_across_models[source_model] = pd.DataFrame(all_values_of_interest).rename(columns={0: f'{source_model}'}) # subjects as index
			
		else:
			raise ValueError(f'value_of_interest {value_of_interest} not implemented')
			
			

	return d_across_models, d_df_across_models


def package_RSA_best_layer_scores(source_models,
								  value_of_interest,
								  target1='NH2015',
								  target2='B2021',
								  roi=None,
								  randnetw='False',
								  agg_method='mean',
								  save=False,
								  RESULTDIR_ROOT=None
								  ):
	"""
	Wrapper for loading the best layer values for RSA and storing them in a dataframe for two targets of interest.

	:params
		source_models: list of strings, source models to load the RSA values from
		target1: string, target to load the RSA values from
		target2: string, target to load the RSA values from
		roi: string or None, region of interest to load the RSA values from
		randnetw: string, whether to load the RSA values from the random network version of the model
		agg_method: string, aggregation method to use for the RSA values (across subjects)
		save: string, whether to save the dataframe to a file
		
	:returns
	    df_merge: dataframe, rows as source models, column as value of interest for two targets
	"""
	
	## NH2015 ##
	d_across_models_rsa_NH2015, d_across_layers_rsa_NH2015 = load_rsa_scores_across_layers_across_models(
		source_models=source_models,
		target=target1,
		roi=roi,
		randnetw=randnetw,
		value_of_interest=value_of_interest,
		agg_method=agg_method,
		RESULTDIR_ROOT=RESULTDIR_ROOT)
	
	## B2021 ##
	d_across_models_rsa_B2021, d_across_layers_rsa_B2021 = load_rsa_scores_across_layers_across_models(
		source_models=source_models,
		target=target2,
		roi=roi,
		randnetw=randnetw,
		value_of_interest=value_of_interest,
		agg_method=agg_method,
		RESULTDIR_ROOT=RESULTDIR_ROOT)
	
	# Obtain a subject X source_model dataframe from the d_across_layers_rsa_NH2015 dictionary and have column names as dict key (source_model)
	df_across_models_rsa_NH2015 = pd.concat([d_across_layers_rsa_NH2015[source_model] for source_model in source_models], axis=1)
	df_across_models_rsa_NH2015 = pd.DataFrame(df_across_models_rsa_NH2015.mean(axis=0)).rename(
		columns={0: f'{value_of_interest}_agg-{agg_method}'})  # average across subjects
	df_across_models_rsa_NH2015['target'] = target1
	
	df_across_models_rsa_B2021 = pd.concat([d_across_layers_rsa_B2021[source_model] for source_model in source_models],axis=1)
	df_across_models_rsa_B2021 = pd.DataFrame(df_across_models_rsa_B2021.mean(axis=0)).rename(
		columns={0: f'{value_of_interest}_agg-{agg_method}'})  # average across subjects
	df_across_models_rsa_B2021['target'] = target2
	
	# Average over subjects and get a df with rows as source model and columns as value of interest and metadata
	
	df_merge = pd.concat([df_across_models_rsa_NH2015, df_across_models_rsa_B2021], axis=0)
	
	# Sort the dataframe by the index (source model), alphabetically
	df_merge = df_merge.sort_index()
	
	# Log
	df_merge['source_model'] = df_merge.index
	df_merge.reset_index(drop=True, inplace=True)
	df_merge['value_of_interest'] = value_of_interest
	df_merge['roi'] = roi
	df_merge['randnetw'] = randnetw
	df_merge['agg_method'] = agg_method
	df_merge['RESULTDIR_ROOT'] = RESULTDIR_ROOT
	
	return df_merge
	
