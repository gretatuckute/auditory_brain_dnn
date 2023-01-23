import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#%% Plot styles
plt.style.use('seaborn-pastel')
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rc('font',family='sans-serif')
plt.rcParams.update({'font.size':12})
matplotlib.rcParams['grid.alpha'] = 1
matplotlib.rcParams['xtick.labelsize'] = 'small'
matplotlib.rcParams['ytick.labelsize'] = 'medium'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

d_roi_colors = {'all': [sns.color_palette("pastel")[0], sns.color_palette("pastel")[2], sns.color_palette("pastel")[3], sns.color_palette("pastel")[4]],
			'any_roi': 'orange', 'none': 'darkslateblue',
			'tonotopic': sns.color_palette("pastel")[0],
			'pitch': sns.color_palette("pastel")[2],
			'music': sns.color_palette("pastel")[3],
			'speech': 'mediumorchid', # sns.color_palette("pastel")[4]
		 	'lowfreq': 'paleturquoise',
			'highfreq': 'cadetblue',
				'envsounds': 'navajowhite'}

d_target = {'NH2015': 'B2021', 'B2021': 'NH2015'}

d_annotate = {True: '_annotated',
			  False: ''}

d_randnetw = {'True': '_randnetw', 'False': ''}

d_roi_anat_colors = {'Anterior_rh':'gold',
					 'Anterior_lh':'gold',
					 'Primary_rh':'darkorange',
					 'Primary_lh':'darkorange',
					 'Lateral_rh':'tomato',
					 'Lateral_lh':'tomato',
					 'Posterior_rh':'firebrick',
					 'Posterior_lh':'firebrick',
					 'Anterior': 'gold',
					 'Primary': 'darkorange',
					 'Lateral': 'tomato',
					 'Posterior': 'firebrick',
					 }

d_model_colors = {'AST':'mediumorchid',
'DCASE2020':'gold',
'DS2':'#E8E869',
'Kell2018word':'#AD233B',
'Kell2018speaker':'#F13336',
'Kell2018music':'#F9671D',
'Kell2018audioset':'#FF80A5',
'Kell2018multitask': '#FFBAA5',#'orangered',
'metricGAN': 'sienna',
'ResNet50word':'darkgreen',
'ResNet50speaker':'#55D409',
'ResNet50music':'mediumseagreen',
'ResNet50audioset':'#80F396',
'ResNet50multitask': '#C3F380',
'S2T': 'blueviolet',
'sepformer': 'purple',
'spectemp':'grey',
'VGGish':'royalblue',
'wav2vec':'plum',
'ZeroSpeech2020':'skyblue',
'mock': 'grey',
}

d_sorted = {'sort_by_performance': '_performance_sorted',
			'sort_by_manual_list': '_manually_sorted',}

d_model_markers = {'Kell2018word':'s',
                   'Kell2018speaker':'D',
                   'Kell2018music':'x',
                   'Kell2018audioset':'+',
                   'Kell2018multitask': 'o',
                   'ResNet50word':'s',
                   'ResNet50speaker':'D',
                   'ResNet50music':'x',
                   'ResNet50audioset':'+',
                   'ResNet50multitask': 'o',}

d_comp_names = {'lowfreq': 'Low frequency',
				'highfreq': 'High frequency',
				'envsounds': 'Env. sound',
				'pitch': 'Pitch',
				'music': 'Music',
				'speech': 'Speech',}

d_model_names = {'AST':'AST',
				'DCASE2020':'DCASE2020',
				'DS2':'DeepSpeech2',
				'Kell2018word':'CochCNN9-Word',
				'Kell2018speaker':'CochCNN9-Speaker',
				'Kell2018music':'CochCNN9-Genre',
				'Kell2018audioset':'CochCNN9-AudioSet',
				'Kell2018multitask': 'CochCNN9-MultiTask',
				'metricGAN': 'MetricGAN',
				'ResNet50word':'CochResNet50-Word',
				'ResNet50speaker':'CochResNet50-Speaker',
				'ResNet50music':'CochResNet50-Genre',
				'ResNet50audioset':'CochResNet50-AudioSet',
				'ResNet50multitask': 'CochResNet50-MultiTask',
				'S2T': 'S2T',
				'sepformer': 'SepFormer',
				'spectemp':'SpectroTemporal',
				'VGGish':'VGGish',
				'wav2vec':'Wav2Vec2',
				 'wav2vecpower': 'Wav2Vec2-Power',
				 'ZeroSpeech2020':'VQ-VAE',
				 'mock':''}

d_value_of_interest = {'median_r2_test': 'Median $R^2$',
					   'median_r2_test_c': 'Median noise-corrected $R^2$',
						'r2_test_c': 'Median noise-corrected $R^2$',
					   'mean_r2_test_c': 'Mean noise-corrected $R^2$',
						'median_r2_train': 'Median $R^2$ train',
						'r2_test': 'Median $R^2$',
					   'rel_pos': 'Relative layer position',
					   'rel_layer_pos': 'Relative best layer',
					   'dim_demean-True': 'ED (trained model, demean=True)',
					   'dim_demean-False': 'ED (trained model, demean=False)',
					   'dim_randnetw_demean-True': 'ED (random network, demean=True)',
					   'dim_randnetw_demean-False': 'ED (random network, demean=False)',}

				
d_sound_category_colors = {'Music':'mediumblue',
						   'Song': 'deepskyblue',
						   'EngSpeech': 'darkgreen',
						   'ForSpeech': 'limegreen',
						  'HumVoc':'darkviolet',
						   'AniVoc': 'orchid',
						   'HumNonVoc': 'firebrick',
						   'AniNonVoc':'hotpink',
						  'Nature':'gold',
						   'Mechanical': 'darkorange',
						   'EnvSound':'grey',}

d_sound_category_names = {'Music':'Instr. Music',
						  'Song':'Vocal Music',
						  'EngSpeech':'English Speech',
						  'ForSpeech':'Foreign Speech',
						  'HumVoc':'NonSpeech Vocal',
						  'AniVoc':'Animal Vocal',
						  'HumNonVoc':'Human NonVocal',
						  'AniNonVoc':'Animal NonVocal',
						  'Nature':'Nature',
						  'Mechanical':'Mechanical',
						  'EnvSound':'Env. Sounds'}

# (Music: Music + Song, Speech: EngSpeech + ForSpeech, Vocal: HumVoc + AniVoc,
            # NonVocal: AniNonVoc + HumNonVoc, Nature: Nature, Mechanical: Mechanical, EnvSound: EnvSound)
d_sound_category_broad = {'Music':'Music_all',
						  'Song':'Music_all',
						  'EngSpeech':'Speech_all',
						  'ForSpeech':'Speech_all',
						  'HumVoc':'Vocal_all',
						  'AniVoc':'Vocal_all',
						  'AniNonVoc':'NonVocal_all',
						  'HumNonVoc':'NonVocal_all',
						  'Nature':'Nature_all',
						  'Mechanical':'Mechanical_all',
						  'EnvSound':'EnvSound_all'}

d_roi_label_general_int = {'Primary': 1, 'Anterior': 2, 'Lateral': 3, 'Posterior': 4}

d_model_tasks = {'AST':'sound_classification',
'DCASE2020':'audio_captioning',
'DS2':'ASR',
'Kell2018word':'word_classification',
'Kell2018speaker':'speaker_classification',
'Kell2018music':'music_classification',
'Kell2018audioset':'sound_classification',
'metricGAN': 'speech_enhancement',
'ResNet50word':'word_classification',
'ResNet50speaker':'speaker_classification',
'ResNet50music':'music_classification',
'ResNet50audioset':'sound_classification',
'S2T': 'ASR',
'sepformer': 'source_separation',
'VGGish':'sound_classification',
'wav2vec':'ASR',
'ZeroSpeech2020':'speech_enchancement',
}

d_dim_labels = {# Not demeaned
				'dim_demean-False': 'Effective dimensionality (trained model, NOT demeaned)',
				'dim': 'Effective dimensionality (trained model, NOT demeaned)', # Two naming conventions -- one with True/False, other one with simply the preprocessing step
				'dim_randnetw_demean-False': 'Effective dimensionality (permuted network, NOT demeaned)',
				'dim_randnetw': 'Effective dimensionality (permuted network, NOT demeaned)',
				
				# Normal demeaned
				'dim_demean-True': 'Effective dimensionality (trained model, demeaned)',
				'dim_demean': 'Effective dimensionality (trained model, demeaned)',
				'dim_randnetw_demean-True': 'Effective dimensionality (permuted network, demeaned)',
				'dim_randnetw_demean': 'Effective dimensionality (permuted network, demeaned)',
	
				# PCA cov (only start using ONE naming convention)
				'dim_pca_cov': 'Effective dimensionality (trained model, PCA cov)',
				'dim_randnetw_pca_cov': 'Effective dimensionality (permuted network, PCA cov)',
	
				# PCA corr
				'dim_pca_corr': 'Effective dimensionality (trained model, PCA corr)',
				'dim_randnetw_pca_corr': 'Effective dimensionality (permuted network, PCA corr)',
	
				# Z-scored
				'dim_zscore-True': 'Effective dimensionality (trained model, z-scored)',
				'dim_randnetw_zscore-True': 'Effective dimensionality (permuted network, z-scored)',
				'dim_zscore': 'Effective dimensionality (trained model, z-scored)',
				'dim_randnetw_zscore': 'Effective dimensionality (permuted network, z-scored)',
	
				# Z-scored, PCA cov
				'dim_zscore_pca_cov': 'Effective dimensionality (trained model, z-scored, PCA cov)',
				'dim_randnetw_zscore_pca_cov': 'Effective dimensionality (permuted network, z-scored, PCA cov)',
	
				# Z-scored, PCA corr
				'dim_zscore_pca_corr': 'Effective dimensionality (trained model, z-scored, PCA corr)',
				'dim_randnetw_zscore_pca_corr': 'Effective dimensionality (permuted network, z-scored, PCA corr)',
				
				# Value of interest labels
				'median_r2_test_c_agg-mean': 'Neural noise-corrected $R^2$ (trained model)',
				'median_r2_test_c_agg-mean_randnetw': 'Neural noise-corrected $R^2$ (permuted network)',
				'all_data_rsa_for_best_layer_agg-mean': 'RSA (trained model)',
				'all_data_rsa_for_best_layer_agg-mean_randnetw': 'RSA (permuted network)',}

# d_colors_tasks = {['ASR', 'audio_captioning', 'music_classification', 'sound_classification', 'source_separation', 'speaker_classification', 'speech_enchancement', 'speech_enhancement', 'word_classification']}

## AUD ##
d_layer_reindex = {
	'Kell2018music': ['input_after_preproc', 'relu0', 'maxpool0', 'relu1', 'maxpool1',
					  'relu2', 'relu3', 'relu4', 'avgpool',  'relufc',],# 'final'], # took out 'fullyconnected', 'dropout',
	'Kell2018word': ['input_after_preproc', 'relu0', 'maxpool0', 'relu1', 'maxpool1',
					  'relu2', 'relu3', 'relu4', 'avgpool', 'relufc',],# 'final'],
	'Kell2018speaker': ['input_after_preproc', 'relu0', 'maxpool0', 'relu1', 'maxpool1',
					  'relu2', 'relu3', 'relu4', 'avgpool', 'relufc',], #'final'], # final/signal/speaker_int
	'Kell2018audioset': ['input_after_preproc', 'relu0', 'maxpool0', 'relu1', 'maxpool1',
					  'relu2', 'relu3', 'relu4', 'avgpool', 'relufc',], #'final'], # final/noise/labels_binary_via_int
	'Kell2018multitask': ['input_after_preproc', 'relu0', 'maxpool0', 'relu1', 'maxpool1',
						 'relu2', 'relu3', 'relu4', 'avgpool', 'relufc'], #'final_word', 'final_speaker', 'final_audioset'],
	'Kell2018init': ['input_after_preproc', 'relu0', 'maxpool0', 'relu1', 'maxpool1',
					 'relu2', 'relu3', 'relu4', 'avgpool', 'relufc', ],#'final'],
	'ResNet50music': ['input_after_preproc', 'conv1_relu1', 'maxpool1', 'layer1', 'layer2', 'layer3',
                     'layer4', 'avgpool',],#'final'],
	'ResNet50word': ['input_after_preproc', 'conv1_relu1', 'maxpool1', 'layer1', 'layer2', 'layer3',
                     'layer4', 'avgpool',],# 'final'],
	'ResNet50speaker': ['input_after_preproc', 'conv1_relu1', 'maxpool1', 'layer1', 'layer2', 'layer3',
                     'layer4', 'avgpool',],# 'final'],
	'ResNet50audioset': ['input_after_preproc', 'conv1_relu1', 'maxpool1', 'layer1', 'layer2', 'layer3',
                     'layer4', 'avgpool',],# 'final'],
	'ResNet50multitask': ['input_after_preproc', 'conv1_relu1', 'maxpool1', 'layer1', 'layer2', 'layer3',
                     'layer4', 'avgpool',],# 'final_word', 'final_speaker', 'final_audioset'],
	'ResNet50init': ['input_after_preproc', 'conv1_relu1', 'maxpool1', 'layer1', 'layer2', 'layer3',
                     'layer4', 'avgpool',],# 'final'],
	'DS2': ['Tanh_1', 'Tanh_2', # 'Conv2d_1', 'Bn2d_1', # 'Conv2d_2', 'Bn2d_2',
			 		'LSTM_1-cell', 'LSTM_2-cell', 'LSTM_3-cell', 'LSTM_4-cell', 'LSTM_5-cell', 'Linear'],
	'wav2vec': ['Embedding', 'Encoder_1', 'Encoder_2', 'Encoder_3', 'Encoder_4', 'Encoder_5', 'Encoder_6',
				'Encoder_7', 'Encoder_8', 'Encoder_9', 'Encoder_10', 'Encoder_11', 'Encoder_12', 'Final'],
	'wav2vecpower': ['Embedding', 'Encoder_1', 'Encoder_2', 'Encoder_3', 'Encoder_4', 'Encoder_5', 'Encoder_6',
				'Encoder_7', 'Encoder_8', 'Encoder_9', 'Encoder_10', 'Encoder_11', 'Encoder_12', 'Logits'],
	'VGGish': ['ReLU_1', 'MaxPool2d_1', 'ReLU_2', 'MaxPool2d_2', 'ReLU_3', 'ReLU_4', 'MaxPool2d_3', 'ReLU_5', 'ReLU_6', 'MaxPool2d_4',
			    'ReLU_7',  'ReLU_8',  'ReLU_9'],
	'DCASE2020': ['GRU_1', 'GRU_2', 'GRU_3', 'GRU_4', 'Linear'],
	'AST': ['Embedding', 'Encoder_1', 'Encoder_2', 'Encoder_3', 'Encoder_4', 'Encoder_5', 'Encoder_6',
				'Encoder_7', 'Encoder_8', 'Encoder_9', 'Encoder_10', 'Encoder_11', 'Encoder_12', 'Final'],
	'ZeroSpeech2020': ['ReLU_1', 'ReLU_2', 'ReLU_3', 'ReLU_4', 'ReLU_5'],
	'sepformer': ['Embedding', 'Encoder_1', 'Encoder_2', 'Encoder_3', 'Encoder_4', 'Encoder_5', 'Encoder_6', 'Encoder_7',
				  'Encoder_8',
				  'Encoder_9', 'Encoder_10', 'Encoder_11', 'Encoder_12', 'Encoder_13', 'Encoder_14', 'Encoder_15',
				  'Encoder_16',
				  'Encoder_17', 'Encoder_18', 'Encoder_19', 'Encoder_20', 'Encoder_21', 'Encoder_22', 'Encoder_23',
				  'Encoder_24',
				  'Encoder_25', 'Encoder_26', 'Encoder_27', 'Encoder_28', 'Encoder_29', 'Encoder_30', 'Encoder_31',
				  'Encoder_32'],
	'metricGAN': ['LSTM_1-cell', 'LSTM_2-cell', 'Linear_1', 'Linear_2'],
	'S2T': ['Embedding', 'Encoder_1', 'Encoder_2', 'Encoder_3', 'Encoder_4', 'Encoder_5', 'Encoder_6',
				'Encoder_7', 'Encoder_8', 'Encoder_9', 'Encoder_10', 'Encoder_11', 'Encoder_12'],
	'spectemp': ['avgpool'],
}


d_layer_legend = {
	'Kell2018music': ['Cochleagram', 'ReLU_1', 'MaxPool2d_1', 'ReLU_2', 'MaxPool2d_2',
					  'ReLU_3', 'ReLU_4', 'ReLU_5', 'AvgPool_1', 'ReLU_6', ],
	'Kell2018word': ['Cochleagram', 'ReLU_1', 'MaxPool2d_1', 'ReLU_2', 'MaxPool2d_2',
					  'ReLU_3', 'ReLU_4', 'ReLU_5', 'AvgPool_1', 'ReLU_6', ],
	'Kell2018speaker': ['Cochleagram', 'ReLU_1', 'MaxPool2d_1', 'ReLU_2', 'MaxPool2d_2',
					  'ReLU_3', 'ReLU_4', 'ReLU_5', 'AvgPool_1', 'ReLU_6', ],
	'Kell2018audioset': ['Cochleagram', 'ReLU_1', 'MaxPool2d_1', 'ReLU_2', 'MaxPool2d_2',
					  'ReLU_3', 'ReLU_4', 'ReLU_5', 'AvgPool_1', 'ReLU_6', ],
	'Kell2018multitask': ['Cochleagram', 'ReLU_1', 'MaxPool2d_1', 'ReLU_2', 'MaxPool2d_2',
					  'ReLU_3', 'ReLU_4', 'ReLU_5', 'AvgPool_1', 'ReLU_6', ],
	'Kell2018init': ['Cochleagram', 'ReLU_1', 'MaxPool2d_1', 'ReLU_2', 'MaxPool2d_2',
					  'ReLU_3', 'ReLU_4', 'ReLU_5', 'AvgPool_1', 'ReLU_6', ],
	'ResNet50music': ['Cochleagram', 'ReLU_1', 'MaxPool_1', 'ResNetBlock_1', 'ResNetBlock_2', 'ResNetBlock_3',
					  'ResNetBlock_4', 'AvgPool_1', ],
	'ResNet50word': ['Cochleagram', 'ReLU_1', 'MaxPool_1', 'ResNetBlock_1', 'ResNetBlock_2', 'ResNetBlock_3',
					  'ResNetBlock_4', 'AvgPool_1', ],
	'ResNet50speaker': ['Cochleagram', 'ReLU_1', 'MaxPool_1', 'ResNetBlock_1', 'ResNetBlock_2', 'ResNetBlock_3',
					  'ResNetBlock_4', 'AvgPool_1', ],
	'ResNet50audioset': ['Cochleagram', 'ReLU_1', 'MaxPool_1', 'ResNetBlock_1', 'ResNetBlock_2', 'ResNetBlock_3',
					  'ResNetBlock_4', 'AvgPool_1', ],
	'ResNet50multitask': ['Cochleagram', 'ReLU_1', 'MaxPool_1', 'ResNetBlock_1', 'ResNetBlock_2', 'ResNetBlock_3',
					  'ResNetBlock_4', 'AvgPool_1', ],
	'ResNet50init': ['Cochleagram', 'ReLU_1', 'MaxPool_1', 'ResNetBlock_1', 'ResNetBlock_2', 'ResNetBlock_3',
					  'ResNetBlock_4', 'AvgPool_1', ],
	'DS2': ['HardTanh_1', 'HardTanh_2',
		'LSTM_1', 'LSTM_2', 'LSTM_3', 'LSTM_4', 'LSTM_5', 'Linear_1'],
	'wav2vec': ['Embedding', 'Encoder_1', 'Encoder_2', 'Encoder_3', 'Encoder_4', 'Encoder_5', 'Encoder_6',
                    'Encoder_7', 'Encoder_8', 'Encoder_9', 'Encoder_10', 'Encoder_11', 'Encoder_12', 'Linear_1'],
	'wav2vecpower': ['Embedding', 'Encoder_1', 'Encoder_2', 'Encoder_3', 'Encoder_4', 'Encoder_5', 'Encoder_6',
                    'Encoder_7', 'Encoder_8', 'Encoder_9', 'Encoder_10', 'Encoder_11', 'Encoder_12', 'Linear_1'],
	'VGGish': ['ReLU_1', 'MaxPool2d_1', 'ReLU_2', 'MaxPool2d_2', 'ReLU_3', 'ReLU_4', 'MaxPool2d_3', 'ReLU_5', 'ReLU_6', 'MaxPool2d_4',
			    'ReLU_7',  'ReLU_8',  'ReLU_9',],
	'DCASE2020': ['GRU_1', 'GRU_2', 'GRU_3', 'GRU_4', 'Linear_1'],
	'AST': ['Embedding', 'Encoder_1', 'Encoder_2', 'Encoder_3', 'Encoder_4', 'Encoder_5', 'Encoder_6',
			'Encoder_7', 'Encoder_8', 'Encoder_9', 'Encoder_10', 'Encoder_11', 'Encoder_12', 'Linear_1'],
	'ZeroSpeech2020': ['ReLU_1', 'ReLU_2', 'ReLU_3', 'ReLU_4', 'ReLU_5'],
	'sepformer': ['Embedding', 'Encoder_1', 'Encoder_2', 'Encoder_3', 'Encoder_4', 'Encoder_5', 'Encoder_6', 'Encoder_7',
				  'Encoder_8',
				  'Encoder_9', 'Encoder_10', 'Encoder_11', 'Encoder_12', 'Encoder_13', 'Encoder_14', 'Encoder_15',
				  'Encoder_16',
				  'Encoder_17', 'Encoder_18', 'Encoder_19', 'Encoder_20', 'Encoder_21', 'Encoder_22', 'Encoder_23',
				  'Encoder_24',
				  'Encoder_25', 'Encoder_26', 'Encoder_27', 'Encoder_28', 'Encoder_29', 'Encoder_30', 'Encoder_31',
				  'Encoder_32'],
	'metricGAN': ['LSTM_1', 'LSTM_2', 'LeakyReLU_1', 'Linear_2'],
	'S2T': ['Embedding', 'Encoder_1', 'Encoder_2', 'Encoder_3', 'Encoder_4', 'Encoder_5', 'Encoder_6',
			'Encoder_7', 'Encoder_8', 'Encoder_9', 'Encoder_10', 'Encoder_11', 'Encoder_12'],
	'spectemp': ['avgpool'],
	
}


d_layer_layer_legend_mapping = {'input_after_preproc': 'Cochleagram'} # todo, when cleaning code base, create one big dictionary for all layers


NH2015_all_models_performance_order = ['ResNet50multitask','AST','VGGish','ResNet50audioset',   'Kell2018multitask',
								 'ResNet50word','ResNet50speaker','Kell2018speaker','Kell2018word', 'Kell2018audioset',
								 'Kell2018music', 'ResNet50music', 'S2T', 'wav2vec','sepformer','DS2','ZeroSpeech2020',
								 'DCASE2020', 'metricGAN'
								 ]

B2021_all_models_performance_order = ['ResNet50multitask', 'AST', 'ResNet50audioset', 'Kell2018multitask',
									  'ResNet50word', 'VGGish', 'ResNet50speaker','Kell2018word',
									  'Kell2018speaker','Kell2018audioset', 'Kell2018music','wav2vec',
									  'ResNet50music', 'S2T', 'sepformer', 'DS2', 'DCASE2020', 'ZeroSpeech2020',
									  'metricGAN']

# For plots that order the 165 sounds into categories
sound_category_order = ['Music', 'Song', 'EngSpeech', 'ForSpeech', 'HumVoc', 'AniVoc', 'HumNonVoc',
                        'AniNonVoc', 'Nature', 'Mechanical', 'EnvSound']


# ['Encoder_1_IntraT', 'Encoder_2_IntraT', 'Encoder_3_IntraT', 'Encoder_4_IntraT', 'Encoder_5_IntraT', 'Encoder_6_IntraT', 'Encoder_7_IntraT', 'Encoder_8_IntraT',
# 				  'Encoder_9_InterT', 'Encoder_10_InterT', 'Encoder_11_InterT', 'Encoder_12_InterT', 'Encoder_13_InterT', 'Encoder_14_InterT', 'Encoder_15_InterT', 'Encoder_16_InterT',
# 				  'Encoder_17_IntraT', 'Encoder_18_IntraT', 'Encoder_19_IntraT', 'Encoder_20_IntraT', 'Encoder_21_IntraT', 'Encoder_22_IntraT', 'Encoder_23_IntraT', 'Encoder_24_IntraT',
# 				  'Encoder_25_InterT', 'Encoder_26_InterT', 'Encoder_27_InterT', 'Encoder_28_InterT', 'Encoder_29_InterT', 'Encoder_30_InterT', 'Encoder_31_InterT', 'Encoder_32_InterT']
