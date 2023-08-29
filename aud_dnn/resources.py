##### Related to source DNN models #####

# Sets of models used in the paper:

# Core set of models, n=20 (for Figure 2 and Figure 5)
FIG_2_5_MODEL_LIST = ['Kell2018word', 'Kell2018speaker',
                      'Kell2018music', 'Kell2018audioset',
                      'Kell2018multitask',
				      'ResNet50word', 'ResNet50speaker',
                      'ResNet50music', 'ResNet50audioset',
                      'ResNet50multitask',
				      'AST', 'wav2vec', 'DCASE2020',
                      'DS2', 'VGGish', 'ZeroSpeech2020',
                      'S2T', 'metricGAN', 'sepformer',
                      'spectemp']

# Pairs of the random seeds used in Figure 2
FIG2_SEED_PAIRS = [['Kell2018word', 'Kell2018wordSeed2'],
                   ['Kell2018speaker', 'Kell2018speakerSeed2'],
                   ['Kell2018audioset', 'Kell2018audiosetSeed2'],
                   ['Kell2018multitask', 'Kell2018multitaskSeed2'],
                   ['ResNet50word', 'ResNet50wordSeed2'],
                   ['ResNet50speaker', 'ResNet50speakerSeed2'],
                   ['ResNet50audioset', 'ResNet50audiosetSeed2'],
                   ['ResNet50multitask', 'ResNet50multitaskSeed2']
                   ]

# Models above spectemp baseline (n=15) (for Figure 7)
FIG_7_MODEL_LIST = ['Kell2018word', 'Kell2018speaker',
                    'Kell2018music', 'Kell2018audioset',
                    'Kell2018multitask',
				    'ResNet50word', 'ResNet50speaker',
                    'ResNet50music', 'ResNet50audioset',
                    'ResNet50multitask',
				    'AST', 'wav2vec',
                    'VGGish', 'S2T', 'sepformer']

# Models used for the clean-speech comparison
FIG_8_MODEL_LIST = ['Kell2018word', 'Kell2018wordClean',
                    'ResNet50word', 'ResNet50wordClean',
                    'Kell2018speaker', 'Kell2018speakerClean',
                    'ResNet50speaker', 'ResNet50speakerClean']

# Random Seed Pairs of the Clean Speech models
CLEAN_SPEECH_LIST_2SEEDS = [['Kell2018word', 'Kell2018wordSeed2'],
                            ['Kell2018wordClean', 'Kell2018wordCleanSeed2'],
                            ['Kell2018speaker', 'Kell2018speakerSeed2'],
                            ['Kell2018speakerClean', 'Kell2018speakerCleanSeed2'],
                            ['ResNet50word', 'ResNet50wordSeed2'],
                            ['ResNet50wordClean', 'ResNet50wordCleanSeed2'],
                            ['ResNet50speaker', 'ResNet50speakerSeed2'],
                            ['ResNet50speakerClean', 'ResNet50speakerCleanSeed2']]

# Paired up Clean Speech models, used for Stats comparisons
CLEAN_SPEECH_LIST_PAIRS = [['Kell2018word','Kell2018wordClean'],
                           ['ResNet50word', 'ResNet50wordClean'],
                           ['Kell2018speaker', 'Kell2018speakerClean'],
                           ['ResNet50speaker', 'ResNet50speakerClean']]


# In-house models (n=10) (for Figure 9)
FIG_9_MODEL_LIST = ['Kell2018word', 'ResNet50word',
                    'Kell2018speaker', 'ResNet50speaker',
                    'Kell2018music', 'ResNet50music',
                   'Kell2018audioset','ResNet50audioset',
                    'Kell2018multitask','ResNet50multitask',]

# All unique models (n=24) (including clean speech networks from Figure 8; for use in ED SI Figure). All seed1.
ALL_MODEL_LIST = ['Kell2018word', 'Kell2018speaker',
                      'Kell2018music', 'Kell2018audioset',
                      'Kell2018multitask',
				      'ResNet50word', 'ResNet50speaker',
                      'ResNet50music', 'ResNet50audioset',
                      'ResNet50multitask',
				      'AST', 'wav2vec', 'DCASE2020',
                      'DS2', 'VGGish', 'ZeroSpeech2020',
                      'S2T', 'metricGAN', 'sepformer',
                      'spectemp',
                       'Kell2018wordClean', 'Kell2018speakerClean',
                       'ResNet50wordClean', 'ResNet50speakerClean',]



# Dictionary with key as the source DNN model name and value as a list of the model layers names in correct order.
# Layers included in this dictionary will be loaded in the final analyses compilations.
d_layer_reindex = {
    'Kell2018music': ['relu0', 'maxpool0', 'relu1', 'maxpool1', 'relu2', 'relu3', 'relu4', 'avgpool', 'relufc', ],
    'Kell2018word': ['relu0', 'maxpool0', 'relu1', 'maxpool1', 'relu2', 'relu3', 'relu4', 'avgpool', 'relufc', ],
    'Kell2018wordClean': ['relu0', 'maxpool0', 'relu1', 'maxpool1', 'relu2', 'relu3', 'relu4', 'avgpool', 'relufc', ],
    'Kell2018wordCleanSeed2': ['relu0', 'maxpool0', 'relu1', 'maxpool1', 'relu2', 'relu3', 'relu4', 'avgpool', 'relufc', ],
    'Kell2018wordSeed2': ['relu0', 'maxpool0', 'relu1', 'maxpool1', 'relu2', 'relu3', 'relu4', 'avgpool', 'relufc', ],
    'Kell2018speaker': ['relu0', 'maxpool0', 'relu1', 'maxpool1', 'relu2', 'relu3', 'relu4', 'avgpool', 'relufc', ],
    'Kell2018speakerClean': ['relu0', 'maxpool0', 'relu1', 'maxpool1', 'relu2', 'relu3', 'relu4', 'avgpool', 'relufc'],
    'Kell2018speakerCleanSeed2': ['relu0', 'maxpool0', 'relu1', 'maxpool1', 'relu2', 'relu3', 'relu4', 'avgpool', 'relufc'],
    'Kell2018speakerSeed2': ['relu0', 'maxpool0', 'relu1', 'maxpool1', 'relu2', 'relu3', 'relu4', 'avgpool', 'relufc', ],
    'Kell2018audioset': ['relu0', 'maxpool0', 'relu1', 'maxpool1', 'relu2', 'relu3', 'relu4', 'avgpool', 'relufc', ],
    'Kell2018audiosetSeed2': ['relu0', 'maxpool0', 'relu1', 'maxpool1', 'relu2', 'relu3', 'relu4', 'avgpool', 'relufc', ],
    'Kell2018multitask': ['relu0', 'maxpool0', 'relu1', 'maxpool1', 'relu2', 'relu3', 'relu4', 'avgpool', 'relufc'],
    'Kell2018multitaskSeed2': ['relu0', 'maxpool0', 'relu1', 'maxpool1', 'relu2', 'relu3', 'relu4', 'avgpool', 'relufc'],

    'ResNet50music': ['conv1_relu1', 'maxpool1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', ],
    'ResNet50word': ['conv1_relu1', 'maxpool1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', ],
    'ResNet50wordClean': ['conv1_relu1', 'maxpool1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', ],
    'ResNet50wordCleanSeed2': ['conv1_relu1', 'maxpool1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool'],
    'ResNet50wordSeed2': ['conv1_relu1', 'maxpool1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', ],
    'ResNet50speaker': ['conv1_relu1', 'maxpool1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', ],
    'ResNet50speakerClean': ['conv1_relu1', 'maxpool1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool'],
    'ResNet50speakerCleanSeed2': ['conv1_relu1', 'maxpool1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool'],
    'ResNet50speakerSeed2': ['conv1_relu1', 'maxpool1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', ],
    'ResNet50audioset': ['conv1_relu1', 'maxpool1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', ],
    'ResNet50audiosetSeed2': ['conv1_relu1', 'maxpool1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', ],
    'ResNet50multitask': ['conv1_relu1', 'maxpool1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', ],
    'ResNet50multitaskSeed2': ['conv1_relu1', 'maxpool1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', ],

    'DS2': ['Tanh_1', 'Tanh_2', 'LSTM_1-cell', 'LSTM_2-cell', 'LSTM_3-cell', 'LSTM_4-cell', 'LSTM_5-cell', 'Linear'],
    'wav2vec': ['Embedding', 'Encoder_1', 'Encoder_2', 'Encoder_3', 'Encoder_4', 'Encoder_5', 'Encoder_6',
                'Encoder_7', 'Encoder_8', 'Encoder_9', 'Encoder_10', 'Encoder_11', 'Encoder_12', 'Final'],
    'VGGish': ['ReLU_1', 'MaxPool2d_1', 'ReLU_2', 'MaxPool2d_2', 'ReLU_3', 'ReLU_4', 'MaxPool2d_3', 'ReLU_5', 'ReLU_6',
               'MaxPool2d_4', 'ReLU_7', 'ReLU_8', 'ReLU_9'],
    'DCASE2020': ['GRU_1', 'GRU_2', 'GRU_3', 'GRU_4', 'Linear'],
    'AST': ['Embedding', 'Encoder_1', 'Encoder_2', 'Encoder_3', 'Encoder_4', 'Encoder_5', 'Encoder_6',
            'Encoder_7', 'Encoder_8', 'Encoder_9', 'Encoder_10', 'Encoder_11', 'Encoder_12', 'Final'],
    'ZeroSpeech2020': ['ReLU_1', 'ReLU_2', 'ReLU_3', 'ReLU_4', 'ReLU_5'],
    'sepformer': ['Embedding', 'Encoder_1', 'Encoder_2', 'Encoder_3', 'Encoder_4', 'Encoder_5', 'Encoder_6',
                  'Encoder_7', 'Encoder_8', 'Encoder_9', 'Encoder_10', 'Encoder_11', 'Encoder_12', 'Encoder_13',
                  'Encoder_14', 'Encoder_15', 'Encoder_16', 'Encoder_17', 'Encoder_18', 'Encoder_19', 'Encoder_20',
                  'Encoder_21', 'Encoder_22', 'Encoder_23', 'Encoder_24', 'Encoder_25', 'Encoder_26', 'Encoder_27',
                  'Encoder_28', 'Encoder_29', 'Encoder_30', 'Encoder_31', 'Encoder_32'],
    'metricGAN': ['LSTM_1-cell', 'LSTM_2-cell', 'Linear_1', 'Linear_2'],
    'S2T': ['Embedding', 'Encoder_1', 'Encoder_2', 'Encoder_3', 'Encoder_4', 'Encoder_5', 'Encoder_6',
            'Encoder_7', 'Encoder_8', 'Encoder_9', 'Encoder_10', 'Encoder_11', 'Encoder_12'],
    'spectemp': ['avgpool'],
}


# Dictionary that maps from the layer values in d_layer_reindex to a nice name for plots.
d_layer_names = {'Kell2018music': {'input_after_preproc': 'Cochleagram', 'relu0': 'ReLU_1', 'maxpool0': 'MaxPool2d_1',
                                      'relu1': 'ReLU_2', 'maxpool1': 'MaxPool2d_2', 'relu2': 'ReLU_3', 'relu3': 'ReLU_4',
                                        'relu4': 'ReLU_5', 'avgpool': 'AvgPool_1', 'relufc': 'ReLU_6', },
                'Kell2018word': {'input_after_preproc': 'Cochleagram', 'relu0': 'ReLU_1', 'maxpool0': 'MaxPool2d_1',
                                        'relu1': 'ReLU_2', 'maxpool1': 'MaxPool2d_2', 'relu2': 'ReLU_3', 'relu3': 'ReLU_4',
                                        'relu4': 'ReLU_5', 'avgpool': 'AvgPool_1', 'relufc': 'ReLU_6', },
                'Kell2018wordSeed2': {'input_after_preproc': 'Cochleagram', 'relu0': 'ReLU_1', 'maxpool0': 'MaxPool2d_1',
                                        'relu1': 'ReLU_2', 'maxpool1': 'MaxPool2d_2', 'relu2': 'ReLU_3', 'relu3': 'ReLU_4',
                                        'relu4': 'ReLU_5', 'avgpool': 'AvgPool_1', 'relufc': 'ReLU_6', },
                 'Kell2018wordClean': {'input_after_preproc': 'Cochleagram', 'relu0': 'ReLU_1', 'maxpool0': 'MaxPool2d_1',
                                       'relu1': 'ReLU_2', 'maxpool1': 'MaxPool2d_2', 'relu2': 'ReLU_3', 'relu3': 'ReLU_4',
                                       'relu4': 'ReLU_5', 'avgpool': 'AvgPool_1', 'relufc': 'ReLU_6', },
                 'Kell2018wordCleanSeed2': {'input_after_preproc': 'Cochleagram', 'relu0': 'ReLU_1', 'maxpool0': 'MaxPool2d_1',
                                        'relu1': 'ReLU_2', 'maxpool1': 'MaxPool2d_2', 'relu2': 'ReLU_3', 'relu3': 'ReLU_4',
                                        'relu4': 'ReLU_5', 'avgpool': 'AvgPool_1', 'relufc': 'ReLU_6', },
                 'Kell2018speaker': {'input_after_preproc': 'Cochleagram', 'relu0': 'ReLU_1', 'maxpool0': 'MaxPool2d_1',
                                        'relu1': 'ReLU_2', 'maxpool1': 'MaxPool2d_2', 'relu2': 'ReLU_3', 'relu3': 'ReLU_4',
                                        'relu4': 'ReLU_5', 'avgpool': 'AvgPool_1', 'relufc': 'ReLU_6', },
                 'Kell2018speakerClean': {'input_after_preproc': 'Cochleagram', 'relu0': 'ReLU_1', 'maxpool0': 'MaxPool2d_1',
                                        'relu1': 'ReLU_2', 'maxpool1': 'MaxPool2d_2', 'relu2': 'ReLU_3', 'relu3': 'ReLU_4',
                                        'relu4': 'ReLU_5', 'avgpool': 'AvgPool_1', 'relufc': 'ReLU_6', },
                 'Kell2018speakerCleanSeed2': {'input_after_preproc': 'Cochleagram', 'relu0': 'ReLU_1', 'maxpool0': 'MaxPool2d_1',
                                        'relu1': 'ReLU_2', 'maxpool1': 'MaxPool2d_2', 'relu2': 'ReLU_3', 'relu3': 'ReLU_4',
                                        'relu4': 'ReLU_5', 'avgpool': 'AvgPool_1', 'relufc': 'ReLU_6', },
                    'Kell2018speakerSeed2': {'input_after_preproc': 'Cochleagram', 'relu0': 'ReLU_1', 'maxpool0': 'MaxPool2d_1',
                                        'relu1': 'ReLU_2', 'maxpool1': 'MaxPool2d_2', 'relu2': 'ReLU_3', 'relu3': 'ReLU_4',
                                        'relu4': 'ReLU_5', 'avgpool': 'AvgPool_1', 'relufc': 'ReLU_6', },
                'Kell2018audioset': {'input_after_preproc': 'Cochleagram', 'relu0': 'ReLU_1', 'maxpool0': 'MaxPool2d_1',
                                        'relu1': 'ReLU_2', 'maxpool1': 'MaxPool2d_2', 'relu2': 'ReLU_3', 'relu3': 'ReLU_4',
                                        'relu4': 'ReLU_5', 'avgpool': 'AvgPool_1', 'relufc': 'ReLU_6', },
                'Kell2018audiosetSeed2': {'input_after_preproc': 'Cochleagram', 'relu0': 'ReLU_1', 'maxpool0': 'MaxPool2d_1',
                                        'relu1': 'ReLU_2', 'maxpool1': 'MaxPool2d_2', 'relu2': 'ReLU_3', 'relu3': 'ReLU_4',
                                        'relu4': 'ReLU_5', 'avgpool': 'AvgPool_1', 'relufc': 'ReLU_6', },
                 'Kell2018multitask': {'input_after_preproc': 'Cochleagram', 'relu0': 'ReLU_1', 'maxpool0': 'MaxPool2d_1',
                                        'relu1': 'ReLU_2', 'maxpool1': 'MaxPool2d_2', 'relu2': 'ReLU_3', 'relu3': 'ReLU_4',
                                        'relu4': 'ReLU_5', 'avgpool': 'AvgPool_1', 'relufc': 'ReLU_6', },
                'Kell2018multitaskSeed2': {'input_after_preproc': 'Cochleagram', 'relu0': 'ReLU_1', 'maxpool0': 'MaxPool2d_1',
                                        'relu1': 'ReLU_2', 'maxpool1': 'MaxPool2d_2', 'relu2': 'ReLU_3', 'relu3': 'ReLU_4',
                                        'relu4': 'ReLU_5', 'avgpool': 'AvgPool_1', 'relufc': 'ReLU_6', },

                 'ResNet50music': {'input_after_preproc': 'Cochleagram', 'conv1_relu1': 'ReLU_1', 'maxpool1': 'MaxPool_1',
                                      'layer1': 'ResNetBlock_1', 'layer2': 'ResNetBlock_2', 'layer3': 'ResNetBlock_3',
                                        'layer4': 'ResNetBlock_4', 'avgpool': 'AvgPool_1', },
                 'ResNet50word': {'input_after_preproc': 'Cochleagram', 'conv1_relu1': 'ReLU_1', 'maxpool1': 'MaxPool_1',
                                        'layer1': 'ResNetBlock_1', 'layer2': 'ResNetBlock_2', 'layer3': 'ResNetBlock_3',
                                        'layer4': 'ResNetBlock_4', 'avgpool': 'AvgPool_1', },
                 'ResNet50wordClean': {'input_after_preproc': 'Cochleagram', 'conv1_relu1': 'ReLU_1', 'maxpool1': 'MaxPool_1',
                                  'layer1': 'ResNetBlock_1', 'layer2': 'ResNetBlock_2', 'layer3': 'ResNetBlock_3',
                                  'layer4': 'ResNetBlock_4', 'avgpool': 'AvgPool_1', },
                'ResNet50wordCleanSeed2': {'input_after_preproc': 'Cochleagram', 'conv1_relu1': 'ReLU_1', 'maxpool1': 'MaxPool_1',
                                    'layer1': 'ResNetBlock_1', 'layer2': 'ResNetBlock_2', 'layer3': 'ResNetBlock_3',
                                    'layer4': 'ResNetBlock_4', 'avgpool': 'AvgPool_1', },
                 'ResNet50wordSeed2': {'input_after_preproc': 'Cochleagram', 'conv1_relu1': 'ReLU_1', 'maxpool1': 'MaxPool_1',
                                        'layer1': 'ResNetBlock_1', 'layer2': 'ResNetBlock_2', 'layer3': 'ResNetBlock_3',
                                        'layer4': 'ResNetBlock_4', 'avgpool': 'AvgPool_1', },
                 'ResNet50speaker': {'input_after_preproc': 'Cochleagram', 'conv1_relu1': 'ReLU_1', 'maxpool1': 'MaxPool_1',
                                        'layer1': 'ResNetBlock_1', 'layer2': 'ResNetBlock_2', 'layer3': 'ResNetBlock_3',
                                        'layer4': 'ResNetBlock_4', 'avgpool': 'AvgPool_1', },
                 'ResNet50speakerClean': {'input_after_preproc': 'Cochleagram', 'conv1_relu1': 'ReLU_1', 'maxpool1': 'MaxPool_1',
                                    'layer1': 'ResNetBlock_1', 'layer2': 'ResNetBlock_2', 'layer3': 'ResNetBlock_3',
                                    'layer4': 'ResNetBlock_4', 'avgpool': 'AvgPool_1', },
                 'ResNet50speakerCleanSeed2': {'input_after_preproc': 'Cochleagram', 'conv1_relu1': 'ReLU_1', 'maxpool1': 'MaxPool_1',
                                    'layer1': 'ResNetBlock_1', 'layer2': 'ResNetBlock_2', 'layer3': 'ResNetBlock_3',
                                    'layer4': 'ResNetBlock_4', 'avgpool': 'AvgPool_1', },
                    'ResNet50speakerSeed2': {'input_after_preproc': 'Cochleagram', 'conv1_relu1': 'ReLU_1', 'maxpool1': 'MaxPool_1',
                                        'layer1': 'ResNetBlock_1', 'layer2': 'ResNetBlock_2', 'layer3': 'ResNetBlock_3',
                                        'layer4': 'ResNetBlock_4', 'avgpool': 'AvgPool_1', },
                 'ResNet50audioset': {'input_after_preproc': 'Cochleagram', 'conv1_relu1': 'ReLU_1', 'maxpool1': 'MaxPool_1',
                                        'layer1': 'ResNetBlock_1', 'layer2': 'ResNetBlock_2', 'layer3': 'ResNetBlock_3',
                                        'layer4': 'ResNetBlock_4', 'avgpool': 'AvgPool_1', },
                    'ResNet50audiosetSeed2': {'input_after_preproc': 'Cochleagram', 'conv1_relu1': 'ReLU_1', 'maxpool1': 'MaxPool_1',
                                        'layer1': 'ResNetBlock_1', 'layer2': 'ResNetBlock_2', 'layer3': 'ResNetBlock_3',
                                        'layer4': 'ResNetBlock_4', 'avgpool': 'AvgPool_1', },
                 'ResNet50multitask': {'input_after_preproc': 'Cochleagram', 'conv1_relu1': 'ReLU_1', 'maxpool1': 'MaxPool_1',
                                        'layer1': 'ResNetBlock_1', 'layer2': 'ResNetBlock_2', 'layer3': 'ResNetBlock_3',
                                        'layer4': 'ResNetBlock_4', 'avgpool': 'AvgPool_1', },
                    'ResNet50multitaskSeed2': {'input_after_preproc': 'Cochleagram', 'conv1_relu1': 'ReLU_1', 'maxpool1': 'MaxPool_1',
                                        'layer1': 'ResNetBlock_1', 'layer2': 'ResNetBlock_2', 'layer3': 'ResNetBlock_3',
                                        'layer4': 'ResNetBlock_4', 'avgpool': 'AvgPool_1', },
                 'AST': {'Embedding': 'Embedding', 'Encoder_1': 'Encoder_1', 'Encoder_2': 'Encoder_2', 'Encoder_3': 'Encoder_3',
                         'Encoder_4': 'Encoder_4', 'Encoder_5': 'Encoder_5', 'Encoder_6': 'Encoder_6',
                         'Encoder_7': 'Encoder_7', 'Encoder_8': 'Encoder_8', 'Encoder_9': 'Encoder_9',
                         'Encoder_10': 'Encoder_10', 'Encoder_11': 'Encoder_11', 'Encoder_12': 'Encoder_12', 'Final': 'Linear_1'},
                 'VGGish': {'ReLU_1': 'ReLU_1', 'MaxPool2d_1': 'MaxPool2d_1', 'ReLU_2': 'ReLU_2', 'MaxPool2d_2': 'MaxPool2d_2',
                            'ReLU_3': 'ReLU_3', 'ReLU_4': 'ReLU_4', 'MaxPool2d_3': 'MaxPool2d_3', 'ReLU_5': 'ReLU_5',
                            'ReLU_6': 'ReLU_6', 'MaxPool2d_4': 'MaxPool2d_4', 'ReLU_7': 'ReLU_7', 'ReLU_8': 'ReLU_8',
                            'ReLU_9': 'ReLU_9', },
                 'sepformer': {'Embedding': 'Embedding', 'Encoder_1': 'Encoder_1', 'Encoder_2': 'Encoder_2',
                               'Encoder_3': 'Encoder_3', 'Encoder_4': 'Encoder_4', 'Encoder_5': 'Encoder_5',
                               'Encoder_6': 'Encoder_6', 'Encoder_7': 'Encoder_7', 'Encoder_8': 'Encoder_8',
                                 'Encoder_9': 'Encoder_9', 'Encoder_10': 'Encoder_10', 'Encoder_11': 'Encoder_11',
                                    'Encoder_12': 'Encoder_12', 'Encoder_13': 'Encoder_13', 'Encoder_14': 'Encoder_14',
                                    'Encoder_15': 'Encoder_15', 'Encoder_16': 'Encoder_16', 'Encoder_17': 'Encoder_17',
                                    'Encoder_18': 'Encoder_18', 'Encoder_19': 'Encoder_19', 'Encoder_20': 'Encoder_20',
                                    'Encoder_21': 'Encoder_21', 'Encoder_22': 'Encoder_22', 'Encoder_23': 'Encoder_23',
                                    'Encoder_24': 'Encoder_24', 'Encoder_25': 'Encoder_25', 'Encoder_26': 'Encoder_26',
                                    'Encoder_27': 'Encoder_27', 'Encoder_28': 'Encoder_28', 'Encoder_29': 'Encoder_29',
                                    'Encoder_30': 'Encoder_30', 'Encoder_31': 'Encoder_31', 'Encoder_32': 'Encoder_32',
                                 },
                 'metricGAN': {'LSTM_1-cell': 'LSTM_1', 'LSTM_2-cell': 'LSTM_2', 'Linear_1': 'LeakyReLU_1', 'Linear_2': 'Linear_2'},
                 'wav2vec': {'Embedding': 'Embedding', 'Encoder_1': 'Encoder_1', 'Encoder_2': 'Encoder_2', 'Encoder_3': 'Encoder_3',
                                'Encoder_4': 'Encoder_4', 'Encoder_5': 'Encoder_5', 'Encoder_6': 'Encoder_6', 'Encoder_7': 'Encoder_7',
                                'Encoder_8': 'Encoder_8', 'Encoder_9': 'Encoder_9', 'Encoder_10': 'Encoder_10', 'Encoder_11': 'Encoder_11',
                                'Encoder_12': 'Encoder_12', 'Final': 'Linear_1'},
                 'ZeroSpeech2020': {'ReLU_1': 'ReLU_1', 'ReLU_2': 'ReLU_2', 'ReLU_3': 'ReLU_3', 'ReLU_4': 'ReLU_4', 'ReLU_5': 'ReLU_5'},
                    'S2T': {'Embedding': 'Embedding', 'Encoder_1': 'Encoder_1', 'Encoder_2': 'Encoder_2', 'Encoder_3': 'Encoder_3',
                            'Encoder_4': 'Encoder_4', 'Encoder_5': 'Encoder_5', 'Encoder_6': 'Encoder_6', 'Encoder_7': 'Encoder_7',
                            'Encoder_8': 'Encoder_8', 'Encoder_9': 'Encoder_9', 'Encoder_10': 'Encoder_10', 'Encoder_11': 'Encoder_11',
                            'Encoder_12': 'Encoder_12'},
                 'DS2': {'Tanh_1': 'HardTanh_1', 'Tanh_2': 'HardTanh_2', 'LSTM_1-cell': 'LSTM_1', 'LSTM_2-cell': 'LSTM_2',
                            'LSTM_3-cell': 'LSTM_3', 'LSTM_4-cell': 'LSTM_4', 'LSTM_5-cell': 'LSTM_5', 'Linear': 'Linear_1'},
                 'DCASE2020': {'GRU_1': 'GRU_1', 'GRU_2': 'GRU_2', 'GRU_3': 'GRU_3', 'GRU_4': 'GRU_4', 'Linear': 'Linear_1'},
                 'spectemp': {'avgpool': 'AvgPool'},
                 }


# Dictionary with keys as the name of the source DNN model and value a dictionary with the keys as the 'pretty',
# interpretable name of the layer, and the value the name of the layer it was saved as (the PyTorch name).
# This is only used for the external models given that they are extracted using Pytorch module names.
source_layer_map = {
    'DCASE2020': {
        'GRU_1': 'GRU(64, 256, batch_first=True, bidirectional=True)--0--hidden',
        'GRU_2': 'GRU(512, 256, batch_first=True, bidirectional=True)--0--hidden',
        'GRU_3': 'GRU(512, 256, batch_first=True, bidirectional=True)--1--hidden',
        'GRU_4': 'GRU(512, 256, batch_first=True)--0--hidden',  # decoder
        'Linear': 'Linear(in_features=256, out_features=4367, bias=True)--0'
    },
    'DS2': {
        'Conv2d_1': 'Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5))--0',
        'Bn2d_1': 'BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)--0',
        'Tanh_1': 'Hardtanh(min_val=0, max_val=20)--0',
        'Conv2d_2': 'Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5))--0',
        'Bn2d_2': 'BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)--1',
        'Tanh_2': 'Hardtanh(min_val=0, max_val=20)--1',
        'LSTM_1': 'LSTM(1312, 1024, bidirectional=True)--0--hidden',  # hidden
        'LSTM_1-cell': 'LSTM(1312, 1024, bidirectional=True)--0--cell', # cell states: these are the ones we use, see d_layer_reindex and paper methods
        'LSTM_2': 'LSTM(1024, 1024, bidirectional=True)--0--hidden',
        'LSTM_2-cell': 'LSTM(1024, 1024, bidirectional=True)--0--cell',
        'LSTM_3': 'LSTM(1024, 1024, bidirectional=True)--1--hidden',
        'LSTM_3-cell': 'LSTM(1024, 1024, bidirectional=True)--1--cell',
        'LSTM_4': 'LSTM(1024, 1024, bidirectional=True)--2--hidden',
        'LSTM_4-cell': 'LSTM(1024, 1024, bidirectional=True)--2--cell',
        'LSTM_5': 'LSTM(1024, 1024, bidirectional=True)--3--hidden',
        'LSTM_5-cell': 'LSTM(1024, 1024, bidirectional=True)--3--cell',
        'Linear': 'Linear(in_features=1024, out_features=29, bias=False)--0'
    },
    'VGGish': {
        'Conv2d_1': 'Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))--0',
        'ReLU_1': 'ReLU()--0', # we use the ReLU outputs to avoid negative values
        'MaxPool2d_1': 'MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)--0',
        'Conv2d_2': 'Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))--0',
        'ReLU_2': 'ReLU()--1',
        'MaxPool2d_2': 'MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)--1',
        'Conv2d_3': 'Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))--0',
        'ReLU_3': 'ReLU()--2',
        'Conv2d_4': 'Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))--0',
        'ReLU_4': 'ReLU()--3',
        'MaxPool2d_3': 'MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)--2',
        'Conv2d_5': 'Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))--0',
        'ReLU_5': 'ReLU()--4',
        'Conv2d_6': 'Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))--0',
        'ReLU_6': 'ReLU()--5',
        'MaxPool2d_4': 'MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)--3',
        'Linear_1': 'Linear(in_features=12288, out_features=4096, bias=True)--0',
        'ReLU_7': 'ReLU()--6',
        'Linear_2': 'Linear(in_features=4096, out_features=4096, bias=True)--0',
        'ReLU_8': 'ReLU()--7',
        'Linear_3': 'Linear(in_features=4096, out_features=128, bias=True)--0',
        'ReLU_9': 'ReLU()--8',
        'Post-Processed_Features': 'Post-Processed_Features'
    },
    'AST': {'Embedding': 'Conv2d(1, 768, kernel_size=(16, 16), stride=(10, 10))--0',  # patch embedding, the conv layer
            'Encoder_1': 'Linear(in_features=3072, out_features=768, bias=True)--0',
            'Encoder_2': 'Linear(in_features=3072, out_features=768, bias=True)--1',
            'Encoder_3': 'Linear(in_features=3072, out_features=768, bias=True)--2',
            'Encoder_4': 'Linear(in_features=3072, out_features=768, bias=True)--3',
            'Encoder_5': 'Linear(in_features=3072, out_features=768, bias=True)--4',
            'Encoder_6': 'Linear(in_features=3072, out_features=768, bias=True)--5',
            'Encoder_7': 'Linear(in_features=3072, out_features=768, bias=True)--6',
            'Encoder_8': 'Linear(in_features=3072, out_features=768, bias=True)--7',
            'Encoder_9': 'Linear(in_features=3072, out_features=768, bias=True)--8',
            'Encoder_10': 'Linear(in_features=3072, out_features=768, bias=True)--9',
            'Encoder_11': 'Linear(in_features=3072, out_features=768, bias=True)--10',
            'Encoder_12': 'Linear(in_features=3072, out_features=768, bias=True)--11',
            'Final': 'Final'},
    'ZeroSpeech2020': {'ReLU_1': 'ReLU(inplace=True)--0',
                       'ReLU_2': 'ReLU(inplace=True)--1',
                       'ReLU_3': 'ReLU(inplace=True)--2',
                       'ReLU_4': 'ReLU(inplace=True)--3',
                       'ReLU_5': 'ReLU(inplace=True)--4', },
    'wav2vec': {'Embedding': 'Embedding',
                'Encoder_1': 'Encoder_1',
                'Encoder_2': 'Encoder_2',
                'Encoder_3': 'Encoder_3',
                'Encoder_4': 'Encoder_4',
                'Encoder_5': 'Encoder_5',
                'Encoder_6': 'Encoder_6',
                'Encoder_7': 'Encoder_7',
                'Encoder_8': 'Encoder_8',
                'Encoder_9': 'Encoder_9',
                'Encoder_10': 'Encoder_10',
                'Encoder_11': 'Encoder_11',
                'Encoder_12': 'Encoder_12',
                'Final': 'Final',},
    'sepformer': {
        'Embedding': 'Conv1d(1, 256, kernel_size=(16,), stride=(8,), bias=False)--0',
        'Encoder_1': 'Linear(in_features=1024, out_features=256, bias=True)--0',
        'Encoder_2': 'Linear(in_features=1024, out_features=256, bias=True)--1',
        'Encoder_3': 'Linear(in_features=1024, out_features=256, bias=True)--2',
        'Encoder_4': 'Linear(in_features=1024, out_features=256, bias=True)--3',
        'Encoder_5': 'Linear(in_features=1024, out_features=256, bias=True)--4',
        'Encoder_6': 'Linear(in_features=1024, out_features=256, bias=True)--5',
        'Encoder_7': 'Linear(in_features=1024, out_features=256, bias=True)--6',
        'Encoder_8': 'Linear(in_features=1024, out_features=256, bias=True)--7',
        'Encoder_9': 'Linear(in_features=1024, out_features=256, bias=True)--8',
        'Encoder_10': 'Linear(in_features=1024, out_features=256, bias=True)--9',
        'Encoder_11': 'Linear(in_features=1024, out_features=256, bias=True)--10',
        'Encoder_12': 'Linear(in_features=1024, out_features=256, bias=True)--11',
        'Encoder_13': 'Linear(in_features=1024, out_features=256, bias=True)--12',
        'Encoder_14': 'Linear(in_features=1024, out_features=256, bias=True)--13',
        'Encoder_15': 'Linear(in_features=1024, out_features=256, bias=True)--14',
        'Encoder_16': 'Linear(in_features=1024, out_features=256, bias=True)--15',
        'Encoder_17': 'Linear(in_features=1024, out_features=256, bias=True)--16',
        'Encoder_18': 'Linear(in_features=1024, out_features=256, bias=True)--17',
        'Encoder_19': 'Linear(in_features=1024, out_features=256, bias=True)--18',
        'Encoder_20': 'Linear(in_features=1024, out_features=256, bias=True)--19',
        'Encoder_21': 'Linear(in_features=1024, out_features=256, bias=True)--20',
        'Encoder_22': 'Linear(in_features=1024, out_features=256, bias=True)--21',
        'Encoder_23': 'Linear(in_features=1024, out_features=256, bias=True)--22',
        'Encoder_24': 'Linear(in_features=1024, out_features=256, bias=True)--23',
        'Encoder_25': 'Linear(in_features=1024, out_features=256, bias=True)--24',
        'Encoder_26': 'Linear(in_features=1024, out_features=256, bias=True)--25',
        'Encoder_27': 'Linear(in_features=1024, out_features=256, bias=True)--26',
        'Encoder_28': 'Linear(in_features=1024, out_features=256, bias=True)--27',
        'Encoder_29': 'Linear(in_features=1024, out_features=256, bias=True)--28',
        'Encoder_30': 'Linear(in_features=1024, out_features=256, bias=True)--29',
        'Encoder_31': 'Linear(in_features=1024, out_features=256, bias=True)--30',
        'Encoder_32': 'Linear(in_features=1024, out_features=256, bias=True)--31',
    },
    'metricGAN': {
        'LSTM_1': 'LSTM(257, 200, num_layers=2, batch_first=True, bidirectional=True)--l1--hidden',
        'LSTM_1-cell': 'LSTM(257, 200, num_layers=2, batch_first=True, bidirectional=True)--l1--cell', # cell states: these are the ones we use, see d_layer_reindex and paper methods
        'LSTM_2': 'LSTM(257, 200, num_layers=2, batch_first=True, bidirectional=True)--l2--hidden',
        'LSTM_2-cell': 'LSTM(257, 200, num_layers=2, batch_first=True, bidirectional=True)--l2--cell',
        'Linear_1': 'Linear(in_features=400, out_features=300, bias=True)--0',
        'Linear_2': 'Linear(in_features=300, out_features=257, bias=True)--0'
    },
    'S2T': {'Embedding': 'Embedding',
            'Encoder_1': 'Encoder_1',
            'Encoder_2': 'Encoder_2',
            'Encoder_3': 'Encoder_3',
            'Encoder_4': 'Encoder_4',
            'Encoder_5': 'Encoder_5',
            'Encoder_6': 'Encoder_6',
            'Encoder_7': 'Encoder_7',
            'Encoder_8': 'Encoder_8',
            'Encoder_9': 'Encoder_9',
            'Encoder_10': 'Encoder_10',
            'Encoder_11': 'Encoder_11',
            'Encoder_12': 'Encoder_12'},

}

###### MORE PLOTTING/FIGURE RELATED ######
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Plot styles
plt.style.use('seaborn-pastel')
matplotlib.rcParams['mathtext.fontset'] = 'custom'
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
                'envsounds': 'navajowhite',
                'Anterior_rh': 'cyan',
                'Anterior_lh': 'cyan',
                'Primary_rh': 'springgreen',
                'Primary_lh': 'springgreen',
                'Lateral_rh': 'blue',
                'Lateral_lh': 'blue',
                'Posterior_rh': 'magenta',
                'Posterior_lh': 'magenta',
                'Anterior': 'cyan',
                'Primary': 'springgreen',
                'Lateral': 'blue',
                'Posterior': 'magenta',}

d_model_colors = {
    'Kell2018word': '#AD233B',
    'Kell2018wordClean': '#CA8484',
    'Kell2018wordCleanSeed2': '#CA8484',
    'Kell2018wordSeed2': '#AD233B',
    'Kell2018speaker': '#F13336',
    'Kell2018speakerClean': '#FDB5B5',
    'Kell2018speakerCleanSeed2': '#FDB5B5',
    'Kell2018speakerSeed2': '#F13336',
    'Kell2018music': '#F9671D',
    'Kell2018audioset': '#FF80A5',
    'Kell2018audiosetSeed2': '#FF80A5',
    'Kell2018multitask': '#FFBAA5',
    'Kell2018multitaskSeed2': '#FFBAA5',
    'ResNet50word': 'darkgreen',
    'ResNet50wordClean': '#90BF95',
    'ResNet50wordCleanSeed2': '#90BF95',
    'ResNet50wordSeed2': 'darkgreen',
    'ResNet50speaker': '#55D409',
    'ResNet50speakerClean': '#B6F982',
    'ResNet50speakerCleanSeed2': '#B6F982',
    'ResNet50speakerSeed2': '#55D409',
    'ResNet50music': 'mediumseagreen',
    'ResNet50audioset': '#80F396',
    'ResNet50audiosetSeed2': '#80F396',
    'ResNet50multitask': '#C3F380',
    'ResNet50multitaskSeed2': '#C3F380',

    'AST': 'mediumorchid',
    'DCASE2020': 'gold',
    'DS2': '#E8E869',
    'metricGAN': 'sienna',
    'S2T': 'blueviolet',
    'sepformer': 'purple',
    'spectemp': 'grey',
    'VGGish': 'royalblue',
    'wav2vec': 'plum',
    'ZeroSpeech2020': 'skyblue',
    'mock': 'grey',
}

d_model_names = {
				'Kell2018word': 'CochCNN9-Word',
                 'Kell2018wordClean': 'CochCNN9-WordClean',
                'Kell2018wordCleanSeed2': 'CochCNN9-WordCleanSeed2',
                'Kell2018wordSeed2': 'CochCNN9-WordSeed2',
				'Kell2018speaker': 'CochCNN9-Speaker',
                 'Kell2018speakerClean': 'CochCNN9-SpeakerClean',
                'Kell2018speakerCleanSeed2': 'CochCNN9-SpeakerCleanSeed2',
				'Kell2018speakerSeed2': 'CochCNN9-SpeakerSeed2',
				'Kell2018music': 'CochCNN9-Genre',
				'Kell2018audioset': 'CochCNN9-AudioSet',
				'Kell2018audiosetSeed2': 'CochCNN9-AudioSetSeed2',
				'Kell2018multitask': 'CochCNN9-MultiTask',
				'Kell2018multitaskSeed2': 'CochCNN9-MultiTaskSeed2',
				'ResNet50word': 'CochResNet50-Word',
                'ResNet50wordClean': 'CochResNet50-WordClean',
                'ResNet50wordCleanSeed2': 'CochResNet50-WordCleanSeed2',
				'ResNet50wordSeed2': 'CochResNet50-WordSeed2',
				'ResNet50speaker': 'CochResNet50-Speaker',
                'ResNet50speakerClean': 'CochResNet50-SpeakerClean',
                'ResNet50speakerCleanSeed2': 'CochResNet50-SpeakerCleanSeed2',
				'ResNet50speakerSeed2': 'CochResNet50-SpeakerSeed2',
				'ResNet50music': 'CochResNet50-Genre',
				'ResNet50audioset': 'CochResNet50-AudioSet',
				'ResNet50audiosetSeed2': 'CochResNet50-AudioSetSeed2',
				'ResNet50multitask': 'CochResNet50-MultiTask',
				'ResNet50multitaskSeed2': 'CochResNet50-MultiTaskSeed2',

				'AST':'AST',
				'DCASE2020':'DCASE2020',
				'DS2':'DeepSpeech2',
				'metricGAN': 'MetricGAN',
				'S2T': 'S2T',
				'sepformer': 'SepFormer',
				'spectemp': 'SpectroTemporal',
				'VGGish':'VGGish',
				'wav2vec':'Wav2Vec2',
				'ZeroSpeech2020':'VQ-VAE',
				'mock':''}

d_model_markers = {'Kell2018word':'s',
                   'Kell2018wordSeed2':'s',
                   'Kell2018speaker':'D',
                     'Kell2018speakerSeed2':'D',
                   'Kell2018music':'x',
                   'Kell2018audioset':'+',
                     'Kell2018audiosetSeed2':'+',
                   'Kell2018multitask': 'o',
                        'Kell2018multitaskSeed2': 'o',
                   'ResNet50word':'s',
                     'ResNet50wordSeed2':'s',
                   'ResNet50speaker':'D',
                        'ResNet50speakerSeed2':'D',
                   'ResNet50music':'x',
                   'ResNet50audioset':'+',
                        'ResNet50audiosetSeed2':'+',
                   'ResNet50multitask': 'o',
                        'ResNet50multitaskSeed2': 'o',}

d_comp_names = {'lowfreq': 'Low frequency',
				'highfreq': 'High frequency',
				'envsounds': 'Env. sound',
				'pitch': 'Pitch',
				'music': 'Music',
				'speech': 'Speech',}

d_value_of_interest = {'median_r2_test': 'Median $R^2$',
                       'median_r2_test_c': 'Median noise-corrected $R^2$',
                       'r2_test_c': 'Median noise-corrected $R^2$',
                       'mean_r2_test_c': 'Mean noise-corrected $R^2$',
                       'median_r2_train': 'Median $R^2$ train',
                       'r2_test': 'Median $R^2$',
                       'rel_pos': 'Relative layer position',
                       'rel_layer_pos': 'Relative best layer', }

d_dim_labels = {
    # PCA covariance
    'dim_pca': 'Effective dimensionality (trained model)',
    'dim_randnetw_pca': 'Effective dimensionality (permuted network)',

    # Z-scored, PCA covariance
    'dim_zscore_pca': 'Effective dimensionality (trained model)',
    'dim_randnetw_zscore_pca': 'Effective dimensionality (permuted network)',

    # Value of interest labels
    'median_r2_test_c_agg-mean': 'Neural noise-corrected $R^2$ (trained model)',
    'median_r2_test_c_agg-mean_randnetw': 'Neural noise-corrected $R^2$ (permuted network)',
    'all_data_rsa_for_best_layer_agg-mean': 'RSA (trained model)',
    'all_data_rsa_for_best_layer_agg-mean_randnetw': 'RSA (permuted network)', }

d_sound_category_colors = {'Music': 'mediumblue',
                           'Song': 'deepskyblue',
                           'EngSpeech': 'darkgreen',
                           'ForSpeech': 'limegreen',
                           'HumVoc': 'darkviolet',
                           'AniVoc': 'orchid',
                           'HumNonVoc': 'firebrick',
                           'AniNonVoc': 'hotpink',
                           'Nature': 'gold',
                           'Mechanical': 'darkorange',
                           'EnvSound': 'grey', }

d_sound_category_names = {'Music': 'Instr. Music',
                          'Song': 'Vocal Music',
                          'EngSpeech': 'English Speech',
                          'ForSpeech': 'Foreign Speech',
                          'HumVoc': 'NonSpeech Vocal',
                          'AniVoc': 'Animal Vocal',
                          'HumNonVoc': 'Human NonVocal',
                          'AniNonVoc': 'Animal NonVocal',
                          'Nature': 'Nature',
                          'Mechanical': 'Mechanical',
                          'EnvSound': 'Env. Sounds'}

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

# For plots that order the 165 sounds into categories
sound_category_order = ['Music', 'Song', 'EngSpeech', 'ForSpeech', 'HumVoc', 'AniVoc', 'HumNonVoc',
                        'AniNonVoc', 'Nature', 'Mechanical', 'EnvSound']

d_randnetw = {'True': '_randnetw',
              'False': ''}

d_annotate = {True: '_annotated',
			  False: ''}

d_sorted = {'sort_by_performance': '_performance_sorted',
			'sort_by_manual_list': '_manually_sorted',}

NH2015_all_models_performance_order = ['ResNet50multitask', 'AST', 'VGGish', 'ResNet50audioset', 'Kell2018multitask',
									 'ResNet50word', 'ResNet50speaker', 'Kell2018speaker', 'Kell2018word',
									 'Kell2018audioset', 'Kell2018music', 'ResNet50music', 'S2T', 'wav2vec',
									 'sepformer', 'DS2', 'ZeroSpeech2020', 'DCASE2020', 'metricGAN'] # From 20230425

B2021_all_models_performance_order = ['ResNet50multitask', 'AST', 'ResNet50audioset', 'Kell2018multitask',
                                     'ResNet50word', 'VGGish', 'ResNet50speaker', 'Kell2018word',
                                     'Kell2018speaker', 'Kell2018audioset', 'Kell2018music', 'wav2vec',
                                     'ResNet50music', 'S2T', 'sepformer', 'DS2', 'DCASE2020', 'ZeroSpeech2020',
                                     'metricGAN'] # From 20230427


