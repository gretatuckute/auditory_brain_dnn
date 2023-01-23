"""Generate simple df roi meta for component data"""
from pathlib import Path
import pandas as pd
import numpy as np
import os
from scipy.io import loadmat


comp_data = loadmat('components.mat')
comp_stimuli_IDs = comp_data['stim_names']
comp_stimuli_IDs = [x[0] for x in comp_stimuli_IDs[0]]

comp_names = comp_data['component_names']
comp_names = [x[0][0] for x in comp_names]

df_roi_meta = pd.DataFrame({'comp': comp_names,
							'comp_idx': np.arange(len(comp_names))})

df_roi_meta.to_pickle('df_roi_meta.pkl')