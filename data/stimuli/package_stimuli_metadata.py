from pathlib import Path
import pandas as pd
import numpy as np
import os
from scipy.io import loadmat

DATADIR = (Path(os.getcwd()) / '..' ).resolve()

"""Store additional information about the 165 nat sounds"""

## Stimuli (original indexing, activations are extracted in this order) ##
sound_meta = np.load(os.path.join(DATADIR, f'neural/NH2015/neural_stim_meta.npy'))

# Extract wav names in order (this is how the neural data is ordered --> order all activations in the same way)
stimuli_IDs = []
for i in sound_meta:
	stimuli_IDs.append(i[0][:-4].decode("utf-8"))  # remove .wav

comp_data = loadmat(os.path.join(DATADIR, f'neural/NH2015comp/components.mat')) # the comp file has category labels
comp_stimuli_IDs = comp_data['stim_names']
comp_stimuli_IDs = [x[0] for x in comp_stimuli_IDs[0]]

cat_labels = comp_data['category_labels']
cat_labels = [x[0] for x in cat_labels[0]]

# create df that matches the stimuli IDs with the category labels
df_cat = pd.DataFrame(cat_labels, index=comp_stimuli_IDs, columns=['category_label'])

# reindex so it is indexed the same way as the neural data and activations (stimuli_IDs)
df_cat = df_cat.reindex(stimuli_IDs)
df_cat.to_pickle('df_stimuli_meta.pkl')