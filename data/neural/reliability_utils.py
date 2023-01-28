from pathlib import Path
import pandas as pd
import numpy as np
import os
import pickle
from scipy.linalg import norm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


### FUNCTION(S) ###
def plot_sessionwise_split_r(arr1, arr2, label1, label2, min=-0.25, alpha=0.3, save=None):
	"""
	Plot scatter with x=y line, square plot
	"""
	plt.figure(figsize=(6, 6))
	plt.plot([min, 1], [min, 1], color='red')
	plt.scatter(arr1, arr2, alpha=alpha)
	plt.xlim([min, 1])
	plt.ylim([min, 1])
	plt.xlabel(label1)
	plt.ylabel(label2)
	plt.show()


def corrected_reliability(voxel_data, split):
	"""
	Estimate spearman-brown corrected reliability based on just two scans (split half reliability)

	split_idx = lists of indices, e.g. [0,1]
	"""
	
	corrected_rv_lst = []
	n_vox = voxel_data.shape[1]
	
	for i in range(0, (n_vox)):
		v = voxel_data[:, i, :]  # all sounds for a given vox across all sessions (reps)

		split_rv = np.corrcoef(v[:, split[0]], v[:, split[1]])[1, 0]

		# corrected_rv = max(3 * split_rv_median / (1 + 2 * split_rv_median), 0.128) # k-corrected
		corrected_rv = 2 * split_rv / (1 + 1 * split_rv)
		corrected_rv_lst.append(corrected_rv)
	
	return corrected_rv_lst