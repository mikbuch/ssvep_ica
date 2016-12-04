"""
======================================================
128 Electrodes Biosemi EEG SSVEP visualization example
======================================================

plot_ICA_01.py

It was good intuition by I made few mistakes.
Afterwards I was inspired by this example:
http://martinos.org/mne/stable/auto_tutorials/plot_artifacts_correction_ica.html?highlight=picks

And I made a better version.
"""

import sys
sys.path.append('..')

import numpy as np
from sklearn.decomposition import FastICA

import pyseeg.modules.filterlib as flt

from dataset import load_coords, load_matlab_data
from visualize import visualize

############################################
#
# Load and analyze data
#
# electrodes = np.array([10, 15, 19, 23, 28, 39])
filepath = '../example_data/SSVEP_14Hz_Trial1_SUBJ1.MAT'
var_name = 'EEGdata'
bandstop = (49, 50)
bandpass = (1, 50)
fs = 256.0

data_raw = load_matlab_data(filepath, var_name)

data_filtered = flt.filter_eeg(
    data=data_raw,
    fs=fs,
    bandstop=bandstop,
    bandpass=bandpass
    )

data = data_filtered

n_components = data.shape[0]

ica = FastICA(n_components=n_components, max_iter=1000)
ica.fit_transform(data.T)

components = ica.components_

############################################
#
# Plot data
#
coords = load_coords('../example_data/128_Biosemi_coords.txt').T[:2].T
visualize(components[:4], coords)
