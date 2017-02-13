"""
======================================================
128 Electrodes Biosemi EEG SSVEP visualization example
======================================================

plot_ICA_02a.py

Version a states for the easiest possible plot of the components.
Using custom matlab data from 128 electrodes biosemi eeg.

Based on:
http://martinos.org/mne/stable/auto_tutorials/plot_artifacts_correction_ica.html

"""

import sys
sys.path.append('..')

import numpy as np

import mne
from mne.datasets import sample

from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs

from dataset import create_MNE_Raw


###############################################################################
# Biosemi 128 electrodes data
data_filepath = '../example_data/SSVEP_14Hz_Trial1_SUBJ1.MAT'
var_name = 'EEGdata'
kind = 'biosemi128'
raw, _ = create_MNE_Raw(data_filepath, var_name, kind, sfreq=256)

raw.filter(1, 40, n_jobs=2)

picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,
                           stim=False, exclude='bads')


###############################################################################
# Before applying artifact correction please learn about your actual artifacts
# by reading :ref:`tut_artifacts_detect`.

###############################################################################
# Fit ICA
# -------
#
# ICA parameters:

n_components = 25  # if float, select n_components by explained variance of PCA
method = 'fastica'  # for comparison with EEGLAB try "extended-infomax" here
decim = 3  # we need sufficient statistics, not all time points -> saves time

# we will also set state of the random number generator - ICA is a
# non-deterministic algorithm, but we want to have the same decomposition
# and the same order of components each time this tutorial is run
random_state = 23

###############################################################################
# Define the ICA object instance
ica = ICA(n_components=n_components, method=method, random_state=random_state)
print(ica)

###############################################################################
# we avoid fitting ICA on crazy environmental artifacts that would
# dominate the variance and decomposition
reject = dict(mag=5e-12, grad=4000e-13)
ica.fit(raw, picks=picks_eeg, decim=decim, reject=reject)
print(ica)

###############################################################################
# Plot ICA components

# ica_basic.plot_components()  # can you spot some potential bad guys?
data = np.dot(ica.mixing_matrix_.T,
              ica.pca_components_[:ica.n_components_])


############################################
#
# Plot data
#
from dataset import load_coords
from visualize import visualize
coords = load_coords('../example_data/128_Biosemi_coords.txt').T[:2].T
visualize(data[:10], coords)
