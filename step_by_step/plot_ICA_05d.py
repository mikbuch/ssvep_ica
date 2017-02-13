"""
======================================================
128 Electrodes Biosemi EEG SSVEP visualization example
======================================================

plot_ICA_05d.py

Automated AMUSE ICA.

Based on:
http://martinos.org/mne/stable/auto_tutorials/plot_artifacts_correction_ica.html

"""

import sys
sys.path.append('..')

import numpy as np
import os
from os.path import join, splitext

import mne
from mne.datasets import sample

from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs

import matplotlib.pyplot as plt

from dataset import create_MNE_Raw
from pyseeg.modules.fft import plot_frequency
import pyseeg.modules.spectrogram as sg


import psutil
process = psutil.Process(os.getpid())


picks_ssvep = [10, 15, 21, 23, 28, 39, 81]
picks_ssvep = [i-1 for i in picks_ssvep]

methods = ['fastica', 'amuse', 'infomax', 'extended-infomax']

exclude_components_lists = ([0],
                            [0, 6],
                            [0, 1, 6],
                            [0, 4, 6],
                            [0, 5, 6],
                            [0, 1, 5, 6],
                            [0, 1, 2, 6],
                            [0, 2, 4, 6],
                            [0, 1, 2, 5, 6],
                            [0, 2, 3, 4, 6])


###############################################################################
# Biosemi 128 electrodes data
# base_dir = '/home/jesmasta/ICA_noise_removal'
base_dir = '/tmp/ICA_noise_removal'

input_dir_rel = 'stimuli'
output_dir_rel = 'analyses'


def clean_data(data_dir, data_filename, output_dir=None, output_dir_prefix=None,
               output_delimiter='\t', method='fastica',
               exclude_components=None):

    data_filepath = join(data_dir, data_filename)

    basename, _ = splitext(data_filename)

    output_raw_filepath = join(output_dir, basename + '_raw.txt')
    output_cln_filepath = join(output_dir, basename + '_cleaned.txt')

    var_name = 'EEGdata'
    kind = 'biosemi128'

    raw = create_MNE_Raw(data_filepath, var_name, kind, sfreq=256)

    raw.filter(1, 40, n_jobs=2)

    picks_eeg = np.array(picks_ssvep)



    ###########################################################################
    # Fit ICA
    # -------
    #
    # ICA parameters:

    # if float, select n_components by explained variance of PCA
    n_components = 7  
    # for comparison with EEGLAB try "extended-infomax" here
    method = method
    # we need sufficient statistics, not all time points -> saves time
    decim = 3  

    # we will also set state of the random number generator - ICA is a
    # non-deterministic algorithm, but we want to have the same decomposition
    # and the same order of components each time this tutorial is run
    random_state = 23

    ###########################################################################
    # Define the ICA object instance
    ica = ICA(n_components=n_components, method=method,
              random_state=random_state)
    print(ica)

    ###########################################################################
    # we avoid fitting ICA on crazy environmental artifacts that would
    # dominate the variance and decomposition
    reject = dict(mag=5e-12, grad=4000e-13)
    ica.fit(raw, picks=picks_eeg, decim=decim, reject=reject)
    print(ica)

    ###########################################################################
    # Plot ICA components

    ############################################
    #
    # Plot data
    #

    # Plot components (all of them).
    # ica.plot_components()  # can you spot some potential bad guys?
    # For interactive usage:
    # ica.plot_components(inst=raw)  # can you spot some potential bad guys?

    ica.exclude.extend(exclude_components)

    # Very helpful for debug.
    # ica.plot_sources(raw,
                     # picks=list(set(range(ica.n_components))-set(exclude)))

    # Show singal before and after cleaning (first 24 seconds).
    # ica.plot_overlay(raw, stop=24.0)

    data_raw, times = raw[picks_ssvep]

    raw_cln = ica.apply(raw, exclude=exclude_components)

    data_cln, _ = raw_cln[picks_ssvep]


    ############################################
    #
    # Save cleaned data to file
    #
    # Create header: electrode names
    header = [ch for (i, ch) in enumerate(raw.info['ch_names']) if i in picks_eeg]
    header = output_delimiter.join(header)


    np.savetxt(output_raw_filepath, data_raw.T, delimiter=output_delimiter,
               header=header, comments='', fmt='%.8f')
    np.savetxt(output_cln_filepath, data_cln.T, delimiter=output_delimiter,
               header=header, comments='', fmt='%.8f')


###############################################################################
#
#       MAIN LOOP
#

input_dir = os.path.join(base_dir, input_dir_rel)

for method in methods:
    for exclude_components in exclude_components_lists:
        for root, dirs, files in os.walk(input_dir):
            if files:
                input_sub = join(input_dir, root)
                sub_dir = os.path.basename(root)

                components_listed = ''.join([str(i) + '_'
                                             for i in exclude_components])
                analysis_dir = os.path.join(method, components_listed)
                output_dir = os.path.join(base_dir, output_dir_rel,
                                          analysis_dir)

                output_sub = join(output_dir, sub_dir)

                if not os.path.exists(output_sub):
                    os.makedirs(output_sub)

                for name in files:
                    clean_data(data_dir=input_sub, data_filename=name,
                               output_dir=output_sub, method=method,
                               exclude_components=exclude_components)
