"""
======================================================
128 Electrodes Biosemi EEG SSVEP visualization example
======================================================

plot_ICA_05c.py

AMUSE ICA

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

from dataset import create_MNE_Raw


picks_ssvep = [10, 15, 21, 23, 28, 39, 81]
picks_ssvep = [i-1 for i in picks_ssvep]

###############################################################################
# Biosemi 128 electrodes data
base_dir = '../example_data'
data_filename = 'SSVEP_14Hz_Trial1_SUBJ1.MAT'

output_dir = '/tmp'


def clean_data(data_dir, data_filename, output_dir, output_delimiter='\t'):

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
    method = 'amuse'  
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

    # exclude = [0, 1, 2, 3, 5, 6]
    exclude = [0, 2, 3, 4, 6]

    ica.exclude.extend(exclude)

    # Very helpful for debug.
    # ica.plot_sources(raw,
                     # picks=list(set(range(ica.n_components))-set(exclude)))

    # Show singal before and after cleaning (first 24 seconds).
    # ica.plot_overlay(raw, stop=24.0)

    data_raw, times = raw[picks_ssvep]

    raw_cln = ica.apply(raw, exclude=exclude)

    data_cln, _ = raw_cln[picks_ssvep]

    import matplotlib.pyplot as plt

    from pyseeg.modules.fft import plot_frequency

    sources = ica._transform_raw(raw, 0, len(raw.times))
    cnt = 1
    for (i, data) in enumerate(sources):
        r, c, n = (9, 1, cnt)
        plt.subplot(r, c, n)

        plot_frequency(data, fs=raw.info['sfreq'], custom_range=None,
                       color=None, show=False)
        cnt += 1
        if cnt == 10:
            cnt = 1
            plt.show()
    plt.show()

    n_electrodes = len(picks_ssvep)

    stacked = np.insert(data_cln, np.arange(len(data_raw)), data_raw, axis=0)


    colors = ('r', 'g')
    prev_range = (0, 0)

    for (i, data) in enumerate(stacked):
        if i%2 == 0:
            prev_range = (data.min(), data.max())
        r, c, n = (n_electrodes, 2, i+1)
        plt.subplot(r, c, n)

        plt.plot(times, data, colors[(i%2)])
        plt.ylim(prev_range)
    plt.suptitle('Data from SSVEP channels before (red) an after (green) ICA cleaning')
    plt.show()

    for (i, data) in enumerate(stacked):
        r, c, n = (n_electrodes, 2, i+1)
        plt.subplot(r, c, n)

        # Remove first value (extremaly high) with custom_range.
        plot_frequency(data, fs=raw.info['sfreq'], custom_range=1,
                       color=colors[(i%2)], show=False)
        plt.xlim(0,40)
    plt.suptitle('Data from SSVEP channels before (red) an after (green) ICA cleaning')
    plt.show()


    import pyseeg.modules.spectrogram as sg
    for (i, data) in enumerate(stacked):
        r, c, n = (n_electrodes, 2, i+1)
        plt.subplot(r, c, n)

        # Remove first value (extremaly high) with custom_range.
        sg.spectrogram(data, raw.info['sfreq'], show_plot=False)
    plt.suptitle('Data from SSVEP channels before (red) an after (green) ICA cleaning')
    plt.show()

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


clean_data(data_dir=base_dir, data_filename=data_filename,
           output_dir=output_dir)
