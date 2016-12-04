"""
======================================================
128 Electrodes Biosemi EEG SSVEP visualization example
======================================================

"""

import mne
import matplotlib.pyplot as plt


def visualize(data, coords):
    number_of_subplots = data.shape[0]
    fig, axes = plt.subplots(1, number_of_subplots)
    for idx in range(number_of_subplots):
        mne.viz.plot_topomap(data[idx], coords, axes=axes[idx], show=False)
    fig.tight_layout()
    fig.show()
