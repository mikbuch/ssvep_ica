import numpy as np
from dataset import load_matlab_data
import matplotlib.pyplot as plt
import pyseeg.modules.filterlib as flt
from sklearn.decomposition import FastICA

filepath = 'SSVEP_14Hz_Trial1_SUBJ1.MAT'
var_name = 'EEGdata'
electrodes = np.array([10, 15, 19, 23, 28, 39])
# Electrodes are count from 1, rows in an array from 0.
electrodes -= 1

data_init = load_matlab_data(filepath, var_name)
data_electrodes = data_init[electrodes]

data_filtered = flt.filter_eeg(
    data=data_electrodes,
    fs=256.0,
    bandstop=(49, 51),
    bandpass=(1, 50)
    )

ica = FastICA(n_components=3)
data_components = ica.fit_transform(data_filtered.T).T

plt.subplot(311)
plt.plot(data_components[0])
plt.subplot(312)
plt.plot(data_components[1])
plt.subplot(313)
plt.plot(data_components[2])
plt.show()
