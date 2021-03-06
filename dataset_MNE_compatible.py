import scipy.io as sio
import numpy as np
from sklearn import preprocessing

import mne

def load_matlab_data(filepath, var_name=None):
    data = sio.loadmat(filepath)
    if var_name is not None:
        data = data[var_name]
    return data

def load_coords(filepath, delimiter=',', scale_0_1=True):
    coords = np.loadtxt(filepath, delimiter=delimiter)
    if scale_0_1:
        coords = preprocessing.MinMaxScaler().fit_transform(coords)

    return coords

def create_MNE_Raw(data_filepath, var_name, kind,
                   sfreq, delimiter_data=',', dbg=False):
    """
    Based on: http://stackoverflow.com/a/38634620
    """

    # Read the CSV file as a NumPy array.
    data = load_matlab_data(data_filepath, var_name=var_name)

    # Sampling rate of the machine [Hz].
    sfreq = sfreq  

    # Channel types. From documentation:
    '''
    ch_types : list of str | str
        Channel types. If None, data are assumed to be misc.
        Currently supported fields are 'ecg', 'bio', 'stim', 'eog', 'misc',
        'seeg', 'ecog', 'mag', 'eeg', 'ref_meg', 'grad', 'hbr' or 'hbo'.
        If str, then all channels are assumed to be of the same type.
    '''
    ch_types = 'eeg'

    montage = mne.channels.read_montage(kind)

    # Create the info structure needed by MNE
    info = mne.create_info(montage.ch_names, sfreq, ch_types, montage)

    # Read montage.
    # 3D montage ==> 2D montage
    # https://gist.github.com/wmvanvliet/6d7c78ea329d4e9e1217
    #  info = mne.create_info(ch_names, sfreq, ch_types, montage)
    layout = mne.channels.make_eeg_layout(info)
    layout.pos = layout.pos[:-3]

    # Update pos to 2D scheme.
    montage.pos = layout.pos
    # Remove last 3 electrodes.
    montage.ch_names = montage.ch_names[:-3]

    info = mne.create_info(montage.ch_names, sfreq, ch_types, montage)

    # Finally, create the Raw object
    raw = mne.io.RawArray(data, info)

    if dbg:
        # Plot it.
        raw.plot()

    return raw, layout
