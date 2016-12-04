import scipy.io as sio
import numpy as np
from sklearn import preprocessing

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
