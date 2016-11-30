import scipy.io as sio

def load_matlab_data(filepath, var_name=None):
    data = sio.loadmat(filepath)
    if var_name is not None:
        data = data[var_name]
    return data
