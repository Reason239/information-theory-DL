import numpy as np


def add_noise(arr, noise_var):
    return arr + np.random.normal(size=arr.shape, scale=noise_var)


def get_info_ind_log(epochs, num_inf):
    """Get exactly num_inf integers from 0 to epochs approximately like in np.logspace"""
    num = num_inf
    ind = np.unique(np.round(np.logspace(start=0, stop=np.log10(epochs - 1), num=num).astype(dtype=np.int)))
    while len(ind) < num_inf:
        num += 1
        ind = np.unique(np.round(np.logspace(start=0, stop=np.log10(epochs - 1), num=num).astype(dtype=np.int)))
    return ind
