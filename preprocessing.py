# -*- coding: utf-8 -*-

import os
import pandas as pd
from scipy import signal
import numpy as np


def import_data():
    data = {}    
    # Collecting all the relative file paths in a single list
    files = []
    for (dirpath, dirnames, filenames) in os.walk('./tctodd/'):
        if dirpath != './tctodd/':  # Not considering files not in a subdirectory
            files += [(dirpath + '/' + f) for f in filenames]
    
    # Processing each file
    for f in files:
        # Opening the file
        file = pd.read_csv(f, sep='\t')
        # Getting the label
        l = os.path.basename(f).split('-')[0]
        # Adding the data to the list for that particular class label
        fixed_length_file = signal.resample(file, 57, axis=0)
        data[l] = data.get(l, []) + [fixed_length_file]
    return data


def create_data_tensor():
    """
    Converts data dictionary created by io.load_data into a 3D tensor.
    Args:
        data: Dictionary with signs as keys and values as list of sign 
            instances.
    Returns:
        X: Tensor of data, where axis 0 corresponds to each data point,
            axis 1 corresponds to features (signals), and axis 2 corresponds
            to signal data across time.
        y: Array of length(num data points), each element corresponding to 
            a class
        class_names: dictionary where each key is a class (0, 1, 2...) and each value
            is the class label
    """
    data = import_data()
    
    class_names = {}
    class_labels = {}
    num_features = 22  # CHANGE FOR LOW QUALITY DATA

    num_samples = 0
    for i, sign in enumerate(data.keys()):
        class_labels[sign] = i
        class_names[i] = sign
        num_samples += len(data[sign])

    X = np.zeros((num_samples, num_features, 57))
    y = np.zeros((num_samples), dtype=np.uint)

    sample_idx = 0
    for i, sign in enumerate(data.keys()):
        for d in data[sign]:
            X[sample_idx, :, :] = d.T
            y[sample_idx] = class_labels[sign]
            sample_idx += 1

    return X, y, class_names


def flatten_data(X):
    return X.swapaxes(1,2).reshape((X.shape[0], X.shape[1]*X.shape[2]))

        