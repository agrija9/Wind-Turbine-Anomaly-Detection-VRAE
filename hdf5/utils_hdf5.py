# -*- coding: utf-8 -*-
import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np


"""
Utility functions for loading, saving,
processing and visualizing hdf5 timeseries files
"""

def path_to_sibling_folders(path):
    """
    """
    parent = path.parent
    for x in parent.iterdir():
        if x.is_dir() and x != path:
            yield x

def path_to_h5_file(data_folder):
    """
    Input: string
        data_folder: name of folder where h5 file is located
    Output: string
        full path of h5 file
    """
    count = 0
    for file in os.listdir(data_folder):
        if file.endswith(".h5"):
            # print(os.path.join(data_folder, file))
            return os.path.join(data_folder, file)
    return None

def scale_time_series(data, scaling="Normal"):
    """
    Scales data to specific range (default 0 to 1)

    Input: numpy array
        data: sensor data in time series format (samples, time_steps, features)

    Output: numpy array
        scaled_data: scaled data in time series format (samples, time_steps, features)
    """

    array_scaled_data = []

    if scaling == "Normal":
        scaler = MinMaxScaler(feature_range=(0, 1))
        for sample in data:
            scaler = scaler.fit(sample)
            scaled_data = scaler.transform(sample)
            array_scaled_data.append(scaled_data)
        return np.asarray(array_scaled_data)

    elif scaling == "Standard":
        scaler = StandardScaler()
        for sample in data:
            scaler = scaler.fit(sample)
            scaled_data = scaler.transform(sample)
            array_scaled_data.append(scaled_data)
        return np.asarray(array_scaled_data)

    else:
        print("[INFO] no support for selected scaling method (returned original data)")
        return data

def filter_timeseries_by_columns(data, columns):
    """
    Filters features in multi-variate time series data

    input (data): 3D numpy array with time series format
    input (cols): array with columns (integer encoded) to be chosen

    return (data): 3D numpy array with no. of featues = len(columns)
    """

    return data[:, :, columns]
    # return data[::, col_a: col_b + 1] # both ends

def plot_timeseries_data(data, downsample_data=False, col_name="Spn1ALxb1", single_col=False, col=0):
    """
    """
    down_sample_factor = 1

    fig = plt.gcf()
    fig.set_size_inches(14.5, 8.5)
    plt.xlabel("data_points")
    plt.grid()

    if downsample_data and not single_col:
        print("ploting full dataset downsampled " \
        "by a factor of %2d..."%(down_sample_factor))
        plt.plot(data[::down_sample_factor,:])

    elif downsample_data and single_col:
        print("ploting column %2d of dataset downsampled " \
        "by a factor of %2d..." %(col, down_sample_factor))
        plt.plot(data[::down_sample_factor,col],
                 c=np.random.rand(3,), label=col_name)
        plt.legend(loc="upper left")

    else:
        # TODO: modify this one to plot full single col (not downsampling)
        print("ploting full dataset...")
        plt.plot(data) # plot y using x as index array 0..N-1

    plt.show()

def timeseries_to_pandas(numpy_ice_data):
    return pd.DataFrame(data=numpy_ice_data)


class Color:
    def __init__(self):
        self.PURPLE = '\033[95m'
        self.CYAN = '\033[96m'
        self.DARKCYAN = '\033[36m'
        self.BLUE = '\033[94m'
        self.GREEN = '\033[92m'
        self.YELLOW = '\033[93m'
        self.RED = '\033[91m'
        self.BOLD = '\033[1m'
        self.UNDERLINE = '\033[4m'
        self.END = '\033[0m'
