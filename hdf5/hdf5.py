# -*- coding: utf-8 -*-
import os
import time
import numpy as np
from pathlib import Path
import h5py
from tqdm import tqdm


class H5DataSet:
    """
    Represents an abstract HDF5 dataset.

    This class allows the user to read headers and sensor data from hdf5 files
    into numpy arrays.

    Input params:
    """
    def __init__(self, data_folder_path, file_path,
                 min_sequence_length, target_sequence_length):

        self.data_folder_path = data_folder_path
        self.file_path = file_path
        self.min_sequence_length = min_sequence_length
        self.target_sequence_length = target_sequence_length

        assert(Path(self.data_folder_path).is_dir())

    def get_headers(self):
        with h5py.File(self.file_path, "r") as f:
            headers = list(f.keys())
            return headers

    def get_weight_configs(self):
        with h5py.File(self.file_path, "r") as f:
            data = f.get("data")
            weight_configurations = list(data.keys())
            return weight_configurations

    def get_sensor_readings(self, npy_file_name, save_npy,
                            downsample_factor, zone_ice):
        """
        This method stacks the 27 sensor readings at each weight configuration.
        It gets only those time series with at least "length_time_series"
        no. of points.

        Returns: np array
            sensor_columns_data with shape (n, m)
            n: total time steps (9601 for each weight configuration)
            m: measured sensor variables
        """
        with h5py.File(self.file_path, "r") as f:

            data = f.get("data")
            weight_configurations = list(data.keys())

            # stack sensor readings for weight configs
            sensor_columns_data = []
            sampled_weight_configurations = []

            # define normal and abnormal configurations (2 class anomaly detection problem)
            normal_configs = ["_0.0-0.0-0.0"]

            # define abnormal configs based on specified zone
            if zone_ice == 3:
                abnormal_configs = ['_0.0-0.0-0.4', '_0.0-0.0-0.8', '_0.0-0.0-1.2',
                                    '_0.0-0.0-1.6', '_0.0-0.0-2.0', '_0.0-0.0-2.4',
                                    '_0.0-0.0-2.8', '_0.0-0.0-3.2', '_0.0-0.0-3.6',
                                    '_0.0-0.0-4.0']

            elif zone_ice == 2:
                abnormal_configs = ['_0.0-0.4-0.0', '_0.0-0.8-0.0', '_0.0-1.2-0.0',
                                    '_0.0-1.6-0.0', '_0.0-2.0-0.0', '_0.0-2.4-0.0',
                                    '_0.0-2.8-0.0', '_0.0-3.2-0.0', '_0.0-3.6-0.0',
                                    '_0.0-4.0-0.0']

            else:
                abnormal_configs = ['_0.4-0.0-0.0', '_0.8-0.0-0.0', '_1.2-0.0-0.0',
                                    '_1.6-0.0-0.0', '_2.0-0.0-0.0', '_2.4-0.0-0.0',
                                    '_2.8-0.0-0.0', '_3.2-0.0-0.0', '_3.6-0.0-0.0',
                                    '_4.0-0.0-0.0']

            start = time.time()
            print("-----------------------------------------------------------")
            print(">>>Reading sensor columns (downsample factor {0})<<<".format(downsample_factor))
            for i in tqdm(range(0, len(weight_configurations))):
                if i % downsample_factor == 0 and weight_configurations[i] in normal_configs \
                                              or weight_configurations[i] in abnormal_configs:
                    weight_k = data.get(weight_configurations[i])
                    values_weight_k = np.array(weight_k) # cast hdf5 into numpy array
                    if values_weight_k[1001::,].shape[0] == self.target_sequence_length:
                        print("{}-th weight configuration".format(i), weight_configurations[i])
                        print("current {}-th weight configuration:".format(i), values_weight_k.shape)
                        print("saving {}-th weight configuration".format(i))
                        print("target sequence length:", self.target_sequence_length)
                        print("time series shape:", values_weight_k.shape)
                        print()
                        # append first 10,000 data points for each simulation
                        # sensor_columns_data.append(values_weight_k[1001::,])
                        sensor_columns_data.append(values_weight_k[0:self.min_sequence_length,])

                    if weight_configurations[i] in abnormal_configs:
                        abnormal_configs.remove(weight_configurations[i])
                        print("current abnormal configs:", abnormal_configs)

            if not sensor_columns_data:
                # TODO: This workaround is not working as expected
                print("Sensor columns data list empty: NO simulation with target sequence length")
                print("adding random data as a REPLACEMENT")
                sensor_columns_data.append(np.random.rand(11*self.min_sequence_length, 27))

            # else:
            sensor_columns_data = np.vstack(sensor_columns_data)
            print("sensor columns stacked data:", sensor_columns_data.shape)

            elapsed_time_fl = (time.time() - start)
            print("[INFO] Ellapsed time to stack sensor readings:", elapsed_time_fl)
            print("-----------------")

            # save data into npy format
            if save_npy:
                if not os.path.exists(os.path.join(self.data_folder_path, "npy_data")):
                    os.makedirs(os.path.join(self.data_folder_path, "npy_data"))

                np.save(os.path.join(self.data_folder_path,
                                     "npy_data", npy_file_name),
                                     sensor_columns_data)

                print("Saved sensor data into npy format...")

            if downsample_factor:
                return sensor_columns_data, weight_configurations
            else:
                return sensor_columns_data, _


# class MultiClassH5DataSet:
#     """
#     Represents an abstract HDF5 dataset.
#
#     This class allows the user to read headers and sensor data from hdf5 files
#     into numpy arrays.
#
#     Input params:
#     """
#     def __init__(self, data_folder_path, file_path,
#                  min_sequence_length, target_sequence_length):
#
#         self.data_folder_path = data_folder_path
#         self.file_path = file_path
#         self.min_sequence_length = min_sequence_length
#         self.target_sequence_length = target_sequence_length
#
#         assert(Path(self.data_folder_path).is_dir())
#
#     def get_headers(self):
#         with h5py.File(self.file_path, "r") as f:
#             headers = list(f.keys())
#             return headers
#
#     def get_weight_configs(self):
#         with h5py.File(self.file_path, "r") as f:
#             data = f.get("data")
#             weight_configurations = list(data.keys())
#             return weight_configurations
#
#     def get_sensor_readings(self, npy_file_name, save_npy, downsample_factor):
#         """
#         This method stacks the 27 sensor readings at each weight configuration.
#         It gets only those time series with at least "length_time_series"
#         no. of points.
#
#         Returns: np array
#             sensor_columns_data with shape (n, m)
#             n: total time steps (9601 for each weight configuration)
#             m: measured sensor variables
#         """
#         with h5py.File(self.file_path, "r") as f:
#
#             data = f.get("data")
#             weight_configurations = list(data.keys())
#
#             # stack sensor readings for weight configs
#             sensor_columns_data = []
#             sampled_weight_configurations = []
#
#             # define normal and abnormal configurations (2 class anomaly detection problem)
#             normal_configs = ["_0.0-0.0-0.0"]
#
#             # take abnormal cases from all 3 zones
#             abnormal_configs = ["_0.0-0.0-0.2", '_0.0-0.0-0.4', "_0.0-0.0-0.6", '_0.0-0.0-0.8', # zone 3
#                                 "_0.0-0.0-1.0", "_0.0-0.0-1.2", "_0.0-0.0-1.4", "_0.0-0.0-1.6",
#                                 "_0.0-0.0-1.8", '_0.0-0.0-2.0', "_0.0-0.0-2.2", '_0.0-0.0-2.4',
#                                 "_0.0-0.0-2.6", '_0.0-0.0-2.8', "_0.0-0.0-3.0", '_0.0-0.0-3.2',
#                                 "_0.0-0.0-3.4", '_0.0-0.0-3.6', "_0.0-0.0-3.8", '_0.0-0.0-4.0',
#                                 "_0.0-0.2-0.0", '_0.0-0.4-0.0', "_0.0-0.6-0.0", '_0.0-0.8-0.0', # zone 2
#                                 "_0.0-1.0-0.0", '_0.0-1.2-0.0', "_0.0-1.4-0.0", '_0.0-1.6-0.0',
#                                 "_0.0-1.8-0.0", '_0.0-2.0-0.0', "_0.0-2.2-0.0", '_0.0-2.4-0.0',
#                                 "_0.0-2.6-0.0", '_0.0-2.8-0.0', "_0.0-3.0-0.0", '_0.0-3.2-0.0',
#                                 "_0.0-3.4-0.0", '_0.0-3.6-0.0', "_0.0-3.8-0.0", '_0.0-4.0-0.0',
#                                 "_0.2-0.0-0.0", '_0.4-0.0-0.0', "_0.6-0.0-0.0", "_0.8-0.0-0.0", # zone 1
#                                 '_1.0-0.0-0.0', '_1.2-0.0-0.0', "_1.4-0.0-0.0", '_1.6-0.0-0.0',
#                                 "_1.8-0.0-0.0", '_2.0-0.0-0.0', "_2.2-0.0-0.0", '_2.4-0.0-0.0',
#                                 "_2.6-0.0-0.0", '_2.8-0.0-0.0', "_3.0-0.0-0.0", '_3.2-0.0-0.0',
#                                 "_3.4-0.0-0.0", '_3.6-0.0-0.0', "_3.8-0.0-0.0", '_4.0-0.0-0.0']
#
#             # NOTE: you are not doing anything with these lists (?)
#             abnormal_configs_3 = ["_0.0-0.0-0.2", '_0.0-0.0-0.4', "_0.0-0.0-0.6", '_0.0-0.0-0.8',
#                                   "_0.0-0.0-1.0", "_0.0-0.0-1.2", "_0.0-0.0-1.4", "_0.0-0.0-1.6",
#                                   "_0.0-0.0-1.8", '_0.0-0.0-2.0', "_0.0-0.0-2.2", '_0.0-0.0-2.4',
#                                   "_0.0-0.0-2.6", '_0.0-0.0-2.8', "_0.0-0.0-3.0", '_0.0-0.0-3.2',
#                                   "_0.0-0.0-3.4", '_0.0-0.0-3.6', "_0.0-0.0-3.8", '_0.0-0.0-4.0']
#
#             abnormal_configs_2 = ["_0.0-0.2-0.0", '_0.0-0.4-0.0', "_0.0-0.6-0.0", '_0.0-0.8-0.0',
#                                   "_0.0-1.0-0.0", '_0.0-1.2-0.0', "_0.0-1.4-0.0", '_0.0-1.6-0.0',
#                                   "_0.0-1.8-0.0", '_0.0-2.0-0.0', "_0.0-2.2-0.0", '_0.0-2.4-0.0',
#                                   "_0.0-2.6-0.0", '_0.0-2.8-0.0', "_0.0-3.0-0.0", '_0.0-3.2-0.0',
#                                   "_0.0-3.4-0.0", '_0.0-3.6-0.0', "_0.0-3.8-0.0", '_0.0-4.0-0.0']
#
#             abnormal_configs_1 = ["_0.2-0.0-0.0", '_0.4-0.0-0.0', "_0.6-0.0-0.0", "_0.8-0.0-0.0",
#                                   '_1.0-0.0-0.0', '_1.2-0.0-0.0', "_1.4-0.0-0.0", '_1.6-0.0-0.0',
#                                   "_1.8-0.0-0.0", '_2.0-0.0-0.0', "_2.2-0.0-0.0", '_2.4-0.0-0.0',
#                                   "_2.6-0.0-0.0", '_2.8-0.0-0.0', "_3.0-0.0-0.0", '_3.2-0.0-0.0',
#                                   "_3.4-0.0-0.0", '_3.6-0.0-0.0', "_3.8-0.0-0.0", '_4.0-0.0-0.0']
#
#             start = time.time()
#             print("-----------------------------------------------------------")
#             print(">>>Reading sensor columns (downsample factor {0})<<<".format(downsample_factor))
#             for i in tqdm(range(0, len(weight_configurations))):
#                 if i % downsample_factor == 0 and weight_configurations[i] in normal_configs \
#                                               or weight_configurations[i] in abnormal_configs:
#
#                     weight_k = data.get(weight_configurations[i])
#                     values_weight_k = np.array(weight_k) # cast hdf5 into numpy array
#
#                     # check time series has a target sequence length
#                     if values_weight_k[1001::,].shape[0] == self.target_sequence_length:
#                         print("{}-th weight configuration".format(i), weight_configurations[i])
#                         print("current {}-th weight configuration:".format(i), values_weight_k.shape)
#                         print("saving {}-th weight configuration".format(i))
#                         print("target sequence length:", self.target_sequence_length)
#                         print("time series shape:", values_weight_k.shape)
#                         print()
#
#                         # append first 10,000 data points for each simulation
#                         sensor_columns_data.append(values_weight_k[0:self.min_sequence_length,])
#                         print("sensor columns data shape:", np.shape(sensor_columns_data))
#                         print()
#
#                     if weight_configurations[i] in abnormal_configs:
#                         abnormal_configs.remove(weight_configurations[i])
#                         print("current abnormal configs:", abnormal_configs)
#                         print()
#
#             # if array is empty it means no simulations with target sequence length
#             if not sensor_columns_data:
#                 # TODO: This workaround is not working as expected
#                 print("Sensor columns data list empty: NO simulation with target sequence length")
#                 print("adding random data as a REPLACEMENT")
#                 print()
#                 sensor_columns_data.append(np.random.rand(11*self.min_sequence_length, 27))
#
#             # else stack data into general sensor columns data
#             sensor_columns_data = np.vstack(sensor_columns_data)
#             print("sensor columns stacked data:", sensor_columns_data.shape)
#             print()
#
#             elapsed_time_fl = (time.time() - start)
#             print("[INFO] Ellapsed time to stack sensor readings:", elapsed_time_fl)
#             print("-----------------")
#
#             # save data into npy format
#             if save_npy:
#                 if not os.path.exists(os.path.join(self.data_folder_path, "npy_data")):
#                     os.makedirs(os.path.join(self.data_folder_path, "npy_data"))
#
#                 np.save(os.path.join(self.data_folder_path,
#                                      "npy_data", npy_file_name),
#                                      sensor_columns_data)
#
#                 print("Saved sensor data into npy format...")
#
#             if downsample_factor:
#                 return sensor_columns_data, weight_configurations
#             else:
#                 return sensor_columns_data, _

class MultiClassH5DataSet:
    """
    Represents an abstract HDF5 dataset.

    This class allows the user to read headers and sensor data from hdf5 files
    into numpy arrays.

    Input params:
    """
    def __init__(self, data_folder_path, file_path,
                 min_sequence_length, target_sequence_length):

        self.data_folder_path = data_folder_path
        self.file_path = file_path
        self.min_sequence_length = min_sequence_length
        self.target_sequence_length = target_sequence_length

        assert(Path(self.data_folder_path).is_dir())

    def get_headers(self):
        with h5py.File(self.file_path, "r") as f:
            headers = list(f.keys())
            return headers

    def get_weight_configs(self):
        with h5py.File(self.file_path, "r") as f:
            data = f.get("data")
            weight_configurations = list(data.keys())
            return weight_configurations

    def get_sensor_readings(self, downsample_factor):
        """
        This method stacks the 27 sensor readings at each weight configuration.
        It gets only those time series with at least "length_time_series" no. of points.
        Returns normal and abnormal cases as numpy arrays.
        """
        with h5py.File(self.file_path, "r") as f:
            data = f.get("data")
            weight_configurations = list(data.keys())

            # stack sensor readings based on weight zone (return all arrays separately for easier processing)
            normal_sensor_data = []
            abnormal_zone3_sensor_data = []
            abnormal_zone2_sensor_data = []
            abnormal_zone1_sensor_data = []

            sampled_weight_configurations = []

            # define normal and abnormal configurations
            normal_configs = ["_0.0-0.0-0.0"]

            # take abnormal cases from all 3 zones
            abnormal_configs = ["_0.0-0.0-0.2", '_0.0-0.0-0.4', "_0.0-0.0-0.6", '_0.0-0.0-0.8', # zone 3
                                "_0.0-0.0-1.0", "_0.0-0.0-1.2", "_0.0-0.0-1.4", "_0.0-0.0-1.6",
                                "_0.0-0.0-1.8", '_0.0-0.0-2.0', "_0.0-0.0-2.2", '_0.0-0.0-2.4',
                                "_0.0-0.0-2.6", '_0.0-0.0-2.8', "_0.0-0.0-3.0", '_0.0-0.0-3.2',
                                "_0.0-0.0-3.4", '_0.0-0.0-3.6', "_0.0-0.0-3.8", '_0.0-0.0-4.0',
                                "_0.0-0.2-0.0", '_0.0-0.4-0.0', "_0.0-0.6-0.0", '_0.0-0.8-0.0', # zone 2
                                "_0.0-1.0-0.0", '_0.0-1.2-0.0', "_0.0-1.4-0.0", '_0.0-1.6-0.0',
                                "_0.0-1.8-0.0", '_0.0-2.0-0.0', "_0.0-2.2-0.0", '_0.0-2.4-0.0',
                                "_0.0-2.6-0.0", '_0.0-2.8-0.0', "_0.0-3.0-0.0", '_0.0-3.2-0.0',
                                "_0.0-3.4-0.0", '_0.0-3.6-0.0', "_0.0-3.8-0.0", '_0.0-4.0-0.0',
                                "_0.2-0.0-0.0", '_0.4-0.0-0.0', "_0.6-0.0-0.0", "_0.8-0.0-0.0", # zone 1
                                '_1.0-0.0-0.0', '_1.2-0.0-0.0', "_1.4-0.0-0.0", '_1.6-0.0-0.0',
                                "_1.8-0.0-0.0", '_2.0-0.0-0.0', "_2.2-0.0-0.0", '_2.4-0.0-0.0',
                                "_2.6-0.0-0.0", '_2.8-0.0-0.0', "_3.0-0.0-0.0", '_3.2-0.0-0.0',
                                "_3.4-0.0-0.0", '_3.6-0.0-0.0', "_3.8-0.0-0.0", '_4.0-0.0-0.0']

            abnormal_configs_3 = ["_0.0-0.0-0.2", '_0.0-0.0-0.4', "_0.0-0.0-0.6", '_0.0-0.0-0.8',
                                  "_0.0-0.0-1.0", "_0.0-0.0-1.2", "_0.0-0.0-1.4", "_0.0-0.0-1.6",
                                  "_0.0-0.0-1.8", '_0.0-0.0-2.0', "_0.0-0.0-2.2", '_0.0-0.0-2.4',
                                  "_0.0-0.0-2.6", '_0.0-0.0-2.8', "_0.0-0.0-3.0", '_0.0-0.0-3.2',
                                  "_0.0-0.0-3.4", '_0.0-0.0-3.6', "_0.0-0.0-3.8", '_0.0-0.0-4.0']

            abnormal_configs_2 = ["_0.0-0.2-0.0", '_0.0-0.4-0.0', "_0.0-0.6-0.0", '_0.0-0.8-0.0',
                                  "_0.0-1.0-0.0", '_0.0-1.2-0.0', "_0.0-1.4-0.0", '_0.0-1.6-0.0',
                                  "_0.0-1.8-0.0", '_0.0-2.0-0.0', "_0.0-2.2-0.0", '_0.0-2.4-0.0',
                                  "_0.0-2.6-0.0", '_0.0-2.8-0.0', "_0.0-3.0-0.0", '_0.0-3.2-0.0',
                                  "_0.0-3.4-0.0", '_0.0-3.6-0.0', "_0.0-3.8-0.0", '_0.0-4.0-0.0']

            abnormal_configs_1 = ["_0.2-0.0-0.0", '_0.4-0.0-0.0', "_0.6-0.0-0.0", "_0.8-0.0-0.0",
                                  '_1.0-0.0-0.0', '_1.2-0.0-0.0', "_1.4-0.0-0.0", '_1.6-0.0-0.0',
                                  "_1.8-0.0-0.0", '_2.0-0.0-0.0', "_2.2-0.0-0.0", '_2.4-0.0-0.0',
                                  "_2.6-0.0-0.0", '_2.8-0.0-0.0', "_3.0-0.0-0.0", '_3.2-0.0-0.0',
                                  "_3.4-0.0-0.0", '_3.6-0.0-0.0', "_3.8-0.0-0.0", '_4.0-0.0-0.0']

            start = time.time()

            print()
            print("[INFO] Started reading hdf5 file: ", self.file_path)
            for i in range(0, len(weight_configurations)):
                weight_k = data.get(weight_configurations[i])
                values_weight_k = np.array(weight_k) # cast hdf5 into numpy array

                if i % downsample_factor == 0 and values_weight_k[1001::,].shape[0] >= self.target_sequence_length: # check target sequence

                    # check normal config zone 0
                    if weight_configurations[i] in normal_configs:
                        # print("TESTING ==> ", values_weight_k[0:self.min_sequence_length,:].shape)
                        print()
                        print("saving {0}-th normal config {1}".format(i, weight_configurations[i])) # append first 10,000 data points for each simulation
                        normal_sensor_data.append(values_weight_k[0:self.min_sequence_length,:])
                        print("normal data shape:", np.shape(normal_sensor_data))

                    # check abnormal config zone 1
                    elif weight_configurations[i] in abnormal_configs_1:
                        print()
                        print("saving {0}-th abnormal config zone 1 {1}".format(i, weight_configurations[i]))
                        abnormal_zone1_sensor_data.append(values_weight_k[0:self.min_sequence_length,:])
                        print("abnormal zone 1 data shape:", np.shape(abnormal_zone1_sensor_data))

                    # check abnormal config zone 2
                    elif weight_configurations[i] in abnormal_configs_2:
                        print()
                        print("saving {0}-th abnormal config zone 2 {1}".format(i, weight_configurations[i]))
                        abnormal_zone2_sensor_data.append(values_weight_k[0:self.min_sequence_length,:])
                        print("abnormal zone 2 data shape:", np.shape(abnormal_zone2_sensor_data))

                    # check abnormal config zone 3
                    else:
                        print()
                        print("saving {0}-th abnormal config zone 3 {1}".format(i, weight_configurations[i]))
                        abnormal_zone3_sensor_data.append(values_weight_k[0:self.min_sequence_length,:])
                        print("abnormal zone 3 data shape:", np.shape(abnormal_zone3_sensor_data))

            # stack weight configs on top of each other to return numpy arrays
            normal_sensor_data = np.asarray(normal_sensor_data) # np.vstack(normal_sensor_data)
            abnormal_zone1_sensor_data = np.asarray(abnormal_zone1_sensor_data)
            abnormal_zone2_sensor_data = np.asarray(abnormal_zone2_sensor_data)
            abnormal_zone3_sensor_data = np.asarray(abnormal_zone3_sensor_data)

            print()
            print("[INFO] get_sensor_readings summary:")
            print("[INFO] normal sensor stacked data:", normal_sensor_data.shape)
            print("[INFO] abnormal zone 1 sensor stacked data:", abnormal_zone1_sensor_data.shape)
            print("[INFO] abnormal zone 2 sensor stacked data:", abnormal_zone2_sensor_data.shape)
            print("[INFO] abnormal zone 3 sensor stacked data:", abnormal_zone3_sensor_data.shape)

            elapsed_time_fl = (time.time() - start)
            print("[INFO] Ellapsed time to stack sensor readings:", elapsed_time_fl)

            # return only sensor data that fulfills the above criteria --> target sequence length
            if normal_sensor_data.data.ndim == 3 and abnormal_zone1_sensor_data.ndim == 3 and abnormal_zone2_sensor_data.ndim == 3 and abnormal_zone3_sensor_data.ndim == 3:
                return normal_sensor_data, abnormal_zone1_sensor_data, abnormal_zone2_sensor_data, abnormal_zone3_sensor_data
            else:
                print("[ERROR] One sensor data array does not have target simulations, returning None")
                return None, None, None, None
