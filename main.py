# -*- coding: utf-8 -*-
import sys
sys.path.append("../")
import argparse
import os
import time
import ast
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.multiprocessing as mp

from hdf5.hdf5 import MultiClassH5DataSet
from hdf5.utils_hdf5 import (path_to_sibling_folders, path_to_h5_file,
                            scale_time_series, filter_timeseries_by_columns,
                            Color, plot_timeseries_data)

from models.vrae.vrae import VRAE
from models.vrae.utils_vrae_multiple_classes import (plot_clustering,
                                                     split_train_val_timeseries,
                                                     plot_latent_vectors_as_lines,
                                                     plot_raw_PCA)

from clustering.clustering import Cluster

"""
Variational Recurrent Autoencoder for Timeseries Clustering.
The intended applications are data compression and anomaly detection.

This script performs the following:
1) Load hdf5 from wind turbine simulation and convert into numpy arrays
2) Convert the numpy arrays into pytorch tensors
3) Convert the tensors into appropriate format to input into neural network
4) Train neural network with given data
5) Save neural network parameters
6) Save 2D PCA/t-SNE projections of evaluation data in images folder

Note: to run the script it is convenient to create a conda environment and
install all the dependencies required above.
"""


def get_HDF5(data_path, min_sequence_length, target_sequence_length):
    """
    This function returns a dictionary containing hdf5 instances for each
    h5 file put in data folder.
    :param length_time_series: int: desired length (datapoints) of time series
    """

    # read h5 files in folder and append them as a dictionary structure
    datasets = {}
    index = 0
    for file in os.listdir(data_path):
        if file.endswith(".h5"):
            print("[INFO] instantiating hdf5 files:", file)
            datasets["{0}".format(index)] = MultiClassH5DataSet(data_path,
                                                                os.path.join(data_path, file),
                                                                min_sequence_length,
                                                                target_sequence_length)
            index += 1

    print("[INFO] HDF5 datasets dictionary created correctly")
    return datasets

def get_data_headers(file_path):
    """
    Retrieve data headers from txt file (in case file is provided).
    This can be used for plotting of specific sensor column in dataset.
    """
    data_header_keys = []
    data_header_values = []

    # open txt file with headers information
    f = open(file_path, "r") # hard-coded paths
    x = [line.rstrip() for line in f]
    # x = f.readlines()
    f.close()

    # create lists containing each string row (remove "[" "]")
    for i in range(len(x)):
        x[i] = x[i].strip('][').split(',')

    # flatten list of lists
    flat_list = [item.strip() for sublist in x for item in sublist]

    # remove whitespaces
    flat_list = list(filter(None, flat_list))

    # remove quotes using eval and append to headers array
    for i in range(len(flat_list)):
        data_header_keys.append(eval(flat_list[i]))
        data_header_values.append(i)

    # create dictionary using keys and values
    data_headers = dict(zip(data_header_keys, data_header_values))

    return data_headers


def main():

    parser = argparse.ArgumentParser(description="VRAE time series clustering")

    parser.add_argument("-df", "--downsample_factor",
                        dest="downsample_factor", default=1, type=int,
                        help="downsample no. timeseries by a factor")

    parser.add_argument("-hs1", "--hidden_size_1",
                        dest="hidden_size_1", default=90, type=int,
                        help="the number of features in the hidden state h")

    parser.add_argument("-hld", "--hidden_layer_depth",
                        dest="hidden_layer_depth", default=1, type=int,
                        help="no of recurrent layers")

    parser.add_argument("-ll", "--latent_length",
                        dest="latent_length", default=20, type=int,
                        help="dimension of latent space vectors")

    parser.add_argument("-bs", "--batch_size",
                        dest="batch_size", default=32, type=int,
                        help="batch size for training")

    parser.add_argument("-lr", "--learning_rate",
                        dest="learning_rate", default=0.0005, type=float,
                        help="learning rate") # 0.00005

    parser.add_argument("-mo", "--momentum",
                        dest="momentum", default=0.9, type=float,
                        help="SGD momentum (0.5 default)")

    parser.add_argument("-ne", "--n_epochs",
                        dest="n_epochs", default=10, type=int,
                        help="number of epochs")

    parser.add_argument("-dr", "--dropout_rate",
                        dest="dropout_rate", default=0.2, type=float,
                        help="dropout rate")

    parser.add_argument("-opt", "--optimizer",
                        dest="optimizer", default="Adam", type=str,
                        help="optimizer for loss function (Adam, SGD)")

    parser.add_argument("-pe", "--print_every",
                        dest="print_every", default=10, type=int,
                        help="print loss value every n iterations")

    parser.add_argument("-clip", "--clip",
                        dest="clip", default=True, type=bool,
                        help="gradient clipping (avoid exploding grads)")

    parser.add_argument("-mgn", "--max_grad_norm",
                        dest="max_grad_norm", default=5, type=int)

    parser.add_argument("-loss", "--loss",
                        dest="loss", default="MSELoss", type=str,
                        help="choose between SmoothL1Loss and MSELoss")

    parser.add_argument("-block", "--block",
                        dest="block", default="LSTM", type=str,
                        help="choose between LSTM or GRU as RNNs")

    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help='random seed (default: 1)')

    parser.add_argument("--plot_engine", default="matplotlib", type=str,
                        help="choose plot cluster (matplotlib or plotly)")

    parser.add_argument("--num_processes", type=int, default=4, metavar="N",
                        help="how many n_train_processes to use (default: 4)")

    parser.add_argument("-cuda", "--cuda", default=False, type=bool,
                        help="bool param for GPU support (true if NVIDIA)")

    parser.add_argument("--multi_gpu", default=False, type=bool,
                        help="bool param for multi-GPU " \
                        "(true if multiple NVIDIA cards (cluster support))")

    parser.add_argument("-kl_annealing", "--kl_annealing", default="CyclicalKL",
                        type=str, help="CyclicalKL or MonotonicalKL")

    args = parser.parse_args()

    # user inputs name_dataset, target_sequence_length and window_size
    name_dataset = "IceDetection_ieckai"
    target_sequence_length = 47000
    min_sequence_length = 10000
    window_size = 1000 # 200, 500, 1000
    d_factor = 1 # factor to downsample dataset (1 if using full dataset)

    data_folder_path = os.path.join(os.path.abspath("../data/" + name_dataset))
    print("[INFO] Data folder path: ", data_folder_path)

    # instantiate hdf5 datasets
    hf5_datasets = get_HDF5(data_path = data_folder_path,
                            min_sequence_length = min_sequence_length,
                            target_sequence_length = target_sequence_length)

    # load weight_configurations
    datasets_weight_configs = [hf5_datasets[str(i)].get_weight_configs() for i in hf5_datasets]
    print("[INFO] Fetched all weight configurations from datasets dictionary")

    # define directory to store hdf5 data as npy files
    npy_folder = "npy_data"

    try:
        os.makedirs(os.path.join(data_folder_path, npy_folder))
        print("[INFO] created npy directory to store data")
    except FileExistsError:
        pass

    # define dictionary with Path to npy files
    npy_files_dict = {"normal" : Path(os.path.join(data_folder_path, npy_folder, "normal.npy")),
                      "zone1" : Path(os.path.join(data_folder_path, npy_folder, "abnormal_zone1.npy")),
                      "zone2" : Path(os.path.join(data_folder_path, npy_folder, "abnormal_zone2.npy")),
                      "zone3" : Path(os.path.join(data_folder_path, npy_folder, "abnormal_zone3.npy"))}

    # if npy data already exists, load it
    # NOTE: right now just checking that one npy file is in folder, ideally want to iterate
    # for key, value in npy_files_dict.items():
    if npy_files_dict["normal"].is_file():
        print("[INFO] npy files exist, loading them")
        np_normal_sensor_data = np.load(npy_files_dict["normal"])
        np_abnormal_zone1_sensor_data = np.load(npy_files_dict["zone1"])
        np_abnormal_zone2_sensor_data = np.load(npy_files_dict["zone2"])
        np_abnormal_zone3_sensor_data = np.load(npy_files_dict["zone3"])

    # else read and save npy files
    else:
        print("[INFO] reading and saving hdf5 data into npy files")
        # store data 4 numpy arrays: 1 normal and 3 abnormal configurations
        array_normal_sensor_data = []
        array_abnormal_zone1_sensor_data = []
        array_abnormal_zone2_sensor_data = []
        array_abnormal_zone3_sensor_data = []

        hdf5_weight_configurations = []

        # iterate over each hdf5 and append sensor data based on normal and abnormal configurations
        print("[INFO] calling get_sensor_readings to fetch normal and abnormal data from ALL ZONES")
        for i in range(len(hf5_datasets)):
            normal_data, abnormal_zone1_data, abnormal_zone2_data, abnormal_zone3_data = \
                hf5_datasets[str(i)].get_sensor_readings(downsample_factor=d_factor)

            # append sensor data of each simulation (it is None if sim has no target sequence length)
            if normal_data is not None:
                array_normal_sensor_data.append(normal_data)
                array_abnormal_zone1_sensor_data.append(abnormal_zone1_data)
                array_abnormal_zone2_sensor_data.append(abnormal_zone2_data)
                array_abnormal_zone3_sensor_data.append(abnormal_zone3_data)

            print("[INFO] finished reading {0}-th hdf5 file".format(i))
            print()

        np_normal_sensor_data = np.vstack(array_normal_sensor_data)
        np_abnormal_zone1_sensor_data = np.vstack(array_abnormal_zone1_sensor_data)
        np_abnormal_zone2_sensor_data = np.vstack(array_abnormal_zone2_sensor_data)
        np_abnormal_zone3_sensor_data = np.vstack(array_abnormal_zone3_sensor_data)

        # save data as npy arrays
        print("[INFO] saving normal and abnormal data into npy files")
        np.save(os.path.join(data_folder_path, npy_folder, "normal.npy"), np_normal_sensor_data)
        np.save(os.path.join(data_folder_path, npy_folder, "abnormal_zone1.npy"), np_abnormal_zone1_sensor_data)
        np.save(os.path.join(data_folder_path, npy_folder, "abnormal_zone2.npy"), np_abnormal_zone2_sensor_data)
        np.save(os.path.join(data_folder_path, npy_folder, "abnormal_zone3.npy"), np_abnormal_zone3_sensor_data)

    print()
    print("[INFO] finished fetching normal and abnormal simulations")
    print("[INFO] summary of data shape after loading/saving all hdf5")
    print("normal:", np_normal_sensor_data.shape)
    print("abnormal 1:", np_abnormal_zone1_sensor_data.shape)
    print("abnormal 2:", np_abnormal_zone2_sensor_data.shape)
    print("abnormal 3:", np_abnormal_zone3_sensor_data.shape)

    print()
    print("[INFO] the above shows the class balance statistics normal vs z1 vs z2 vs z3")

    print()
    print("[INFO] started scaling time series data")
    # NOTE: scaling data requires to provide data as 2D (num_samples, features), not 3D
    start_time = time.time()
    normal_data_scaled = scale_time_series(np_normal_sensor_data, scaling="Normal")
    abnormal_zone1_data_scaled = scale_time_series(np_abnormal_zone1_sensor_data, scaling="Normal")
    abnormal_zone2_data_scaled = scale_time_series(np_abnormal_zone2_sensor_data, scaling="Normal")
    abnormal_zone3_data_scaled = scale_time_series(np_abnormal_zone3_sensor_data, scaling="Normal")
    end_time = time.time()
    print("[INFO] finished scaling, toook %f seconds to scale data." % (end_time - start_time))

    print()
    print("[INFO] data headers")
    data_headers = get_data_headers(file_path=os.path.join(data_folder_path, "data_headers.txt"))
    print(data_headers)

    print()
    print("[INFO] filtering data based on selected features")
    features = [data_headers["Spn1ALxb1"], data_headers["Spn1ALyb1"],
                data_headers["Spn1ALxb2"], data_headers["Spn1ALyb2"],
                data_headers["Spn1ALxb3"], data_headers["Spn1ALyb3"]]

    print("[INFO] target features (integer encoded):", features)
    # print(normal_data_scaled[:, :, [8,9,10,11,12,13]].shape)
    normal_data_filtered = filter_timeseries_by_columns(normal_data_scaled, columns=features)
    abnormal_zone1_data_filtered = filter_timeseries_by_columns(abnormal_zone1_data_scaled, columns=features)
    abnormal_zone2_data_filtered = filter_timeseries_by_columns(abnormal_zone2_data_scaled, columns=features)
    abnormal_zone3_data_filtered = filter_timeseries_by_columns(abnormal_zone3_data_scaled, columns=features)

    print("[INFO] filtered normal data:", normal_data_filtered.shape)
    print("[INFO] abnormal zone 1 data:", abnormal_zone1_data_filtered.shape)
    print("[INFO] abnormal zone 2 data:", abnormal_zone2_data_filtered.shape)
    print("[INFO] abnormal zone 3 data:", abnormal_zone3_data_filtered.shape)

    # define directory to model (model.pth file)
    model_download = os.path.join("../checkpoints", name_dataset)

    try:
        os.makedirs(model_download)
        print()
        print("[INFO] created checkpoint directory for dataset: {0}".format(name_dataset))
    except FileExistsError:
        pass

    print()
    print("[INFO] calling split_train_val_timeseries: split and label normal and abnormal data into train and validation sets")
    X_train, X_val, X_val_labels = split_train_val_timeseries(normal_data_filtered,
                                                              abnormal_zone1_data_filtered,
                                                              abnormal_zone2_data_filtered,
                                                              abnormal_zone3_data_filtered,
                                                              min_sequence_length,
                                                              window_size,
                                                              ratio_train=0.70)

    print()
    print("[INFO] summary of training and validation arrays")
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_val labels:", X_val_labels)

    print()
    print("[INFO] saving labels as npy") # (save in checkpoints or images)
    np.save(model_download + "/labels.npy", X_val_labels)

    # convert numpy data to pyotrch tensors (then to pytorch dataset)
    train_dataset = TensorDataset(torch.from_numpy(X_train))
    test_dataset = TensorDataset(torch.from_numpy(X_val))

    # provide sequence length and features VRAE object
    sequence_length = X_train.shape[1] # sequence length eg. 8600, 47000
    number_of_features = X_train.shape[2] # input features 6

    print()
    print("[INFO] instantiated VRAE")
    vrae = VRAE(sequence_length,
                number_of_features,
                args.hidden_size_1,
                args.hidden_layer_depth,
                args.latent_length,
                args.batch_size,
                args.learning_rate,
                args.momentum,
                args.n_epochs,
                args.dropout_rate,
                args.optimizer,
                args.cuda,
                args.print_every,
                args.clip,
                args.max_grad_norm,
                args.loss,
                args.block,
                args.kl_annealing,
                model_download)

    # setup CUDA device (GPU support)
    use_cuda = args.cuda and torch.cuda.is_available()
    # cuda_GPU=0
    # device = torch.device("cuda:{}".format(cuda_GPU) if use_cuda else "cpu")
    device = torch.device("cuda" if use_cuda else "cpu")
    print("[INFO] using {0} device".format(device))

    # setup multi-gpu processing (gpu node in loewenburg)
    if torch.cuda.device_count() > 1:
        args.multi_gpu = True
        print("[INFO] cuda device cout > 1: entered multi-gpu parallelization")
        print("using:", torch.cuda.device_count(), "GPUs")
        vrae = nn.DataParallel(vrae)
        print("parallelized model given available GPUs...")

    # locate pytorch tensors into specified device
    vrae.to(device)

    if use_cuda:
        print()
        print("[INFO] CUDA summary")
        print("torch cuda:", use_cuda)
        print("torch cuda device location:", torch.cuda.device(0))
        print("torch cuda current device:", torch.cuda.current_device())
        print("torch cuda device name:", torch.cuda.get_device_name(0))
        print("torch cuda device count:", torch.cuda.device_count())

    # define PATH variable to store model params
    PATH = model_download + "/MODEL_multiple_vrae_classes_" + name_dataset + "_all_zones_" + "_features_" + str(len(features)) + \
           "_epochs_" + str(args.n_epochs) + "_batches_" + str(args.batch_size) + "_hidden_size_" + str(args.hidden_size_1) + "_latent_length_" + str(args.latent_length) + ".pth"

    # define TRAIN and TEST pkl to store losses
    TRAIN_pkl = model_download + "/TRAIN_LOSS_multiple_vrae_classes_" + name_dataset + "_all_zones_" + "_features_" + str(len(features)) + \
                "_epochs_" + str(args.n_epochs) + "_batches_" + str(args.batch_size) + "_hidden_size_" + str(args.hidden_size_1) + "_latent_length_" + str(args.latent_length) + ".pkl"

    TEST_pkl = model_download + "/TEST_LOSS_multiple_vrae_classes_" + name_dataset + "_all_zones_" + "_features_" + str(len(features)) + \
              "_epochs_" + str(args.n_epochs) + "_batches_" + str(args.batch_size) + "_hidden_size_" + str(args.hidden_size_1) + "_latent_length_" + str(args.latent_length) + ".pkl"

    if Path(PATH).is_file():
        print()
        print("[INFO] PATH file exists, loading pre-trained model")
        vrae.module.load(PATH=PATH) if args.multi_gpu else vrae.load(PATH=PATH)

    else:
        print()
        print("[INFO] entered train model (saving model params and losses)")
        print("running for {} epochs...".format(args.n_epochs))
        vrae.module.fit(train_dataset, test_dataset) if args.multi_gpu else vrae.fit(train_dataset, test_dataset)
        vrae.module.save(PATH=PATH) if args.multi_gpu else vrae.save(PATH=PATH)
        vrae.module.save_log(TRAIN_pkl=TRAIN_pkl, TEST_pkl=TEST_pkl) if args.multi_gpu else vrae.save_log(TRAIN_pkl=TRAIN_pkl, TEST_pkl=TEST_pkl)

    print()
    print("[INFO] applying compression to obtain latent vectors")
    print("torch train size:", torch.from_numpy(X_train).size())
    print("torch val size:", torch.from_numpy(X_val).size())

    # define PATH_z_run to store latent vectors (d-dimensional arrays)
    PATH_z_run = model_download + "/Z_RUN_multiple_vrae_classes_" + name_dataset + "_all_zones_" + "_features_" + str(len(features)) + \
                "_epochs_" + str(args.n_epochs) + "_batches_" + str(args.batch_size) + "_hidden_size_" + str(args.hidden_size_1) + "_latent_length_" + str(args.latent_length) + ".pkl"

    if Path(PATH_z_run).is_file():
        print()
        print("[INFO] entered load z_run")
        if args.multi_gpu:
            z_run = vrae.module.transform(test_dataset, file_name=PATH_z_run, load=True)
        else:
            z_run = vrae.transform(test_dataset, file_name=PATH_z_run, load=True)
    else:
        print()
        print("[INFO] entered apply and save z_run")
        if args.multi_gpu:
            z_run = vrae.module.transform(test_dataset, file_name=PATH_z_run, save=True)
        else:
            z_run = vrae.transform(test_dataset, file_name=PATH_z_run, save=True)

    # define path to save 2D projection images of time series
    image_download = os.path.join("../images/clustering", name_dataset)
    try:
        os.makedirs(model_download)
        print("created image directory for {0}".format(name_dataset))
    except FileExistsError:
        pass

    image_name = "/IMAGE_multiple_vrae_classes_" + name_dataset + "_all_zones_" + "_features_" + str(len(features)) + \
                 "_epochs_" + str(args.n_epochs) + "_batches_" + str(args.batch_size) + "_hidden_size_" + str(args.hidden_size_1) + "_latent_length_" + str(args.latent_length)

    print()
    print("[INFO] projecting latent vectors (z_run) onto 2D")
    if args.plot_engine == "matplotlib":
        plot_clustering(z_run, X_val_labels, image_download, image_name, engine="matplotlib", download=True)
    else:
        plot_clustering(z_run, X_val_labels, image_download, image_name, engine="plotly", download=True)

    print()
    print("[INFO] plotting latent vectors as 2D lines")
    plot_latent_vectors_as_lines(z_run, X_val_labels, image_download, image_name, download=True)

    print()
    print("[INFO] performing PCA on raw data")
    plot_raw_PCA(X_val, X_val_labels, image_download, image_name, download=True)

    print()
    print("[INFO] running unsupervised clustering on latent vectors (z_run) (under KERNEL PCA PROJECTION)")
    cluster_model = Cluster(z_run, image_download, image_name)

    print("k means ++")
    km, y_km = cluster_model.kmeans_clustering(clusters=4)
    cluster_model.plot_clustering_model(y_km, "_k_means", download=True)

    print("spectral clustering")
    sc, y_sc = cluster_model.spectral_clustering(clusters=4)
    cluster_model.plot_clustering_model(y_sc, "_spectral", download=True)

    print("hierarchical clustering")
    hc, y_hc = cluster_model.hierarchichal_clustering(clusters=4)
    cluster_model.plot_clustering_model(y_hc, "_hierarchical", download=True)

    print("OPTICS clustering")
    optics, y_optics = cluster_model.optics_clustering()
    cluster_model.plot_clustering_model(y_optics, "_optics", download=True)

    print("DBSCAN")
    dbscan, y_dbscan = cluster_model.dbscan_clustering()
    cluster_model.plot_clustering_model(y_dbscan, "_dbscan", download=True)

    print()
    print("[INFO] decoding latent vectors to reconstruct time series")

    # define variable to store reconstruction values
    PATH_x_decoded = model_download + "/X_DECODED_multiple_vrae_classes_" + name_dataset + "_all_zones_" + "_features_" + str(len(features)) + \
                     "_epochs_" + str(args.n_epochs) + "_batches_" + str(args.batch_size) + "_hidden_size_" + str(args.hidden_size_1) + "_latent_length_" + str(args.latent_length) + ".pkl"

    if Path(PATH_x_decoded).is_file():
        print()
        print("[INFO] loading reconstruction network (decoder)")
        if args.multi_gpu:
            x_decoded = vrae.module.reconstruct(test_dataset, file_name=PATH_x_decoded, load=True)
            np.save(model_download + "/x_original.npy", X_val)
        else:
            x_decoded = vrae.reconstruct(test_dataset, file_name=PATH_x_decoded, load=True)
            np.save(model_download + "/x_original.npy", X_val)
    else:
        print()
        print("[INFO] applying reconstruction on latent vectors using decoder")
        if args.multi_gpu:
            x_decoded = vrae.module.reconstruct(test_dataset, file_name=PATH_x_decoded, save=True)
            np.save(model_download + "/x_original.npy", X_val)
        else:
            x_decoded = vrae.reconstruct(test_dataset, file_name=PATH_x_decoded, save=True)
            np.save(model_download + "/x_original.npy", X_val)

    print()
    print("---------------")
    print("|FINISHED MAIN|")
    print("---------------")


if __name__ == "__main__":
    print("numpy version:", np.version.version)
    print("torch version:", torch.__version__)
    print(sys.version)
    main()
