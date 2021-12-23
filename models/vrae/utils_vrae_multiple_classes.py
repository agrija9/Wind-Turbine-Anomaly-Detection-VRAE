# -*- coding: utf-8 -*-
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import re
from itertools import cycle
sys.path.append("../")

"""
Utility functions for processing timerseries data:
1) convert input data into timeseries format
2) plotting py timeseries into pytorch tensors to input into VRAE
"""

def split_train_val_timeseries(normal_data,
                               abnormal_zone1_data,
                               abnormal_zone2_data,
                               abnormal_zone3_data,
                               length_time_series,
                               window_size,
                               ratio_train=0.8):
    """
    Splits 3D time series data into train and validation sets

    It also assigns labels to data (for cluster visualization):

    0 - normal cases zone 0
    1 - abnormal case zone 1
    2 - abnormal case zone 2
    3 - abnormal case zone 3

    :param normal and abnormal data: 3D np array of dimensions (samples, length_time_series, features)
    :param seq_length: sequence length
    :param ration_train: argument to split data into training and testing

    Output: np.arrays
    (x_train, x_val):
    (shuffled timeseries until ind_cut,
    rest of shuffled timeseries starting in ind_cut)
    """

    features = normal_data.shape[2]

    print("[INFO] total normal time series:", len(normal_data))
    print("[INFO] total abnormal zone 1 time series:", len(abnormal_zone1_data))
    print("[INFO] total abnormal zone 2 time series:", len(abnormal_zone2_data))
    print("[INFO] total abnormal zone 3 time series:", len(abnormal_zone3_data))
    print("[INFO] sequence length:", length_time_series)
    print("[INFO] features:", features)

    print()
    print("[INFO] splitting data into chunks every {0} time step based on window size parameter".format(window_size))
    # if window_size:
    normal_chunks = int(normal_data.shape[1] / window_size)
    abnormal_chunks_zone1 = int(abnormal_zone1_data.shape[1] / window_size)
    abnormal_chunks_zone2 = int(abnormal_zone2_data.shape[1] / window_size)
    abnormal_chunks_zone3 = int(abnormal_zone3_data.shape[1] / window_size)

    print("normal chunks (per sample):", normal_chunks)
    print("abnormal chunks zone 1 (per sample):", abnormal_chunks_zone1)
    print("abnormal chunks zone 2 (per sample):", abnormal_chunks_zone2)
    print("abnormal chunks zone 3 (per sample):", abnormal_chunks_zone3)

    # reshape chunks into time series format
    normal_data = normal_data.reshape(len(normal_data) * normal_chunks, window_size, features)
    abnormal_zone1_data = abnormal_zone1_data.reshape(len(abnormal_zone1_data) * abnormal_chunks_zone1, window_size, features)
    abnormal_zone2_data = abnormal_zone2_data.reshape(len(abnormal_zone2_data) * abnormal_chunks_zone2, window_size, features)
    abnormal_zone3_data = abnormal_zone3_data.reshape(len(abnormal_zone3_data) * abnormal_chunks_zone3, window_size, features)

    print()
    print("normal (after chunk split):", normal_data.shape)
    print("abnormal zone1 data (after chunk split):", abnormal_zone1_data.shape)
    print("abnormal zone2 data (after chunk split):", abnormal_zone2_data.shape)
    print("abnormal zone3 data (after chunk split)::", abnormal_zone3_data.shape)

    # create random indices to shuffle TRAIN and VALIDATION data
    normal_index_cut = int(ratio_train * len(normal_data))
    normal_indices = np.random.permutation(len(normal_data))

    abnormal_zone1_index_cut = int(ratio_train * len(abnormal_zone1_data))
    abnormal_zone1_indices = np.random.permutation(len(abnormal_zone1_data))

    abnormal_zone2_index_cut = int(ratio_train * len(abnormal_zone2_data))
    abnormal_zone2_indices = np.random.permutation(len(abnormal_zone2_data))

    abnormal_zone3_index_cut = int(ratio_train * len(abnormal_zone3_data))
    abnormal_zone3_indices = np.random.permutation(len(abnormal_zone3_data))

    # else:
    #     # NOTE: almost always working with window size
    #     # perform random selection of timeseries for training and evaluation
    #     ind_cut = int(ratio_train * normal_samples)
    #     ind = np.random.permutation(normal_samples)

    # create X_train and X_val arrays
    X_normal_train = normal_data[normal_indices[:normal_index_cut], 0:, :]
    X_abnormal_zone1_train = abnormal_zone1_data[abnormal_zone1_indices[:abnormal_zone1_index_cut], 0:, :]
    X_abnormal_zone2_train = abnormal_zone2_data[abnormal_zone2_indices[:abnormal_zone2_index_cut], 0:, :]
    X_abnormal_zone3_train = abnormal_zone3_data[abnormal_zone3_indices[:abnormal_zone3_index_cut], 0:, :]

    X_normal_val = normal_data[normal_indices[normal_index_cut:], 0:, :]
    X_abnormal_zone1_val = abnormal_zone1_data[abnormal_zone1_indices[abnormal_zone1_index_cut:], 0:, :]
    X_abnormal_zone2_val = abnormal_zone2_data[abnormal_zone2_indices[abnormal_zone2_index_cut:], 0:, :]
    X_abnormal_zone3_val = abnormal_zone3_data[abnormal_zone3_indices[abnormal_zone3_index_cut:], 0:, :]

    # create class labels with train/val arrays
    normal_labels, abnormal_zone1_labels, abnormal_zone2_labels, abnormal_zone3_labels = \
            create_timeseries_labels(X_normal_val, X_abnormal_zone1_val, X_abnormal_zone2_val, X_abnormal_zone3_val)

    # once labels are generated per zone, stack all train/val arrays vertically (no more need to differentiate them)
    X_train = np.vstack((X_normal_train, X_abnormal_zone1_train, X_abnormal_zone2_train, X_abnormal_zone3_train))
    X_val = np.vstack((X_normal_val, X_abnormal_zone1_val, X_abnormal_zone2_val, X_abnormal_zone3_val))
    X_val_labels = np.hstack((normal_labels, abnormal_zone1_labels, abnormal_zone2_labels, abnormal_zone3_labels))

    return X_train, X_val, X_val_labels

def create_timeseries_labels(normal_validation, abnormal_zone1_validation, abnormal_zone2_validation, abnormal_zone3_validation):
    """
    Assigns integer values to each zone for later label 2D embedding projections
    """
    normal_labels = [0 for i in range(len(normal_validation))]
    abnormal_zone1_labels = [1 for i in range(len(abnormal_zone1_validation))]
    abnormal_zone2_labels = [2 for i in range(len(abnormal_zone2_validation))]
    abnormal_zone3_labels = [3 for i in range(len(abnormal_zone3_validation))]

    return normal_labels, abnormal_zone1_labels, abnormal_zone2_labels, abnormal_zone3_labels

def get_data_offset(data, samples, length_time_series):
    offset = len(data) - samples*length_time_series
    print("offset of dataset:", offset)

    return data[:-offset,:]

def plot_clustering(z_run, labels, image_folder, image_name, engine = 'matplotlib', download=False):
    """
    Given latent variables for all timeseries, and output of k-means, run PCA
    and tSNE on latent vectors and color the points using cluster_labels.
    :param z_run: Latent vectors for all input tensors
    :param labels: Cluster labels for all input tensors
    :param engine: plotly/matplotlib
    :param download: If true, it will download plots in `image_folder`
    :param image_folder: Download folder to dump plots
    :return:
    """

    def plot_clustering_matplotlib(z_run, labels, download, image_folder, image_name):

        labels = labels[:z_run.shape[0]] # because of weird batch_size

        # hex_colors = []
        # for _ in np.unique(labels):
        #     hex_colors.append('#%06X' % randint(0, 0xFFFFFF))
        #
        # colors = [hex_colors[int(i)] for i in labels]

        z_run_pca = TruncatedSVD(n_components=3).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=50, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)
        z_run_se = SpectralEmbedding(n_components=2).fit_transform(z_run)
        z_run_kernel_PCA = KernelPCA(n_components=3, kernel="rbf", gamma=0.5).fit_transform(z_run)
        z_run_iso = Isomap(n_components=2, n_neighbors=10).fit_transform(z_run)

        print("[INFO] saving 2D projections from different methods")
        np.save(image_folder + "/z_run_PCA.npy", z_run_pca)
        np.save(image_folder + "/z_run_tSNE.npy", z_run_tsne)
        np.save(image_folder + "/z_run_SE.npy", z_run_se)
        np.save(image_folder + "/z_run_kernel_PCA.npy", z_run_kernel_PCA)
        np.save(image_folder + "/z_run_iso.npy", z_run_iso)

        print("z_run shape:", z_run.shape)
        print("z_run PCA shape", z_run_pca.shape)

        # fetch indices of labels
        zero_index, = np.where(labels == 0) # integers
        one_index, = np.where(labels == 1) # integers
        two_index, = np.where(labels == 2) # integers
        three_index, = np.where(labels == 3) # integers

        print()
        # print("z runs for zero and one indices")
        # print("PCA zero index shape:", z_run_pca[zero_index].shape)
        # print("PCA one index shape:", z_run_pca[one_index].shape)
        # print("PCA two index shape:", z_run_pca[two_index].shape)
        # print("PCA three index shape:", z_run_pca[three_index].shape)

        # won't always get full x_val points (depends on batch size when do vrae.transform)
        scatter_limit = z_run_pca.shape[0]

        # PLOT PCA PROJECTIONS
        # plot 1,2 PCA components
        plt.rcParams.update({'font.size': 22})
        fig = plt.figure(figsize=(12,12))
        ax1 = fig.add_subplot(111)

        component_a = 1
        component_b = 2
        ax1.scatter(z_run_pca[zero_index][:,component_a-1], z_run_pca[zero_index][:,component_b-1],
                    c="red", marker='8', linewidths=0, label="normal")

        ax1.scatter(z_run_pca[one_index][:,component_a-1], z_run_pca[one_index][:,component_b-1],
                    c="blue", marker='D', linewidths=0, label="abnormal zone 1")

        ax1.scatter(z_run_pca[two_index][:,component_a-1], z_run_pca[two_index][:,component_b-1],
                    c="green", marker='o', linewidths=0, label="abnormal zone 2")

        ax1.scatter(z_run_pca[three_index][:,component_a-1], z_run_pca[three_index][:,component_b-1],
                    c="black", marker='x', linewidths=0, label="abnormal zone 3")

        # plt.rcParams.update({'font.size': 22})
        plt.title("Latent vectors projected in 2D using PCA", fontdict = {'fontsize' : 30})
        plt.xlabel("Principal Component {}".format(component_a), fontsize=22)
        plt.ylabel("Principal Component {}".format(component_b), fontsize=22)
        plt.legend()
        plt.grid()

        if download:
            print("Saving PCA image...")
            if os.path.exists(image_folder):
                pass
            else:
                os.mkdir(image_folder)
            plt.savefig(image_folder + image_name + "_comp_a_" + str(component_a) + "_comp_b_" + str(component_b) + "_pca.png")
        else:
            plt.show()

        # plot 1,3 PCA components
        plt.rcParams.update({'font.size': 22})
        fig = plt.figure(figsize=(12,12))
        ax1 = fig.add_subplot(111)

        component_c = 1
        component_d = 3
        ax1.scatter(z_run_pca[zero_index][:,component_c-1], z_run_pca[zero_index][:,component_d-1],
                    c="red", marker='8', linewidths=0, label="normal")

        ax1.scatter(z_run_pca[one_index][:,component_c-1], z_run_pca[one_index][:,component_d-1],
                    c="blue", marker='D', linewidths=0, label="abnormal")

        ax1.scatter(z_run_pca[two_index][:,component_c-1], z_run_pca[two_index][:,component_d-1],
                    c="green", marker='o', linewidths=0, label="abnormal zone 2")

        ax1.scatter(z_run_pca[three_index][:,component_c-1], z_run_pca[three_index][:,component_d-1],
                    c="black", marker='x', linewidths=0, label="abnormal zone 3")

        plt.title("Latent vectors projected in 2D using PCA", fontdict = {'fontsize' : 30})
        plt.xlabel("Principal Component {}".format(component_c), fontsize=22)
        plt.ylabel("Principal Component {}".format(component_d), fontsize=22)
        plt.legend()
        plt.grid()

        if download:
            print("Saving PCA image...")
            if os.path.exists(image_folder):
                pass
            else:
                os.mkdir(image_folder)
            plt.savefig(image_folder + image_name + "_comp_c_" + str(component_c) + "_comp_d_" + str(component_d) + "_pca.png")
        else:
            plt.show()

        # plot 2,3 PCA components
        plt.rcParams.update({'font.size': 22})
        fig = plt.figure(figsize=(12,12))
        ax1 = fig.add_subplot(111)

        component_e = 2
        component_f = 3

        ax1.scatter(z_run_pca[zero_index][:,component_e-1], z_run_pca[zero_index][:,component_f-1],
                    c="red", marker='8', linewidths=0, label="normal")

        ax1.scatter(z_run_pca[one_index][:,component_e-1], z_run_pca[one_index][:,component_f-1],
                    c="blue", marker='D', linewidths=0, label="abnormal")

        ax1.scatter(z_run_pca[two_index][:,component_e-1], z_run_pca[two_index][:,component_f-1],
                    c="green", marker='o', linewidths=0, label="abnormal zone 2")

        ax1.scatter(z_run_pca[three_index][:,component_e-1], z_run_pca[three_index][:,component_f-1],
                    c="black", marker='x', linewidths=0, label="abnormal zone 3")

        plt.title("Latent vectors projected in 2D using PCA", fontdict = {'fontsize' : 30})
        plt.xlabel("Principal Component {}".format(component_e), fontsize=22)
        plt.ylabel("Principal Component {}".format(component_f), fontsize=22)
        plt.legend()
        plt.grid()

        if download:
            print("Saving PCA image...")
            if os.path.exists(image_folder):
                pass
            else:
                os.mkdir(image_folder)
            plt.savefig(image_folder + image_name + "_comp_e_" + str(component_e) + "_comp_f_" + str(component_f) + "_pca.png")
        else:
            plt.show()

        # PLOT tSNE PROJECTIONS
        plt.rcParams.update({'font.size': 22})
        fig = plt.figure(figsize=(12,12))
        ax1 = fig.add_subplot(111)

        ax1.scatter(z_run_tsne[zero_index][:,0], z_run_tsne[zero_index][:,1],
                    c="red", marker='8', linewidths=0, label="normal")

        ax1.scatter(z_run_tsne[one_index][:,0], z_run_tsne[one_index][:,1],
                    c="blue", marker='D', linewidths=0, label="abnormal zone 1")

        ax1.scatter(z_run_tsne[two_index][:,0], z_run_tsne[two_index][:,1],
                    c="green", marker='D', linewidths=0, label="abnormal zone 2")

        ax1.scatter(z_run_tsne[three_index][:,0], z_run_tsne[three_index][:,1],
                    c="black", marker='D', linewidths=0, label="abnormal zone 3")

        plt.title("Latent vectors projected in 2D using t-SNE", fontdict = {'fontsize' : 30})
        plt.legend()
        plt.grid()

        if download:
            print("Saving tSNE image...")
            if os.path.exists(image_folder):
                pass
            else:
                os.mkdir(image_folder)
            plt.savefig(image_folder + image_name + "_tsne.png")
        else:
            plt.show()

        # PLOT DIFFUSION MAP 2D PROJECTIONS
        plt.rcParams.update({'font.size': 22})
        fig = plt.figure(figsize=(12,12))
        ax1 = fig.add_subplot(111)

        ax1.scatter(z_run_se[zero_index][:,0], z_run_se[zero_index][:,1],
                    c="red", marker='8', linewidths=0, label="normal")

        ax1.scatter(z_run_se[one_index][:,0], z_run_se[one_index][:,1],
                    c="blue", marker='D', linewidths=0, label="abnormal zone 1")

        ax1.scatter(z_run_se[two_index][:,0], z_run_se[two_index][:,1],
                    c="green", marker='D', linewidths=0, label="abnormal zone 2")

        ax1.scatter(z_run_se[three_index][:,0], z_run_se[three_index][:,1],
                    c="black", marker='D', linewidths=0, label="abnormal zone 3")

        plt.title("Latent vectors projected in 2D using Spectral Embedding", fontdict = {'fontsize' : 30})
        plt.legend()
        plt.grid()

        if download:
            print("Saving SE image...")
            if os.path.exists(image_folder):
                pass
            else:
                os.mkdir(image_folder)
            plt.savefig(image_folder + image_name + "_se.png")
        else:
            plt.show()

        # PLOT KERNEL PCA 2D PROJECTIONS
        plt.rcParams.update({'font.size': 22})
        fig = plt.figure(figsize=(12,12))
        ax1 = fig.add_subplot(111)

        ax1.scatter(z_run_kernel_PCA[zero_index][:,0], z_run_kernel_PCA[zero_index][:,1],
                    c="red", marker='8', linewidths=0, label="normal")

        ax1.scatter(z_run_kernel_PCA[one_index][:,0], z_run_kernel_PCA[one_index][:,1],
                    c="blue", marker='D', linewidths=0, label="abnormal zone 1")

        ax1.scatter(z_run_kernel_PCA[two_index][:,0], z_run_kernel_PCA[two_index][:,1],
                    c="green", marker='D', linewidths=0, label="abnormal zone 2")

        ax1.scatter(z_run_kernel_PCA[three_index][:,0], z_run_kernel_PCA[three_index][:,1],
                    c="black", marker='D', linewidths=0, label="abnormal zone 3")

        plt.title("Latent vectors projected in 2D using Kernel PCA (rbf)", fontdict = {'fontsize' : 30})
        plt.legend()
        plt.grid()

        if download:
            print("Saving Kernel PCA image...")
            if os.path.exists(image_folder):
                pass
            else:
                os.mkdir(image_folder)
            plt.savefig(image_folder + image_name + "_kernel_pca.png")
        else:
            plt.show()

        # PLOT ISOMAP 2D PROJECTIONS
        plt.rcParams.update({'font.size': 22})
        fig = plt.figure(figsize=(12,12))
        ax1 = fig.add_subplot(111)

        ax1.scatter(z_run_iso[zero_index][:,0], z_run_iso[zero_index][:,1],
                    c="red", marker='8', linewidths=0, label="normal")

        ax1.scatter(z_run_iso[one_index][:,0], z_run_iso[one_index][:,1],
                    c="blue", marker='D', linewidths=0, label="abnormal zone 1")

        ax1.scatter(z_run_iso[two_index][:,0], z_run_iso[two_index][:,1],
                    c="green", marker='D', linewidths=0, label="abnormal zone 2")

        ax1.scatter(z_run_iso[three_index][:,0], z_run_iso[three_index][:,1],
                    c="black", marker='D', linewidths=0, label="abnormal zone 3")

        plt.title("Latent vectors projected in 2D using isomap", fontdict = {'fontsize' : 30})
        plt.legend()
        plt.grid()

        if download:
            print("Saving isomap image...")
            if os.path.exists(image_folder):
                pass
            else:
                os.mkdir(image_folder)
            plt.savefig(image_folder + image_name + "_iso.png")
        else:
            plt.show()

    if engine == 'matplotlib':
        plot_clustering_matplotlib(z_run, labels, download, image_folder, image_name)

def plot_latent_vectors_as_lines(z_run, labels, image_folder, image_name, download=False):

    # labels = labels[:z_run.shape[0]] # because of weird batch_size

    zero_index, = np.where(labels == 0) # integers
    one_index, = np.where(labels == 1) # integers
    two_index, = np.where(labels == 2) # integers
    three_index, = np.where(labels == 3) # integers

    # PLOT 20-dim LATENT VECTORS AS LINES
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(111)

    # downsample every 45 lines
    print("testing:", z_run[zero_index][::50,].shape)
    print("testing:", z_run[one_index][::50,].shape)

    plt.plot(z_run[zero_index][::15,].T, c="red", marker='8') # normal
    plt.plot(z_run[one_index][::15,].T, c="blue", marker='D') # abnormal
    plt.plot(z_run[two_index][::15,].T, c="green", marker='o') # normal
    plt.plot(z_run[three_index][::15,].T, c="black", marker='x') # abnormal

    plt.title("Latent vectors (z_run) as lines", fontdict = {'fontsize' : 30})
    # plt.xlabel("Principal Component {}".format(component_a), fontsize=22)
    # plt.ylabel("Principal Component {}".format(component_b), fontsize=22)
    plt.legend()
    plt.grid()

    if download:
        print("Saving plotted image...")
        if os.path.exists(image_folder):
            pass
        else:
            os.mkdir(image_folder)
        plt.savefig(image_folder + image_name + "_latent_vectors.png")
    else:
        plt.show()

def plot_raw_PCA(data, labels, image_folder, image_name, download=False):
    """
    Implements PCA and kernel PCA on original data

    data: np.array([75, 200, 6])
    """
    reshaped_data = data.reshape((data.shape[0], data.shape[1]*data.shape[2]))

    print("original data", data.shape)
    print("reshaped data", reshaped_data.shape)

    z_raw_pca = TruncatedSVD(n_components=3).fit_transform(reshaped_data)
    z_raw_kernel_PCA = KernelPCA(n_components=3, kernel="rbf", gamma=0.5).fit_transform(reshaped_data)

    zero_index, = np.where(labels == 0) # integers
    one_index, = np.where(labels == 1) # integers
    two_index, = np.where(labels == 2) # integers
    three_index, = np.where(labels == 3) # integers

    print("z_raw_pca", z_raw_pca.shape)

    scatter_limit = z_raw_pca.shape[0]

    # PLOT PCA PROJECTIONS
    # plot 1,2 PCA components
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(111)

    component_a = 1
    component_b = 2
    ax1.scatter(z_raw_pca[zero_index][:,component_a-1], z_raw_pca[zero_index][:,component_b-1],
                c="red", marker='8', linewidths=0, label="normal")

    ax1.scatter(z_raw_pca[one_index][:,component_a-1], z_raw_pca[one_index][:,component_b-1],
                c="blue", marker='D', linewidths=0, label="abnormal zone 1")

    ax1.scatter(z_raw_pca[two_index][:,component_a-1], z_raw_pca[two_index][:,component_b-1],
                c="green", marker='D', linewidths=0, label="abnormal zone 2")

    ax1.scatter(z_raw_pca[three_index][:,component_a-1], z_raw_pca[three_index][:,component_b-1],
                c="black", marker='D', linewidths=0, label="abnormal zone 3")

    # plt.rcParams.update({'font.size': 22})
    plt.title("PCA 2D projection on original data", fontdict = {'fontsize' : 30})
    plt.xlabel("Principal Component {}".format(component_a), fontsize=22)
    plt.ylabel("Principal Component {}".format(component_b), fontsize=22)
    plt.legend()
    plt.grid()

    if download:
        print("Saving plotted image...")
        if os.path.exists(image_folder):
            pass
        else:
            os.mkdir(image_folder)
        plt.savefig(image_folder + image_name + "_z_raw_PCA.png")
    else:
        plt.show()

    # PLOT KERNEL PCA PROJECTIONS
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(111)

    component_a = 1
    component_b = 2
    ax1.scatter(z_raw_kernel_PCA[zero_index][:,component_a-1], z_raw_kernel_PCA[zero_index][:,component_b-1],
                c="red", marker='8', linewidths=0, label="normal")

    ax1.scatter(z_raw_kernel_PCA[one_index][:,component_a-1], z_raw_kernel_PCA[one_index][:,component_b-1],
                c="blue", marker='D', linewidths=0, label="abnormal zone 1")

    ax1.scatter(z_raw_kernel_PCA[two_index][:,component_a-1], z_raw_pca[two_index][:,component_b-1],
                c="green", marker='D', linewidths=0, label="abnormal zone 2")

    ax1.scatter(z_raw_kernel_PCA[three_index][:,component_a-1], z_raw_kernel_PCA[three_index][:,component_b-1],
                c="black", marker='D', linewidths=0, label="abnormal zone 3")

    # plt.rcParams.update({'font.size': 22})
    plt.title("Kenrel PCA (rbf) 2D projection on original data", fontdict = {'fontsize' : 30})
    plt.xlabel("Principal Component {}".format(component_a), fontsize=22)
    plt.ylabel("Principal Component {}".format(component_b), fontsize=22)
    plt.legend()
    plt.grid()

    if download:
        print("Saving plotted image...")
        if os.path.exists(image_folder):
            pass
        else:
            os.mkdir(image_folder)
        plt.savefig(image_folder + image_name + "_z_raw_kernel_PCA.png")
    else:
        plt.show()
