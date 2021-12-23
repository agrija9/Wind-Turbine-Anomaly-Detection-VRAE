# -*- coding: utf-8 -*-
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE, SpectralEmbedding
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from plotly.graph_objs import *
import plotly
import sys
import re
from itertools import cycle
sys.path.append("../")

"""
Utility functions for processing timerseries data:
1) convert input data into timeseries format
2) plotting py timeseries into pytorch tensors to input into VRAE
"""

def data_to_time_series(normal_data, abnormal_data,
                       length_time_series, window_size,
                       ratio_train=0.8):
    """
    Converts data into time series format to input into neural network

    :param normal (abnormal) data: np 2D array of dimensions (samples*length_time_series, features)
    :param seq_length: sequence length
    :param ration_train: argument to split data into training and testing

    Output: np.arrays
    (x_train, x_val):
    (shuffled timeseries until ind_cut,
    rest of shuffled timeseries starting in ind_cut)
    """

    print("--------------------------")
    print(">>>Processing data into time series format<<<")
    print("Normal data original shape:", normal_data.shape)
    print("Abormal data original shape:", abnormal_data.shape)

    normal_samples = int(normal_data.shape[0]/length_time_series)
    abnormal_samples = int(abnormal_data.shape[0]/length_time_series)
    features = normal_data.shape[1]
    print("No. of normal time series (simulations):", normal_samples)
    print("No. of abnormal time series (simulations):", abnormal_samples)
    print("sequence length:", length_time_series)
    print("features:", features)

    print("--------------------------")
    print("Converting into time series format")

    # downsample data if there is an offset in no. of rows (both datasets)
    if len(normal_data) != normal_samples*length_time_series:
        normal_data = get_data_offset(normal_data, normal_samples, length_time_series)

    if len(abnormal_data) != abnormal_samples*length_time_series:
        abnormal_data = get_data_offset(abnormal_data, abnormal_samples, length_time_series)

    # reshape into (samples, seq_length, features) according to LSTM Input
    # example: (921, 8600, 6)
    normal_time_series_data = normal_data.reshape((normal_samples, length_time_series, features))
    abnormal_time_series_data = abnormal_data.reshape((abnormal_samples, length_time_series, features))
    print("normal time series before splitting chunks:", normal_time_series_data.shape)
    print("abnormal time series before splitting chunks:", abnormal_time_series_data.shape)

    # split data into chunks based on window size
    # separate chunks separately and then merge them
    if window_size:
        print("Splitting time series every {0} steps (w_size)".format(window_size))

        normal_chunks = int(normal_time_series_data.shape[1] / window_size)
        abnormal_chunks = int(abnormal_time_series_data.shape[1] / window_size)

        print("normal chunks generated per simulation:", normal_chunks)
        print("abnormal chunks generated per simulation:", abnormal_chunks)

        normal_time_series_data = normal_time_series_data.reshape(normal_samples*normal_chunks, window_size, features)
        abnormal_time_series_data = abnormal_time_series_data.reshape(abnormal_samples*abnormal_chunks, window_size, features)
        print("normal series after splitting into chunks:", normal_time_series_data.shape)
        print("abnormal series after splitting into chunks:", abnormal_time_series_data.shape)

        # create random indices to shuffle train and validation data
        normal_index_cut = int(ratio_train * normal_samples * normal_chunks)
        normal_indices = np.random.permutation(normal_samples * normal_chunks)

        abnormal_index_cut = int(ratio_train * abnormal_samples * abnormal_chunks)
        abnormal_indices = np.random.permutation(abnormal_samples * abnormal_chunks)

        # ind_cut = int(ratio_train * samples * no_chunks)
        # ind = np.random.permutation(samples * no_chunks)

    else:
        # NOTE: this is outdated since almost always working with window size
        # perform random selection of timeseries for training and evaluation
        ind_cut = int(ratio_train * normal_samples)
        ind = np.random.permutation(normal_samples)

    # create X_train and X_val arrays
    X_normal_train = normal_time_series_data[normal_indices[:normal_index_cut], 0:, :]
    X_abnormal_train = abnormal_time_series_data[abnormal_indices[:abnormal_index_cut], 0:, :]

    X_normal_val = normal_time_series_data[normal_indices[normal_index_cut:], 0:, :]
    X_abnormal_val = abnormal_time_series_data[abnormal_indices[abnormal_index_cut:], 0:, :]

    # create class labels
    normal_labels = [0 for i in range(len(X_normal_val))]
    abnormal_labels = [1 for i in range(len(X_abnormal_val))]

    X_train = np.vstack((X_normal_train, X_abnormal_train))
    X_val = np.vstack((X_normal_val, X_abnormal_val))
    X_val_labels = np.hstack((normal_labels, abnormal_labels))

    return X_train, X_val, X_val_labels

def get_data_offset(data, samples, length_time_series):
    offset = len(data) - samples*length_time_series
    print("offset of dataset:", offset)

    return data[:-offset,:]

def plot_clustering(z_run, labels, image_folder,
                    image_name, engine = 'plotly', download=False):
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

        hex_colors = []
        for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

        colors = [hex_colors[int(i)] for i in labels]

        z_run_pca = TruncatedSVD(n_components=3).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)
        z_run_se = SpectralEmbedding(n_components=2).fit_transform(z_run)

        # print(z_run_pca)
        np.save(image_folder + "/z_run_data.npy", z_run_pca)

        print("*******")
        print("z_run shape:", z_run.shape)
        print("z_run PCA shape", z_run_pca.shape)
        print("z_run tSNE shape", z_run_tsne.shape)
        print("z_run SE shape", z_run_se.shape)

        zero_index, = np.where(labels == 0) # integers
        one_index, = np.where(labels == 1) # integers
        # print("zero indices", zero_index)
        # print("one indices", one_index)

        print("z runs for zero and one indices")
        print("PCA zero index shape:", z_run_pca[zero_index].shape)
        print("PCA one index shape:", z_run_pca[one_index].shape)
        print("tSNE zero index shape:", z_run_tsne[zero_index].shape)
        print("tSNE one index shape:", z_run_tsne[one_index].shape)
        print("*******")

        # won't always get full x_val points (depends on batch size when do vrae.transform)
        scatter_limit = z_run_pca.shape[0]

        # PLOT PCA PROJECTIONS

        ###################################
        # plot 1,2 PCA components
        plt.rcParams.update({'font.size': 22})
        fig = plt.figure(figsize=(12,12))
        ax1 = fig.add_subplot(111)

        component_a = 1
        component_b = 2
        ax1.scatter(z_run_pca[zero_index][:,component_a-1], z_run_pca[zero_index][:,component_b-1],
                    c="red", marker='8', linewidths=0, label="normal")

        ax1.scatter(z_run_pca[one_index][:,component_a-1], z_run_pca[one_index][:,component_b-1],
                    c="blue", marker='D', linewidths=0, label="abnormal")

        # plt.rcParams.update({'font.size': 22})
        plt.title('PCA on z_run', fontdict = {'fontsize' : 30})
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

        ###################################
        # # plot 1,3 PCA components
        plt.rcParams.update({'font.size': 22})
        fig = plt.figure(figsize=(12,12))
        ax1 = fig.add_subplot(111)

        component_c = 1
        component_d = 3
        ax1.scatter(z_run_pca[zero_index][:,component_c-1], z_run_pca[zero_index][:,component_d-1],
                    c="red", marker='8', linewidths=0, label="normal")

        ax1.scatter(z_run_pca[one_index][:,component_c-1], z_run_pca[one_index][:,component_d-1],
                    c="blue", marker='D', linewidths=0, label="abnormal")

        plt.title('PCA on z_run', fontdict = {'fontsize' : 30})
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
        #
        # ###################################
        # # plot 2,3 PCA components
        # plt.rcParams.update({'font.size': 22})
        # fig = plt.figure(figsize=(12,12))
        # ax1 = fig.add_subplot(111)
        #
        # component_e = 2
        # component_f = 3
        # ax1.scatter(z_run_pca[zero_index][:,component_e-1], z_run_pca[zero_index][:,component_f-1],
        #             c="red", marker='8', linewidths=0, label="normal")
        #
        # ax1.scatter(z_run_pca[one_index][:,component_e-1], z_run_pca[one_index][:,component_f-1],
        #             c="blue", marker='D', linewidths=0, label="abnormal")
        #
        # plt.title('PCA on z_run', fontdict = {'fontsize' : 30})
        # plt.xlabel("Principal Component {}".format(component_e), fontsize=22)
        # plt.ylabel("Principal Component {}".format(component_f), fontsize=22)
        # plt.legend()
        # plt.grid()
        #
        # if download:
        #     print("Saving PCA image...")
        #     if os.path.exists(image_folder):
        #         pass
        #     else:
        #         os.mkdir(image_folder)
        #     plt.savefig(image_folder + image_name + "_comp_e_" + str(component_e) + "_comp_f_" + str(component_f) + "_pca.png")
        # else:
        #     plt.show()
        #
        # # PLOT tSNE PROJECTIONS
        plt.rcParams.update({'font.size': 22})
        fig = plt.figure(figsize=(12,12))
        ax1 = fig.add_subplot(111)

        ax1.scatter(z_run_tsne[zero_index][:,0], z_run_tsne[zero_index][:,1],
                    c="red", marker='8', linewidths=0, label="normal")

        ax1.scatter(z_run_tsne[one_index][:,0], z_run_tsne[one_index][:,1],
                    c="blue", marker='D', linewidths=0, label="abnormal")

        plt.title('t-SNE on z_run', fontdict = {'fontsize' : 30})
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

        # PLOT DIFFUSION MAPS PROJECTIONS
        plt.rcParams.update({'font.size': 22})
        fig = plt.figure(figsize=(12,12))
        ax1 = fig.add_subplot(111)

        ax1.scatter(z_run_se[zero_index][:,0], z_run_se[zero_index][:,1],
                    c="red", marker='8', linewidths=0, label="normal")

        ax1.scatter(z_run_se[one_index][:,0], z_run_se[one_index][:,1],
                    c="blue", marker='D', linewidths=0, label="abnormal")

        plt.title('Spectral Embedding on z_run', fontdict = {'fontsize' : 30})
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

    def plot_clustering_plotly(z_run):

       labels = labels[:z_run.shape[0]] # because of weird batch_size

       hex_colors = []
       for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

       colors = [hex_colors[int(i)] for i in labels]

       z_run_pca = TruncatedSVD(n_components=2).fit_transform(z_run)
       z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)

       trace = Scatter(x=z_run_pca[:, 0], y=z_run_pca[:, 1], mode='markers', marker=dict(color=colors))
       data = Data([trace])
       layout = Layout(title='PCA on z_run', showlegend=False)
       fig = Figure(data=data, layout=layout)
       plotly.offline.iplot(fig)

       trace = Scatter(x=z_run_tsne[:, 0], y=z_run_tsne[:, 1], mode='markers', marker=dict(color=colors))
       data = Data([trace])
       layout = Layout(title='tSNE on z_run', showlegend=False)
       fig = Figure(data=data, layout=layout)
       plotly.offline.iplot(fig)

    if (download == False) & (engine == 'plotly'):
        plot_clustering_plotly(z_run, labels)

    if (download) & (engine == 'plotly'):
        print("Can't download plotly plots")

    if engine == 'matplotlib':
        plot_clustering_matplotlib(z_run, labels, download, image_folder, image_name)

def plot_latent_vectors_as_lines(z_run, labels, image_folder, image_name, download=False):

    # labels = labels[:z_run.shape[0]] # because of weird batch_size

    zero_index, = np.where(labels == 0) # integers
    one_index, = np.where(labels == 1) # integers

    # PLOT 20-dim LATENT VECTORS AS LINES
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(111)

    # downsample every 45 lines
    print("testing:", z_run[zero_index][::30,].shape)
    print("testing:", z_run[one_index][::30,].shape)

    plt.plot(z_run[zero_index][::30,].T, c="red", marker='8') # normal
    plt.plot(z_run[one_index][::30,].T, c="blue", marker='D') # abnormal

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
    data: np.array([75, 200, 6])
    """
    reshaped_data = data.reshape((data.shape[0], data.shape[1]*data.shape[2]))

    # print("reshaped data", reshaped_data.shape)

    z_raw_pca = TruncatedSVD(n_components=3).fit_transform(reshaped_data)

    zero_index, = np.where(labels == 0) # integers
    one_index, = np.where(labels == 1) # integers

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
                c="blue", marker='D', linewidths=0, label="abnormal")

    # plt.rcParams.update({'font.size': 22})
    plt.title('PCA on z_raw', fontdict = {'fontsize' : 30})
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
