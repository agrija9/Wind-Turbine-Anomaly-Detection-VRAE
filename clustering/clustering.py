import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import KernelPCA
from sklearn.cluster import (KMeans, AgglomerativeClustering,
                             SpectralClustering, DBSCAN, OPTICS)
from sklearn import svm

class Cluster():
    """
    Projects latent vectors in 2D and implements clustering methods on top of them.
    It uses kernel PCA for 2D projection (can be any other dimension-reduction method)
    """
    def __init__(self, z_run, image_folder, image_name):
        self.z_run_kernel_PCA = KernelPCA(n_components=3, kernel="rbf").fit_transform(z_run)
        # self.z_run_pca = TruncatedSVD(n_components=3).fit_transform(z_run)
        self.image_folder = image_folder
        self.image_name = image_name

    def kmeans_clustering(self, clusters):
        # set kmeans to find n clusters
        km = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
        y_km = km.fit_predict(self.z_run_kernel_PCA)
        return km, y_km # returns models and predictions

    def spectral_clustering(self, clusters):
        spectral_model_rbf = SpectralClustering(n_clusters = clusters, affinity ='rbf')
        y_spectral_model = spectral_model_rbf.fit_predict(self.z_run_kernel_PCA)
        return spectral_model_rbf, y_spectral_model

    def hierarchichal_clustering(self, clusters):
        aglomerative_clustering = AgglomerativeClustering(n_clusters=clusters, affinity="l2", linkage="complete")
        y_aglomerative_clustering = aglomerative_clustering.fit_predict(self.z_run_kernel_PCA)
        return aglomerative_clustering, y_aglomerative_clustering

    def dbscan_clustering(self):
        dbscan = DBSCAN(algorithm='auto', eps=0.4, leaf_size=30)
        y_dbscan = dbscan.fit_predict(self.z_run_kernel_PCA)

        return dbscan, y_dbscan

    def optics_clustering(self):
        optics = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)
        y_optics = optics.fit_predict(self.z_run_kernel_PCA)

        return optics, y_optics

    def plot_clustering_model(self, model_predictions, model_name, download=False):
        plt.rcParams.update({'font.size': 22})
        fig = plt.figure(figsize=(12,12))

        plt.title("{} on 2D latent vectors".format(model_name), fontdict = {'fontsize' : 30})

        print()
        print("Model predictions:")
        print(model_predictions)

        plt.scatter(self.z_run_kernel_PCA[model_predictions == 0, 0],
                    self.z_run_kernel_PCA[model_predictions == 0, 1],
                    s=50, c='lightgreen',
                    marker='s', edgecolor='black',
                    label='cluster 1')

        plt.scatter(self.z_run_kernel_PCA[model_predictions == 1, 0],
                    self.z_run_kernel_PCA[model_predictions == 1, 1],
                    s=50, c='orange',
                    marker='o', edgecolor='black',
                    label='cluster 2')

        plt.scatter(self.z_run_kernel_PCA[model_predictions == 2, 0],
                    self.z_run_kernel_PCA[model_predictions == 2, 1],
                    s=50, c='red',
                    marker='x', edgecolor='black',
                    label='cluster 3')

        plt.scatter(self.z_run_kernel_PCA[model_predictions == 3, 0],
                    self.z_run_kernel_PCA[model_predictions == 3, 1],
                    s=50, c='blue',
                    marker='*', edgecolor='black',
                    label='cluster 4')

        plt.legend(scatterpoints=1)
        plt.grid()

        if download:
            print("Saving clusters image...")
            if os.path.exists(self.image_folder):
                pass
            else:
                os.mkdir(self.image_folder)
            plt.savefig(self.image_folder + self.image_name + "{}.png".format(model_name))
        else:
            plt.show()
