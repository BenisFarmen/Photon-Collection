from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import SpectralBiclustering
import sklearn
import scipy
import numpy as np
import os
from matplotlib import pyplot as plt
#from photonai.photonlogger.Logger import Logger

class GraphModifier(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, k_distance = 10, logs=''):
        self.k_distance = k_distance
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        # use the mean 2d image of all samples for creating the different graph structures
        X_mean = np.squeeze(np.mean(X, axis=0))

        d, idx = self.distance_sklearn_metrics(X_mean, k=10, metric='euclidean')
        adjacency = self.adjacency(d, idx).astype(np.float32)
        
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], -1)
        #X = X[..., None] + adjacency[None, None, :] #use broadcasting to speed up computation
        adjacency = np.repeat(adjacency[np.newaxis, :, :, np.newaxis], X.shape[0], axis=0)
        X = np.append(X, adjacency, axis=0)

        #Todo: concatenate matrices, so that you have an extra dimension for the adjacency matrix
        #Todo: CAVE!!! check that the matrices have similar shape, so that you can actually concatenate them (and make sure that they are compatible with the pytorch_geometric)
        #Todo: make sure we get the adjacency matrix in the transform statement as well

        return self

    def distance_sklearn_metrics(self, z, k, metric='euclidean'):
        """Compute exact pairwise distances."""
        d = sklearn.metrics.pairwise.pairwise_distances(
            z, metric=metric, n_jobs=-2)
        # k-NN graph.
        idx = np.argsort(d)[:, 1:k + 1]
        d.sort()
        d = d[:, 1:k + 1]
        return d, idx

    def adjacency(self, dist, idx):
        """Return the adjacency matrix of a kNN graph."""
        M, k = dist.shape
        assert M, k == idx.shape
        assert dist.min() >= 0

        # Weights.
        sigma2 = np.mean(dist[:, -1]) ** 2
        dist = np.exp(- dist ** 2 / sigma2)

        # Weight matrix.
        I = np.arange(0, M).repeat(k)
        J = idx.reshape(M * k)
        V = dist.reshape(M * k)
        W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

        # No self-connections.
        W.setdiag(0)

        # Non-directed graph.
        bigger = W.T > W
        W = W - W.multiply(bigger) + W.T.multiply(bigger)

        assert W.nnz % 2 == 0
        assert np.abs(W - W.T).mean() < 1e-10
        assert type(W) is scipy.sparse.csr.csr_matrix

        return W

    def transform(self, X):
        X_reordered = np.empty(X.shape)
        for i in range(X.shape[0]):
            x = np.squeeze(X[i,:,:])
            x_clust = x[np.argsort(self.biclustModel.row_labels_)]
            x_clust = x_clust[:, np.argsort(self.biclustModel.column_labels_)]
            X_reordered[i, :, :] = x_clust
        return X_reordered

