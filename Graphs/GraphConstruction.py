"""
===========================================================
Project: PHOTON Graph
===========================================================
Description
-----------
A wrapper containing functions for turning connectivity matrices into graph structures

Version
-------
Created:        12-07-2019
Last updated:   03-08-2019


Author
------
Vincent Holstein
Translationale Psychiatrie
Universitaetsklinikum Muenster
"""

#TODO: add graph construction based on random walks
#TODO: debug/check kNN-based graph approaches, "deposit" atlas coordinate files
#TODO: add documentation for every method

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import sklearn
import scipy
import pylab
import os
import random


class GraphConstructorKNN(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    #constructs the adjacency matrix for the connectivity matrices by a kNN approach
    #adapted from Ktena et al., 2017

    def __init__(self, k_distance = 10, logs=''):
        self.k_distance = k_distance
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        pass

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
        # use the mean 2d image of all samples for creating the different graph structures
        X_mean = np.squeeze(np.mean(X, axis=0))

        # generate adjacency matrix
        d, idx = self.distance_sklearn_metrics(X_mean, k=10, metric='euclidean')
        adjacency = self.adjacency(d, idx).astype(np.float32)

        #turn adjacency into numpy matrix for concatenation
        adjacency = adjacency.toarray()

        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], -1))
        # X = X[..., None] + adjacency[None, None, :] #use broadcasting to speed up computation
        adjacency = np.repeat(adjacency[np.newaxis, :, :, np.newaxis], X.shape[0], axis=0)
        X = np.append(X, adjacency, axis=3)
        print(X.shape)

        # Todo: CAVE!!! check that the matrices have similar shape, so that you can actually concatenate them (and make sure that they are compatible with the pytorch_geometric)

        return X



class GraphConstructorSpatial(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, k_distance = 10,
                 atlas_name = 'ho', logs=''):
        self.k_distance = k_distance
        self.atlas_name = atlas_name
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()


    def fit(self, X, y):
        pass


    def distance_scipy_spatial(self, z, k, metric='euclidean'):
        """Compute exact pairwise distances."""
        d = scipy.spatial.distance.pdist(z, metric)
        d = scipy.spatial.distance.squareform(d)
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


    def get_atlas_coords(atlas_name='ho'):
        """
            atlas_name   : name of the atlas used
        returns:
            matrix       : matrix of roi 3D coordinates in MNI space (num_rois x 3)
        """
        root_folder = 1
        coords_file = os.path.join(root_folder, atlas_name + '_coords.csv')
        coords = np.loadtxt(coords_file, delimiter=',')

        if atlas_name == 'ho':
            coords = np.delete(coords, 82, axis=0)

        return coords


    def transform(self, X):
        # use the mean 2d image of all samples for creating the different graph structures
        X_mean = np.squeeze(np.mean(X, axis=0))

        #get atlas coords
        coords = self.get_atlas_coords(atlas_name=self.atlas_name)

        # generate adjacency matrix
        dist, idx = self.distance_scipy_spatial(coords, k=10, metric='euclidean')
        adjacency = self.adjacency(dist, idx).astype(np.float32)

        #turn adjacency into numpy matrix for concatenation
        adjacency = adjacency.toarray()

        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], -1))
        # X = X[..., None] + adjacency[None, None, :] #use broadcasting to speed up computation
        adjacency = np.repeat(adjacency[np.newaxis, :, :, np.newaxis], X.shape[0], axis=0)
        X = np.append(X, adjacency, axis=3)
        print(X.shape)





class GraphConstructorThreshold(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    # threshold a matrix to generate the adjacency matrix
    # you can use both a different and the own matrix

    def __init__(self, k_distance = 10, threshold = 0.1, adjacency_axis = 0,
                 concatenation_axis = 3,
                 one_hot_nodes = 0, logs=''):
        self.k_distance = k_distance
        self.threshold = threshold
        self.adjacency_axis = adjacency_axis
        self.concatenation_axis = concatenation_axis
        self.one_hot_nodes = one_hot_nodes
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        pass

    def transform(self, X):

        #This creates and indvidual adjacency matrix for each person
        Threshold_matrix = X[:,:,:,self.adjacency_axis]
        Threshold_matrix[Threshold_matrix > self.threshold] = 1
        Threshold_matrix[Threshold_matrix < self.threshold] = 0
        #add extra dimension to make sure that concatenation works later on
        Threshold_matrix = Threshold_matrix.reshape(Threshold_matrix.shape[0], Threshold_matrix.shape[1], Threshold_matrix.shape[2], -1)

        #Add the matrix back again
        if self.one_hot_nodes == 1:
            #construct an identity matrix
            identity_matrix = np.identity((X.shape[1]))
            #expand its dimension for later re-addition
            identity_matrix = np.reshape(identity_matrix, (-1, identity_matrix.shape[0], identity_matrix.shape[1]))
            identity_matrix = np.reshape(identity_matrix, (identity_matrix.shape[0], identity_matrix.shape[1], identity_matrix.shape[2], -1))
            one_hot_node_features = np.repeat(identity_matrix, X.shape[0], 0)
            #concatenate matrices
            X = np.concatenate((one_hot_node_features, Threshold_matrix), axis=self.concatenation_axis)
        else:
            X = np.concatenate((X, Threshold_matrix), axis=self.concatenation_axis)
            X = np.delete(X, self.adjacency_axis, self.concatenation_axis)


        return X


class GraphConstructorPercentage(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    # use the mean 2d FA or NOS DTI-Matrix of all samples for thresholding the graphs

    def __init__(self, percentage = 0.8, adjacency_axis = 0, logs=''):
        self.percentage = percentage
        self.adjacency_axis = adjacency_axis
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        pass

    def transform(self, X):

        #generate binary matrix
        BinaryMatrix = np.zeros((1, X.shape[1], X.shape[2], 1))

        for i in range(X.shape[0]):
            #select top percent connections
            # calculate threshold from given percentage cutoff
            lst = X[i, :, :, self.adjacency_axis].tolist()
            lst = [item for sublist in lst for item in sublist]
            lst.sort()
            #new_lst = lst[int(len(lst) * self.percentage): int(len(lst) * 1)]
            #threshold = new_lst[0]
            threshold = lst[int(len(lst) * self.percentage)]


            #Threshold matrix X to create adjacency matrix
            BinarizedMatrix = X[i, :, :, self.adjacency_axis]
            BinarizedMatrix[BinarizedMatrix > threshold] = 1
            BinarizedMatrix[BinarizedMatrix < threshold] = 0
            BinarizedMatrix = BinarizedMatrix.reshape((-1, BinaryMatrix.shape[0], BinaryMatrix.shape[1]))
            BinarizedMatrix = BinarizedMatrix.reshape((BinaryMatrix.shape[0], BinaryMatrix.shape[1], BinaryMatrix.shape[2], -1))

            #concatenate matrix back
            BinaryMatrix = np.concatenate((BinaryMatrix, BinarizedMatrix), axis = 3)

        #drop first matrix as it is empty
        BinaryMatrix = np.delete(BinaryMatrix, 0, 3)
        BinaryMatrix = np.swapaxes(BinaryMatrix, 3, 0)
        X = np.concatenate((X, BinaryMatrix), axis = 3)

        return X



class GraphConstructorRandomWalks(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    #uses random walks to generate the connectivity matrix for graph structures
    class GraphConstructorKNN(BaseEstimator, TransformerMixin):
        _estimator_type = "transformer"

        def __init__(self, k_distance=10, number_of_walks=10,
                     walk_length = 3, window_size = 5, logs=''):
            self.k_distance = k_distance
            self.number_of_walks = number_of_walks
            self.walk_length = walk_length
            self.window_size = window_size
            if logs:
                self.logs = logs
            else:
                self.logs = os.getcwd()

        def fit(self, X, y):
            pass

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

        def random_walk(self, vertices, walk_length, startpoint, window_size):
            """Performs a random walk on a given graph"""

            dims = 2
            step_n = walk_length
            step_set = [-1, 0, 1]
            #start on a given random vertex
            origin = vertices[np.random.randint(low=0, high=vertices.shape[0]), np.random.randint(low=0, high=vertices.shape[1])]

            #step onto other random vertex
            step_shape = (step_n, dims)
            steps = np.random.choice(a=step_set, size=step_shape)
            #append that vertex value to the series
            path = np.concatenate([origin, steps])
            #slide window and check if they co-occur
            start = path[:1]
            stop = path[-1:]


            return #frequency of co-occurence

        def transform(self, X):
            # use the mean 2d image of all samples for creating the different graph structures
            X_mean = np.squeeze(np.mean(X, axis=0))

            d, idx = self.distance_sklearn_metrics(X_mean, k=10, metric='euclidean')
            adjacency = self.adjacency(d, idx).astype(np.float32)

            #use random walks for
            frequency = np.zeros((adjacency.shape))
            for i in range(0, self.number_of_walks):
                shuffled_vertices = random.shuffle(d) #do stuff
                for i in range(len(shuffled_vertices)):
                    for j in range(len(shuffled_vertices[i])):
                        vertex = d[i, j]
                        self.random_walk(vertices = shuffled_vertices, walk_length=self.walk_length, startpoint=vertex)


            # turn adjacency into numpy matrix for concatenation
            adjacency = adjacency.toarray()

            X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], -1))
            # X = X[..., None] + adjacency[None, None, :] #use broadcasting to speed up computation
            adjacency = np.repeat(adjacency[np.newaxis, :, :, np.newaxis], X.shape[0], axis=0)
            X = np.append(X, adjacency, axis=3)
            print(X.shape)

            # Todo: CAVE!!! check that the matrices have similar shape, so that you can actually concatenate them (and make sure that they are compatible with the pytorch_geometric)

            return X
