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

    # use the mean 2d FA or NOS DTI-Matrix of all samples for thresholding the graphs

    def __init__(self, k_distance = 10, threshold = 0.1, adjacency_axis = 0, logs=''):
        self.k_distance = k_distance
        self.threshold = threshold
        self.adjacency_axis = adjacency_axis
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

        #Add the matrix back again
        X = np.concatenate(X, Threshold_matrix, axis=4)

        return X
