from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import SpectralBiclustering
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
#from photonai.photonlogger.Logger import Logger

class OpenCV_Resize(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, resize_factor = 0.8, random_state=42, logs=''):
        self.random_state = random_state
        self.resize_factor = resize_factor
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y=None):
        return self

    #downsamples image according to resize_factor without blurring
    def transform(self, X):

        dim_0 = X.shape[0]
        dim_1 = int(X.shape[1] * self.resize_factor)
        dim_2 = int(X.shape[2] * self.resize_factor)
        newsize = (dim_1, dim_2)

        X_reordered = np.empty([dim_0, dim_1, dim_2])

        for i in range(X.shape[0]):
            X_reordered[i,:,:] = cv.resize(X[i], newsize)

        #print(X_reordered.shape)

        return X_reordered


class OpenCV_PyrDown(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, n_components=None, logs=''):
        self.n_components = n_components
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()
        self.pca = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_reordered = cv.pyrDown(X)
        return X_reordered
