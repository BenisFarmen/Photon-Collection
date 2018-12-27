from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
#from photonai.photonlogger.Logger import Logger


class OpenCV_Smooth(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, filtertype = 0, filter_size1 = 3,
                 filter_size2 = 3, logs=''):
        self.random_state = random_state
        # here the parameters of the filter for the smoothing operation are defined
        self.filtertype = filtertype
        self.filter_size1 = filter_size1
        self.filter_size2 = filter_size2
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        dim_0 = int(X.shape[0])
        dim_1 = int(X.shape[1])
        dim_2 = int(X.shape[2])

        X_reordered = np.empty([dim_0, dim_1, dim_2])

        if self.filtertype == 0:
            for i in range (X.shape[0]):
                X_reordered[i,:,:] = cv.Smooth(X[i], smoothtype = self.filter)

        if self.filtertype == 1:
            for i in range (X.shape[0]):
                X_reordered[i,:,:] = cv.Smooth(X[i], smoothtype = self.filter)

        if self.filtertype == 2:
            for i in range (X.shape[0]):
                X_reordered[i,:,:] = cv.Smooth(X[i], smoothtype = self.filter)


        return X_reordered



class OpenCV_Blur(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, kernel_dim1 = 5, kernel_dim2 = 5, logs=''):
        self.random_state = random_state
        self.kernel_dim1 = kernel_dim1
        self.kernel_dim2 = kernel_dim2

        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        dim_0 = int(X.shape[0])
        dim_1 = int(X.shape[1])
        dim_2 = int(X.shape[2])

        X_reordered = np.empty([dim_0, dim_1, dim_2])

        for i in range (X.shape[0]):
            X_reordered[i,:,:] = cv.blur(X[i], (self.kernel_dim1, self.kernel_dim2))

        return X_reordered


class OpenCV_GaussianBlur(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, filtertype = 'CV_GAUSSIAN', filter_size1 = 3,
                 filter_size2 = 3, logs=''):
        self.random_state = random_state
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        dim_0 = int(X.shape[0])
        dim_1 = int(X.shape[1])
        dim_2 = int(X.shape[2])

        X_reordered = np.empty([dim_0, dim_1, dim_2])

        for i in range (X.shape[0]):
            X_reordered[i,:,:] = cv.GaussianBlur(X[i], (5,5), 0) #make kernel dims a hyperparameter

        return X_reordered



class medianBlur(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, filtertype = 'CV_GAUSSIAN', filter_size1 = 3,
                 filter_size2 = 3, logs=''):
        self.random_state = random_state
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        dim_0 = int(X.shape[0])
        dim_1 = int(X.shape[1])
        dim_2 = int(X.shape[2])

        X_reordered = np.empty([dim_0, dim_1, dim_2])

        for i in range (X.shape[0]):
            X_reordered[i,:,:] = cv.medianBlur(X[i], 5) #make kernel dims a hyperparameter

        return X_reordered