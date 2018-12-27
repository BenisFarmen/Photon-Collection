from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
#from photonai.photonlogger.Logger import Logger


#Bilateral Filter
class OpenVC_BilateralFilter(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, filter_size = 9, sigmacolour = 75,
                 sigmaspace = 75, logs=''):
        self.random_state = random_state
        self.filtersize = filter_size
        self.sigmacolour = sigmacolour
        self.sigmaspace = sigmaspace

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
            X_reordered[i,:,:] = cv.bilateralFilter(X[i], self.filtersize, self.sigmacolour, self.sigmaspace) #make kernel dims a hyperparameter

        return X_reordered


#Laplacian Filter
class OpenCV_Laplacian(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, ddepth = 2, k_size = 5,
                 filter_size2 = 3, logs=''):
        self.random_state = random_state
        self.ddepth = ddepth
        self.k_size = k_size

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
            X_reordered[i,:,:] = cv.Laplacian(X[i], self.k_size)

        return X_reordered


#Canny Edge Detection Filter
class OpenCV_Canny(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, max_val = 200, min_val = 100, logs=''):
        self.random_state = random_state
        # values for filter max/min
        self.max_val = max_val
        self.min_val = min_val

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
            X_reordered[i,:,:] = cv.Canny(X[i], self.max_val, self.min_val)

        return X_reordered



# Sobel Filter X Direction
class OpenCV_SobelX(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, k_size = 5, logs=''):
        self.random_state = random_state
        # k_size is kernel size
        self.k_size = k_size

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
            X_reordered[i,:,:] = cv.Sobel(X[i], np.uint8, 1, 0, ksize=self.k_size)

        return X_reordered



# Sobel Filter Y Direction
class OpenCV_SobelY(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, k_size = 5, logs=''):
        self.random_state = random_state
        # k_size is kernel size
        self.k_size = k_size

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
            X_reordered[i,:,:] = cv.Sobel(X[i], np.uint8, 0, 1, ksize=self.k_size)

        return X_reordered


# Scharr Filter
class OpenCV_Scharr(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, ddepth = 5, dx = 1, dy = 1, logs=''):
        self.random_state = random_state
        # These hyperparameters control the Scharr Filter operation
        self.ddepth = ddepth
        self.dx = dx
        self.dy = dy

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

        for i in range(X.shape[0]):
            X_reordered[i, :, :] = cv.Scharr(X[i], self.ddepth, self.dx, self.dy)

        return X_reordered