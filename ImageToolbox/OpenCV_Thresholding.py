from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
#from photonai.photonlogger.Logger import Logger



class OpenCV_Thresholding(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, max_val=200, min_val=100,
                 filter_type = cv.THRESH_BINARY, logs=''):
        self.random_state = random_state

        # filter types: cv.THRESH_BINARY, cv.THRESH_BINARY_INV, cv.THRESH_TRUNC, cv.THRESH_TOZERO, cv.THRESH_TOZERO_INV
        self.min_val = min_val
        self.max_val = max_val
        self.filter_type = filter_type

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
            X_reordered[i, :, :] = cv.threshold(X[i], self.min_val, self.max_val, self.filter_type)

        return X_reordered




class OpenCV_AdativeThresholding(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, max_val=200, min_val=100,
                 filter_type = cv.ADAPTIVE_THRESH_MEAN_C, logs=''):
        self.random_state = random_state

        # filter types: cv.ADAPTIVE_THRESH_GAUSSIAN_C
        self.min_val = min_val
        self.max_val = max_val
        self.filter_type = filter_type

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
            X_reordered[i, :, :] = cv.adaptiveThreshold(X[i],255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)

        return X_reordered




class OpenCV_OtsuThresholding(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, max_val=200, min_val=100,
                 gaussian_filtering = 0, kernel_size1 = 5, kernel_size2 = 5, logs=''):
        self.random_state = random_state

        # filter types: cv.ADAPTIVE_THRESH_GAUSSIAN_C
        self.min_val = min_val
        self.max_val = max_val
        self.gaussian_filtering = gaussian_filtering
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2

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

        if self.gaussian_filtering == 1:
            for i in range(X.shape[0]):
                X_reordered[i, :, :] = cv.GaussianBlur(X[i], (self.kernel_size1, self.kernel_size2), 0)
            for i in range(X_reordered.shape[0]):
                X_reordered[i, :, :] = cv.threshold(X[i],0, 255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        else:
            for i in range(X.shape[0]):
                X_reordered[i, :, :] = cv.threshold(X[i],0, 255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        return X_reordered
