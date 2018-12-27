from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
#from photonai.photonlogger.Logger import Logger



class OpenCV_Translation(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, logs=''):
        self.random_state = random_state

        # TODO: add parameters for controlling the translation

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

        rows, cols = dim_1, dim_2
        M = np.float32([[1, 0, 100], [0, 1, 50]])

        for i in range(X.shape[0]):
            X_reordered[i, :, :] = cv.warpAffine(X[i], M, (cols, rows))

        return X_reordered


class OpenCV_Rotation(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, logs=''):
        self.random_state = random_state

        # TODO: add parameters for controlling the rotation

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

        rows, cols = dim_1, dim_2
        M = cv.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)

        for i in range(X.shape[0]):
            X_reordered[i, :, :] = cv.warpAffine(X[i], M, (cols, rows))

        return X_reordered