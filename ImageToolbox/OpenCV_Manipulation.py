from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
#from photonai.photonlogger.Logger import Logger



class OpenCV_Erosion(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, kernel_size1= 5, kernel_size2= 5,
                 filter_size2=3, max_val=200, min_val=100, logs=''):
        self.random_state = random_state

        # TODO: Make it possible to choose kernel, maybe define it in main file
        # make kernel a hyperparameter!
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2

        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        kernel = np.ones((self.kernel_size1, self.kernel_size2), np.uint8)

        dim_0 = int(X.shape[0])
        dim_1 = int(X.shape[1])
        dim_2 = int(X.shape[2])

        X_reordered = np.empty([dim_0, dim_1, dim_2])

        for i in range(X.shape[0]):
            X_reordered[i, :, :] = cv.erode(X[i],kernel,iterations = 1)

        return X_reordered




class OpenCV_Dilation(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, kernel_size1=5, kernel_size2=5,
                 filter_size2=3, max_val=200, min_val=100, logs=''):
        self.random_state = random_state

        # TODO: Make it possible to choose kernel, maybe define it in main file
        # make kernel a hyperparameter!
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2

        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        kernel = np.ones((self.kernel_size1, self.kernel_size2), np.uint8)

        dim_0 = int(X.shape[0])
        dim_1 = int(X.shape[1])
        dim_2 = int(X.shape[2])

        X_reordered = np.empty([dim_0, dim_1, dim_2])

        for i in range(X.shape[0]):
            X_reordered[i, :, :] = cv.dilate(X[i], kernel, iterations=1)

        return X_reordered




class OpenCV_Opening(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, kernel_size1=5, kernel_size2=5,
                 filter_size2=3, max_val=200, min_val=100, logs=''):
        self.random_state = random_state

        # TODO: Make it possible to choose kernel, maybe define it in main file
        # make kernel a hyperparameter!
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2

        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        kernel = np.ones((self.kernel_size1, self.kernel_size2), np.uint8)

        dim_0 = int(X.shape[0])
        dim_1 = int(X.shape[1])
        dim_2 = int(X.shape[2])

        X_reordered = np.empty([dim_0, dim_1, dim_2])

        for i in range(X.shape[0]):
            X_reordered[i, :, :] = cv.morphologyEx(X[i], cv.MORPH_OPEN, kernel,)

        return X_reordered


class OpenCV_Closing(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, kernel_size1=5, kernel_size2=5,
                 filter_size2=3, max_val=200, min_val=100, logs=''):
        self.random_state = random_state

        # TODO: Make it possible to choose kernel, maybe define it in main file
        # make kernel a hyperparameter!
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2

        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        kernel = np.ones((self.kernel_size1, self.kernel_size2), np.uint8)

        dim_0 = int(X.shape[0])
        dim_1 = int(X.shape[1])
        dim_2 = int(X.shape[2])

        X_reordered = np.empty([dim_0, dim_1, dim_2])

        for i in range(X.shape[0]):
            X_reordered[i, :, :] = cv.morphologyEx(X[i], cv.MORPH_CLOSE, kernel)

        return X_reordered





class OpenCV_Closing(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, kernel_size1=5, kernel_size2=5,
                 filter_size2=3, max_val=200, min_val=100, logs=''):
        self.random_state = random_state

        # TODO: Make it possible to choose kernel, maybe define it in main file
        # make kernel a hyperparameter!
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2

        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        kernel = np.ones((self.kernel_size1, self.kernel_size2), np.uint8)

        dim_0 = int(X.shape[0])
        dim_1 = int(X.shape[1])
        dim_2 = int(X.shape[2])

        X_reordered = np.empty([dim_0, dim_1, dim_2])

        for i in range(X.shape[0]):
            X_reordered[i, :, :] = cv.morphologyEx(X[i], cv.MORPH_GRADIENT, kernel)

        return X_reordered





class OpenCV_Closing(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, kernel_size1=5, kernel_size2=5,
                 filter_size2=3, max_val=200, min_val=100, logs=''):
        self.random_state = random_state

        # TODO: Make it possible to choose kernel, maybe define it in main file
        # make kernel a hyperparameter!
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2

        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        kernel = np.ones((self.kernel_size1, self.kernel_size2), np.uint8)

        dim_0 = int(X.shape[0])
        dim_1 = int(X.shape[1])
        dim_2 = int(X.shape[2])

        X_reordered = np.empty([dim_0, dim_1, dim_2])

        for i in range(X.shape[0]):
            X_reordered[i, :, :] = cv.morphologyEx(X[i], cv.MORPH_TOPHAT, kernel)

        return X_reordered


class OpenCV_Closing(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, kernel_size1=5, kernel_size2=5,
                 filter_size2=3, max_val=200, min_val=100, logs=''):
        self.random_state = random_state

        # TODO: Make it possible to choose kernel, maybe define it in main file
        # make kernel a hyperparameter!
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2

        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        kernel = np.ones((self.kernel_size1, self.kernel_size2), np.uint8)

        dim_0 = int(X.shape[0])
        dim_1 = int(X.shape[1])
        dim_2 = int(X.shape[2])

        X_reordered = np.empty([dim_0, dim_1, dim_2])

        for i in range(X.shape[0]):
            X_reordered[i, :, :] = cv.morphologyEx(X[i], cv.MORPH_BLACKHAT, kernel)

        return X_reordered