from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import SpectralBiclustering
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
#from photonai.photonlogger.Logger import Logger

class OpenCV_Resize(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, resize_factor = 0.8, random_state=42, logs='',
                 interpolation_method = cv.INTER_AREA):
        self.random_state = random_state
        self.resize_factor = resize_factor
        self.interpolation_method = interpolation_method
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

    def __init__(self, random_state=42, logs=''):
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
            X_reordered[i,:,:] = cv.pyrDown(X[i])

        X_reordered = cv.pyrDown(X)
        return X_reordered


class OpenCV_Smooth(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, filtertype = 'CV_GAUSSIAN', filter_size1 = 3,
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

        for i in range (X.shape[0]):
            X_reordered[i,:,:] = cv.Smooth(X[i], smoothtype = self.filter)

        X_reordered = cv.pyrDown(X)
        return X_reordered

class OpenCV_Blur(BaseEstimator, TransformerMixin):
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
            X_reordered[i,:,:] = cv.blur(X[i], (5,5)) #make kernel dims a hyperparameter

        X_reordered = cv.pyrDown(X)
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

        X_reordered = cv.pyrDown(X)
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


class OpenVC_BilateralFilter(BaseEstimator, TransformerMixin):
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
            X_reordered[i,:,:] = cv.bilateralFilter(X[i], 9, 75, 75) #make kernel dims a hyperparameter

        return X_reordered


#Laplacian Filter
class OpenCV_Laplacian(BaseEstimator, TransformerMixin):
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
            X_reordered[i,:,:] = cv.Laplacian(X[i])

        return X_reordered


#Canny Edge Detection Filter
class OpenCV_Canny(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, filtertype = 'CV_GAUSSIAN', filter_size1 = 3,
                 filter_size2 = 3, max_val = 200, min_val = 100, logs=''):
        self.random_state = random_state
        # values for
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

    def __init__(self, random_state=42, filtertype = 'CV_GAUSSIAN', filter_size1 = 3,
                 filter_size2 = 3, max_val = 200, min_val = 100, logs=''):
        self.random_state = random_state
        # values for
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
            X_reordered[i,:,:] = cv.Sobel(X[i], np.uint8, 1, 0, ksize=5)

        return X_reordered


# Sobel Filter Y Direction
class OpenCV_SobelY(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, filtertype = 'CV_GAUSSIAN', filter_size1 = 3,
                 filter_size2 = 3, max_val = 200, min_val = 100, logs=''):
        self.random_state = random_state
        # make ksize a hyperparameter?

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
            X_reordered[i,:,:] = cv.Sobel(X[i], np.uint8, 0, 1, ksize=5)

        return X_reordered


class OpenCV_Scharr(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, random_state=42, filtertype='CV_GAUSSIAN', filter_size1=3,
                 filter_size2=3, max_val=200, min_val=100, logs=''):
        self.random_state = random_state
        # make ksize a hyperparameter?

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
            X_reordered[i, :, :] = cv.Scharr() #add parameters

        return X_reordered


# Erosion
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(img,kernel,iterations = 1)

# Dilation
dilation = cv.dilate(img,kernel,iterations = 1)

# Opening
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

# Closing
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

# Gradient
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

#TopHat
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)

# Black Hat
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

# thresholding
ret,thresh1 = cv.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

# adaptive thresholding
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)

# Otsu's thresholding
ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# Translation
M = np.float32([[1,0,100],[0,1,50]])
dst = cv.warpAffine(img,M,(cols,rows))

# Rotation
rows,cols = img.shape
M = cv.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = cv.warpAffine(img,M,(cols,rows))

#still need to add the Fourier transformations
