import numpy
from sklearn.base import BaseEstimator, TransformerMixin
import os

class PhotonVectorizer(BaseEstimator, TransformerMixin):
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
        X_reordered = numpy.reshape(X, (X.shape[0], -1))
        return X_reordered

    # def set_params(self, **params):
    #     if 'n_components' in params:
    #         self.n_clusters = params['n_components']
    #     if 'logs' in params:
    #         self.logs = params.pop('logs', None)
    #
    #     if not self.biclustModel:
    #         self.biclustModel = self.createBiclustering()
    #     self.biclustModel.set_params(**params)
    #
    # def get_params(self, deep=True):
    #     if not self.biclustModel:
    #         self.biclustModel = self.createBiclustering()
    #     biclust_dict = self.biclustModel.get_params(deep)
    #     biclust_dict['logs'] = self.logs
    #     return biclust_dict
