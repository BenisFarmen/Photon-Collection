import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dropout, Dense
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ShuffleSplit
from photonai.photonlogger.Logger import Logger
from photonai.helpers.TFUtilities import binary_to_one_hot
from photonai.modelwrapper.KerasBaseEstimator import KerasBaseEstimator

class KerasDNNClassifier(BaseEstimator, ClassifierMixin, KerasBaseEstimator):

    def __init__(self, hidden_layer_sizes=[10, 20], dropout_rate=0.5, target_dimension=10, act_func='prelu',
                 learning_rate=0.1, batch_normalization=True, nb_epoch=100, early_stopping_flag=True,
                 eaSt_patience=20, reLe_factor = 0.4, reLe_patience=5, batch_size=64, verbosity=0):
        super(KerasBaseEstimator, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rate = dropout_rate
        self.act_func = act_func
        self.learning_rate = learning_rate
        self.target_dimension = target_dimension
        self.batch_normalization = batch_normalization
        self.nb_epoch = nb_epoch
        self.early_stopping_flag = early_stopping_flag
        self.eaSt_patience = eaSt_patience
        self.reLe_factor = reLe_factor
        self.reLe_patience = reLe_patience
        self.batch_size = batch_size
        self.verbosity = verbosity


    def fit(self, X, y):

        #disentangle adjacency matrix and graph
        adjacency = X[:,:,:,1]
        X = X[:,:,:,0]

        #TODO: disentangle data, so that the dataset can be used as intended by the model class

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Net().to(device)
        data = dataset[0].to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        model.train()
        for epoch in range(200):
            #TODO: check that the right loss is being used
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X):
        predict_result = self.model.predict(X, batch_size=self.batch_size)
        max_index = np.argmax(predict_result, axis=1)
        return max_index

    def predict_proba(self, X):
        """
        Predict probabilities
        :param X: array-like
        :type data: float
        :return: predicted values, array
        """
        return self.model.predict(X, batch_size=self.batch_size)

    def score(self, X, y_true):
        return np.zeros(1)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #TODO: extract num_features to construct features
        #TODO: can I reference y here, aka do I get the info from Photon?
        #TODO: Make sure this net will predict a graph level output, do I need to add the right layers as pooling?
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)