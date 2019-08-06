import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import numpy as np
import torch
from torch_geometric.data import Data, dataset, DataLoader
from torch_geometric.nn import GCNConv, DynamicEdgeConv, global_max_pool, global_mean_pool, SplineConv, graclus, max_pool, max_pool_x
from torch.nn import Sequential as Seq, Dropout, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.utils import from_scipy_sparse_matrix, normalized_cut
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ShuffleSplit
from photonai.photonlogger.Logger import Logger
from photonai.modelwrapper.KerasBaseEstimator import KerasBaseEstimator


class GraphClassificationNet(torch.nn.Module):
    def __init__(self, dataset):
        self.dataset = dataset

        super(GraphClassificationNet, self).__init__()
        self.conv1 = SplineConv(dataset.num_node_features, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 3)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_mean_pool(x, batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)

        return F.log_softmax(self.fc2(x), dim=1)


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


class GraphClassificationConvNet(BaseEstimator, ClassifierMixin, KerasBaseEstimator):

    def __init__(self, hidden_layer_sizes=[10, 20], dropout_rate=0.5, target_dimension=10, act_func='prelu',
                 learning_rate=0.1, batch_normalization=True, nb_epoch=100, early_stopping_flag=True,
                 eaSt_patience=20, reLe_factor = 0.4, reLe_patience=5, batch_size=32, verbosity=0):
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

        #transform y to long format
        y = y.long()

        #disentangle adjacency matrix and graph
        adjacency = X[:,:,:,1]
        X = X[:, :, :, 0]

        #make torch tensor
        X = torch.from_numpy(X)
        X = X.float()

        #make data list for the Data_loader
        data_list = []

        #to scipy_sparse_matrix and to COO format
        for matrix in range(adjacency.shape[0]):
            #tocoo is already called in from from_scipy_sparse_matrix
            adjacency_matrix = coo_matrix(adjacency[matrix, :, :])
            edge_index, edge_attributes = from_scipy_sparse_matrix(adjacency_matrix)
            #call the right X matrix
            X_matrix = X[matrix, :, :]
            #initialize the right y value
            y_value = y[matrix]
            #build data object
            data_list.append(Data(x=X_matrix, edge_index=edge_index, edge_attr=edge_attributes, y=y_value))

        list_cutoff = int(len(data_list) * 0.8)

        init_data = data_list[0]

        #build a pseudo test-set
        train_dataset = data_list[:list_cutoff]
        test_dataset = data_list[list_cutoff:]

        #initialize the data loader
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #Call the model class with the right parameters
        model = GraphClassificationNet(dataset=init_data).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        for epoch in range(1, self.nb_epoch):
            model.train()

            if epoch == 16:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.001

            if epoch == 26:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.0001

            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                F.nll_loss(model(data), data.y).backward()
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
    def __init__(self, dataset):
        self.dataset = dataset
        super(Net, self).__init__()
        #TODO: can I reference y here, aka do I get the info from Photon?
        #TODO: Make sure this net will predict a graph level output, do I need to add the right layers as pooling?
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        self.conv2 = GCNConv(16, 2, cached=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphConvNet(BaseEstimator, ClassifierMixin, KerasBaseEstimator):

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

        #make y targets for testing
        y= np.random.rand(114)
        y[y > 0.5] = 1
        y[y < 0.5] = 0
        y = torch.from_numpy(y)
        y = y.long()

        #disentangle adjacency matrix and graph
        adjacency = X[:,:,:,1]
        X = X[1, :, :, 0]
        #make torch tensor
        X = torch.from_numpy(X)
        X = X.float()

        #to scipy_sparse_matrix and to COO format
        for matrix in range(adjacency.shape[0]):
            #tocoo is already called in from from_scipy_sparse_matrix
            adjacency_matrix = coo_matrix(adjacency[matrix, :, :])
            edge_index, edge_attributes = from_scipy_sparse_matrix(adjacency_matrix)

        #specify training and testing nodes
        num_nodes = X.shape[0]
        perm = torch.randperm(num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.uint8)
        train_mask[perm[0:50]] = 1
        val_mask = torch.zeros(num_nodes, dtype=torch.uint8)
        val_mask[perm[50:100]] = 1
        test_mask = torch.zeros(num_nodes, dtype=torch.uint8)
        test_mask[perm[100:-1]] = 1

        #create dataobject that you can feed into GCN
        data = Data(x=X, edge_index=edge_index, edge_attr=edge_attributes, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #Call the model class with the right parameters
        model = Net(dataset=data).to(device)
        data = data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        model.train()
        for epoch in range(200):
            #TODO: check that the right loss is being used
            print('Training bitches')
            optimizer.zero_grad()
            out = model(data=data)
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






#dgcnn network architecture

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class DGCNNet(torch.nn.Module):
    def __init__(self, out_channels, k=20, aggr='max'):
        self.dataset = dataset
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = MLP([128 + 64, 1024])

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, out_channels))

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)


class DGCNN(BaseEstimator, ClassifierMixin, KerasBaseEstimator):

    def __init__(self, hidden_layer_sizes=[10, 20], dropout_rate=0.5, target_dimension=10, act_func='prelu',
                 learning_rate=0.1, batch_normalization=True, nb_epoch=100, early_stopping_flag=True,
                 eaSt_patience=20, reLe_factor = 0.4, reLe_patience=5, batch_size=32, verbosity=0):
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

        #transform y to long format
        y = y.long()

        #disentangle adjacency matrix and graph
        adjacency = X[:,:,:,1]
        X = X[:, :, :, 0]

        #make torch tensor
        X = torch.from_numpy(X)
        X = X.float()

        data_list = []

        #to scipy_sparse_matrix and to COO format
        for matrix in range(adjacency.shape[0]):
            #tocoo is already called in from from_scipy_sparse_matrix
            adjacency_matrix = coo_matrix(adjacency[matrix, :, :])
            edge_index, edge_attributes = from_scipy_sparse_matrix(adjacency_matrix)
            #call the right X matrix
            X_matrix = X[matrix, :, :]
            #initialize the right y value
            y_value = y[matrix]
            #build data object
            data_list.append(Data(x=X_matrix, edge_index=edge_index, edge_attr=edge_attributes, y=y_value))

        list_cutoff = len(data_list) * 0.8

        #build a pseudo test-set
        train_dataset = data_list[:list_cutoff]
        test_dataset = data_list[list_cutoff:]

        #initialize the data loader
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #Call the model class with the right parameters
        model = DGCNNet(dataset=data).to(device)
        data = data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        model.train()
        for epoch in range(200):
            #TODO: check that the right loss is being used
            print('Training bitches')
            optimizer.zero_grad()
            out = model(data=data)
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
