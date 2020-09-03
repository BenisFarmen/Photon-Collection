import os


class DenseToNetworkxTransformer(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    # turns a dense adjacency matrix coming from a photonai_graph constructor into a networkx photonai_graph

    def __init__(self, adjacency_axis = 0,
                 logs=''):
        self.adjacency_axis = adjacency_axis
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        return self

    def transform(self, X):

        graph_list = []

        for i in range(X.shape[0]):
            networkx_graph = from_numpy_matrix(A=X[i, :, :, self.adjacency_axis])
            graph_list.append(networkx_graph)

        X = graph_list

        return X


class DenseToTorchGeometricTransformer(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    # turns a dense adjacency and feature matrix coming 
    # from a photonai_graph constructor into a pytorch geometric data object

    def __init__(self, adjacency_axis = 0,
                 concatenation_axis = 3, data_as_list = 1,
                 logs=''):
        self.adjacency_axis = adjacency_axis
        self.concatenation_axis = concatenation_axis
        self.data_as_list = data_as_list
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        return self

    def transform(self, X, y):

        if self.data_as_list == 1:

            # transform y to long format
            y = y.long()

            # disentangle adjacency matrix and photonai_graph
            adjacency = X[:, :, :, 1]
            feature_matrix = X[:, :, :, 0]

            # make torch tensor
            feature_matrix = torch.from_numpy(feature_matrix)
            feature_matrix = feature_matrix.float()

            # make data list for the Data_loader
            data_list = []

            # to scipy_sparse_matrix and to COO format
            for matrix in range(adjacency.shape[0]):
                # tocoo is already called in from from_scipy_sparse_matrix
                adjacency_matrix = coo_matrix(adjacency[matrix, :, :])
                edge_index, edge_attributes = from_scipy_sparse_matrix(adjacency_matrix)
                # call the right X matrix
                X_matrix = X[matrix, :, :]
                # initialize the right y value
                y_value = y[matrix]
                # build data object
                data_list.append(Data(x=X_matrix, edge_index=edge_index, edge_attr=edge_attributes, y=y_value))

            X = data_list

        return X



def GraphConverter(X, y, conversion_type = 'DenseToNetworkx', adjacency_axis = 0):

    # Convert a Dense Graph format to a Networkx format
    if conversion_type == 'DenseToNetworkx':
        #print('converting Dense to Networkx')
        graph_list = []

        for i in range(X.shape[0]):
            networkx_graph = from_numpy_matrix(A=X[i, :, :, adjacency_axis])
            graph_list.append(networkx_graph)

        X_converted = graph_list

    elif conversion_type == 'NetworkxToNumpyDense':
        #print('converting Networkx to Numpy Dense')
        graph_list = []

        for i in X:
            numpy_matrix = to_numpy_matrix(i)
            graph_list.append(numpy_matrix)

        X_converted = graph_list

    elif conversion_type == 'DenseToTorchGeometric':
        #print('converting Dense to Torch Geometric')
        if isinstance(X, list):

            # transform y to long format
            y = y.long()

            # disentangle adjacency matrix and photonai_graph
            adjacency = X[:, :, :, 1]
            feature_matrix = X[:, :, :, 0]

            # make torch tensor
            feature_matrix = torch.as_tensor(feature_matrix)
            feature_matrix = feature_matrix.float()

            # make data list for the Data_loader
            data_list = []

            # to scipy_sparse_matrix and to COO format
            for matrix in range(adjacency.shape[0]):
                # tocoo is already called in from from_scipy_sparse_matrix
                adjacency_matrix = coo_matrix(adjacency[matrix, :, :])
                edge_index, edge_attributes = from_scipy_sparse_matrix(adjacency_matrix)
                # call the right X matrix
                X_matrix = X[matrix, :, :]
                # initialize the right y value
                y_value = y[matrix]
                # build data object
                data_list.append(Data(x=X_matrix, edge_index=edge_index, edge_attr=edge_attributes, y=y_value))

            X_converted = data_list

    elif conversion_type == 'TorchGeometricToDense':
        # Convert Dense to Torch Geometric
        print('Not implemented yet')
    elif conversion_type == 'NetworkxToTorchGeometric':
        # Convert Networkx to Torch Geometric
        print('Not implemented yet')
    elif conversion_type == 'TorchGeometricToNetworkx':
        # Convert Torch Geometric to Networkx
        print('Not implemented yet')
    elif conversion_type == 'GraphEmbeddings':
        # Convert GraphEmbeddings?
        print('Not implemented yet')
