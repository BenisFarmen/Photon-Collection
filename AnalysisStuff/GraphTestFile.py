from GRAPH.GraphConstruction import GraphConstructorPercentage
from GRAPH.GraphConvNet import GraphClassificationConvNet
from GRAPH.GraphMeasureTransform import GraphMeasureTransform
from GRAPH.GraphUtilities import DenseToNetworkx
import numpy as np
from scipy.sparse import csr_matrix
import torch
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data

RandomCtrlData = np.random.rand(10, 114, 114, 2)
RandomLabelData = np.random.rand(10)

y= np.random.rand(114)
y[y > 0.5] = 1
y[y < 0.5] = 1
y = torch.from_numpy(y)
y = y.long()

constructor = GraphConstructorPercentage()
transformed_data = constructor.transform(X=RandomCtrlData)

converter = DenseToNetworkx(adjacency_axis=1)
networkxgraphs = converter.transform(X=transformed_data)

GlobalMeasures = GraphMeasureTransform(global_efficiency=1, global_reaching_centrality=1, graph_clique_number=1)
measures = GlobalMeasures.transform(X=networkxgraphs)

GCN = GraphClassificationConvNet()
train = GCN.fit(X=transformed_data, y= y)



adjacency = transformed_data[1,:,:,1]
X = transformed_data[1, :, :, 0]
X = torch.from_numpy(X)
print(X.type())
adjacency_matrix = csr_matrix(adjacency)
edge_index, edge_attributes = from_scipy_sparse_matrix(adjacency_matrix)
X = X.float()

num_nodes = X.shape[0]
perm = torch.randperm(num_nodes)
train_mask = torch.zeros(num_nodes, dtype=torch.uint8)
train_mask[perm[0:50]] = 1
val_mask = torch.zeros(num_nodes, dtype=torch.uint8)
val_mask[perm[50:100]] = 1
test_mask = torch.zeros(num_nodes, dtype=torch.uint8)
test_mask[perm[100:-1]] = 1

data = Data(x=X, edge_index=edge_index, edge_attr=edge_attributes, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


import os.path as osp

import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa


#dataset = 'Cora'
#path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
#dataset = Planetoid(path, dataset, T.NormalizeFeatures())
#data = dataset[0]
#print(type(data))


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, 16, cached=True)
        self.conv2 = GCNConv(16, 2, cached=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
model = model.float()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
print(log.format(epoch, train_acc, best_val_acc, test_acc))
