import torch
from torch.nn import Linear

from ir.handlers import handlers
from ir.trace import get_backprop_memory, trace
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv
from utils.logger import create_logger
from utils.visualizer import *
from utils.visualizer import plot_gradients


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.

        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h


def gnn_graph():
    dataset = KarateClub()
    data = dataset[0]
    model = GCN()

    _, h = model(data.x, data.edge_index)
    gnn_graph = trace(model, (data.x, data.edge_index))
    return gnn_graph
