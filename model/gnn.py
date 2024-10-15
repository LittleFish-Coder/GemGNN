import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=16, out_channels=2, add_dropout=True):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 256)  # 768 -> 256
        self.conv2 = GCNConv(256, hidden_channels)  # 256 -> 16
        self.conv3 = GCNConv(hidden_channels, out_channels) # 16 -> 2
        self.add_dropout = add_dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if self.add_dropout:
            x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        if self.add_dropout:
            x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.sigmoid(x)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=16, out_channels=2, add_dropout=True):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 256)
        self.conv2 = GATConv(256, hidden_channels)
        self.conv3 = GATConv(hidden_channels, out_channels)
        self.add_dropout = add_dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if self.add_dropout:
            x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        if self.add_dropout:
            x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.sigmoid(x)
        return x