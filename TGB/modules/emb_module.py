"""
GNN-based modules used to transform node memory to node embedding.
See TGN for options: https://arxiv.org/pdf/2006.10637.pdf
- Identity
- Temporal Graph Attention
- Temporal Graph Sum
"""

import math
from torch_geometric.nn import TransformerConv, GraphConv
from torch.nn import ReLU, BatchNorm1d, Dropout
import torch


class IdentityEmbedding(torch.nn.Module):
    """
    The simplest embedding module which directly uses the node memory as its embedding.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x, last_update, edge_index, t, msg):
        return x
        
    
class GraphAttentionEmbedding(torch.nn.Module):
    """
    Reference:
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
    Our changes: added second convolutional layer since we have bipartite graph,
    added batch norm.
    """

    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv1 = TransformerConv(
            in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim
        )
        self.bns = BatchNorm1d(out_channels)
        self.relu = ReLU()
        self.conv2 = TransformerConv(
            out_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim
        )

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bns(x)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x
    
    
class GraphSumEmbedding(torch.nn.Module):
    """
    Simpler GNN embedding that uses GraphConv and sum aggregation.
    Similar to Temporal Graph Sum from TGN.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GraphConv(in_channels, out_channels, aggr="add")
        self.bns = BatchNorm1d(out_channels)
        self.relu = ReLU()
        self.dropout = Dropout(p=0.5, inplace=False)
        self.conv2 = GraphConv(out_channels, out_channels, aggr="add")

    def forward(self, x, last_update, edge_index, t, msg):
        x = self.conv1(x, edge_index)
        x = self.bns(x)
        x = self.relu(x)
        if self.training:
            x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x
    

class TimeEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        class NormalLinear(torch.nn.Linear):
            # From TGN code: From JODIE code
            def reset_parameters(self):
                stdv = 1.0 / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer = NormalLinear(1, self.out_channels)

    def forward(self, x, last_update, t):
        rel_t = last_update - t
        embeddings = x * (1 + self.embedding_layer(rel_t.to(x.dtype).unsqueeze(1)))

        return embeddings
