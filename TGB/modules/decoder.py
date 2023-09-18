"""
Decoder modules for dynamic link prediction

"""

import torch
import torch.nn.init as init
from torch.nn import Linear
import torch.nn.functional as F


class LinkPredictorTGNPL(torch.nn.Module):
    """
    Link predictor for TGN-PL. Compared to TGN decoder (below):
    - we have three input embeddings instead of two, because of hyperedge,
    - we concatenate then pass through lin_hidden,
    - we remove the final sigmoid since our loss, softmax cross-entropy,
      uses non-normalized logits.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.lin_hidden = Linear(in_channels * 3, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst, z_prod):
        h = self.lin_hidden(torch.cat([z_src, z_dst, z_prod], dim=1))
        h = h.relu()
        return self.lin_final(h)


class LinkPredictor(torch.nn.Module):
    """
    Reference:
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
    """

    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h).sigmoid()


class NodePredictor(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_node = Linear(in_dim, in_dim)
        self.out = Linear(in_dim, out_dim)

    def forward(self, node_embed):
        h = self.lin_node(node_embed)
        h = h.relu()
        h = self.out(h)
        output = F.log_softmax(h, dim=-1)
        return output
