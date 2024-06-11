"""
Decoder modules for dynamic link prediction

"""

import torch
import torch.nn.init as init
from torch.nn import Linear
import torch.nn.functional as F

class DecoderTGNPL(torch.nn.Module):
    """
    Decoder for TGN-PL, used for link prediction and amount prediction.
    Compared to TGN decoder (below):
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