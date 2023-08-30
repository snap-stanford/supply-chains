"""
Message Function Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""

import torch
from torch import Tensor
from torch.nn import Linear, LayerNorm, init, ReLU

class TGNPLMessage(torch.nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.out_channels = raw_msg_dim + (3 * memory_dim) + time_dim
        self.lin = Linear(self.out_channels, self.out_channels)
        init.kaiming_uniform_(self.lin.weight, nonlinearity="relu")
        self.layer_norm = LayerNorm(self.out_channels)
        self.relu = ReLU()

    def forward(self, z_src: Tensor, z_dst: Tensor, z_prod: Tensor, raw_msg: Tensor, t_enc: Tensor):
        x = torch.cat([z_src, z_dst, z_prod, raw_msg, t_enc], dim=-1)
        x = self.lin(x)
        x = self.layer_norm(x)
        x = self.relu(x)        
        return x
        
    
class IdentityMessage(torch.nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim

    def forward(self, z_src: Tensor, z_dst: Tensor, raw_msg: Tensor, t_enc: Tensor):
        return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)