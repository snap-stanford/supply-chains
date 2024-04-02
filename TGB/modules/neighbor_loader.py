"""
Neighbor Loader

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""
import copy
from typing import Callable, Dict, Tuple

import torch
from torch import Tensor

EMPTY_VALUE = -1

class LastNeighborLoader:
    def __init__(self, num_nodes: int, size: int, device=None):
        self.size = size

        self.neighbors = torch.empty((num_nodes, size), dtype=torch.long, device=device)
        self.e_id = torch.empty((num_nodes, size), dtype=torch.long, device=device)
        self._assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

        self.reset_state()

    def __call__(self, n_id: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        neighbors = self.neighbors[n_id]
        nodes = n_id.view(-1, 1).repeat(1, self.size)
        e_id = self.e_id[n_id]

        # Filter invalid neighbors (identified by `e_id < 0`).
        mask = e_id >= 0
        neighbors, nodes, e_id = neighbors[mask], nodes[mask], e_id[mask]

        # Relabel node indices.
        n_id = torch.cat([n_id, neighbors]).unique()
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        neighbors, nodes = self._assoc[neighbors], self._assoc[nodes]

        return n_id, torch.stack([neighbors, nodes]), e_id

    def insert(self, src: Tensor, dst: Tensor):
        # Inserts newly encountered interactions into an ever-growing
        # (undirected) temporal graph.

        # Collect central nodes, their neighbors and the current event ids.
        neighbors = torch.cat([src, dst], dim=0)
        nodes = torch.cat([dst, src], dim=0)
        e_id = torch.arange(
            self.cur_e_id, self.cur_e_id + src.size(0), device=src.device
        ).repeat(2)
        self.cur_e_id += src.numel()

        # Convert newly encountered interaction ids so that they point to
        # locations of a "dense" format of shape [num_nodes, size].
        nodes, perm = nodes.sort()
        neighbors, e_id = neighbors[perm], e_id[perm]

        n_id = nodes.unique()
        self._assoc[n_id] = torch.arange(n_id.numel(), device=n_id.device)

        dense_id = torch.arange(nodes.size(0), device=nodes.device) % self.size
        dense_id += self._assoc[nodes].mul_(self.size)

        dense_e_id = e_id.new_full((n_id.numel() * self.size,), -1)
        dense_e_id[dense_id] = e_id
        dense_e_id = dense_e_id.view(-1, self.size)

        dense_neighbors = e_id.new_empty(n_id.numel() * self.size)
        dense_neighbors[dense_id] = neighbors
        dense_neighbors = dense_neighbors.view(-1, self.size)

        # Collect new and old interactions...
        e_id = torch.cat([self.e_id[n_id, : self.size], dense_e_id], dim=-1)
        neighbors = torch.cat(
            [self.neighbors[n_id, : self.size], dense_neighbors], dim=-1
        )

        # And sort them based on `e_id`.
        e_id, perm = e_id.topk(self.size, dim=-1)
        self.e_id[n_id] = e_id
        self.neighbors[n_id] = torch.gather(neighbors, 1, perm)

    def reset_state(self):
        self.cur_e_id = 0
        self.e_id.fill_(-1)

class LastNeighborLoaderTGNPL(LastNeighborLoader):
    def __init__(self, num_nodes: int, size: int, device=None):
        super().__init__(num_nodes, size, device) 

    def __call__(self, f_id: Tensor, p_id: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        n_id = torch.cat([f_id, p_id])
        return super().__call__(n_id)

    def insert(self, src: Tensor, dst: Tensor, prod: Tensor):
        for i in range(src.size()[0]):  
            super().insert(src[i:i+1], prod[i:i+1])
            super().insert(dst[i:i+1], prod[i:i+1])
        
    def reset_state(self):
        return super().reset_state()

class LastNeighborLoaderTime:
    '''
    This version makes two main changes to the LastNeighborLoader class:
    1. Attached a timestamp to every edge
    2. __call__() inputs and outputs in the format of (batch_size, ) to meet the need of graphmixer
        a. we only return neighbors instead of a set of both neighbors and nodes
        b. we maintain the same info storage approach in insert(), contrary to the storage approach in TGBBaseline
    '''
    def __init__(self, num_nodes: int, num_neighbors: int, time_gap: int, edge_feat_dim: int, device=None):
        self.size = max(num_neighbors, time_gap) # maximum number of neighbors

        self.neighbors = torch.empty((num_nodes, self.size), dtype=torch.long, device=device)
        self.e_id = torch.empty((num_nodes, self.size), dtype=torch.long, device=device)
        self.t_id = torch.empty((num_nodes, self.size), dtype=torch.long, device=device)
        self.msg = torch.empty((num_nodes, self.size, edge_feat_dim), dtype=torch.float, device=device)
        self._assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
        self.edge_feat_dim = edge_feat_dim

        self.reset_state()

    def __call__(self, n_id: Tensor, size: int = 10) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        '''
        Inputs
        :param n_id: tensor, shape (batch_size, )
        :param size: int, number of neighbors to return for each node, since sometimes we don't load all neighbors stored
        Returns
        :param neighbors: tensor, shape (batch_size, size)
        :param e_id: tensor, shape (batch_size, size)
        :param t_id: tensor, shape (batch_size, size)
        :param msg: tensor, shape (batch_size, size, self.edge_feat_dim)
        '''
        assert size <= self.size, "Error: neighbor loader __call__ size should be <= maximum num neighbors stored"
        neighbors = self.neighbors[n_id][:, :size]
        e_id = self.e_id[n_id][:, :size]
        t_id = self.t_id[n_id][:, :size]
        msg = self.msg[n_id][:, :size, :]

        # Filter invalid neighbors (identified by `e_id < 0`).
        mask = e_id < 0
        neighbors[mask] = EMPTY_VALUE
        e_id[mask] = EMPTY_VALUE
        t_id[mask] = EMPTY_VALUE
        msg[mask] = EMPTY_VALUE

        # Relabel node indices. # NOT NEEDED IN THIS VERSION
        # nodes = n_id.view(-1, 1).repeat(1, self.size)
        # n_id = torch.cat([n_id.unique(), neighbors.unique()]).unique()
        # self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        # neighbors, nodes = self._assoc[neighbors], self._assoc[nodes]
        return neighbors, None, e_id, t_id, msg

    def insert(self, src: Tensor, dst: Tensor, t: Tensor, msg: Tensor):
        # Inserts newly encountered interactions into an ever-growing
        # (undirected) temporal graph.

        # Collect central nodes, their neighbors and the current event ids.
        neighbors = torch.cat([src, dst], dim=0)
        nodes = torch.cat([dst, src], dim=0)
        e_id = torch.arange(
            self.cur_e_id, self.cur_e_id + src.size(0), device=src.device
        ).repeat(2)
        t_id = t.repeat(2)
        msg = msg.repeat(2, 1)
        self.cur_e_id += src.numel()

        # Convert newly encountered interaction ids so that they point to
        # locations of a "dense" format of shape [num_nodes, size].
        nodes, perm = nodes.sort()
        neighbors, e_id, t_id, msg = neighbors[perm], e_id[perm], t_id[perm], msg[perm]

        n_id = nodes.unique()
        self._assoc[n_id] = torch.arange(n_id.numel(), device=n_id.device)

        dense_id = torch.arange(nodes.size(0), device=nodes.device) % self.size
        dense_id += self._assoc[nodes].mul_(self.size)

        dense_e_id = e_id.new_full((n_id.numel() * self.size,), -1)
        dense_e_id[dense_id] = e_id
        dense_e_id = dense_e_id.view(-1, self.size)

        dense_t_id = t_id.new_full((n_id.numel() * self.size,), -1)
        dense_t_id[dense_id] = t_id
        dense_t_id = dense_t_id.view(-1, self.size)
        
        dense_msg = msg.new_full((n_id.numel() * self.size, self.edge_feat_dim, ), -1)
        dense_msg[dense_id] = msg
        dense_msg = dense_msg.view(-1, self.size, self.edge_feat_dim)

        dense_neighbors = e_id.new_empty(n_id.numel() * self.size)
        dense_neighbors[dense_id] = neighbors
        dense_neighbors = dense_neighbors.view(-1, self.size)

        # Collect new and old interactions...
        e_id = torch.cat([self.e_id[n_id, : self.size], dense_e_id], dim=-1)
        t_id = torch.cat([self.t_id[n_id, : self.size], dense_t_id], dim=-1)
        msg = torch.cat(
            [self.msg[n_id, : self.size], dense_msg], dim=-2
        )
        neighbors = torch.cat(
            [self.neighbors[n_id, : self.size], dense_neighbors], dim=-1
        )

        # TODO: should be e_id != -1 then largest timestamp goes first 
        # Sort by time in graphmixer to load most recent neighbors
        # And sort them based on `t_id`, used to be based on `e_id` for TGNPL
        e_id, perm = e_id.topk(self.size, dim=-1) # returns the largest elements 
        self.e_id[n_id] = e_id
        self.t_id[n_id] = torch.gather(t_id, 1, perm)        
        msg_perm = perm.unsqueeze(-1).repeat(1, 1, self.edge_feat_dim) # torch.gather requires index.shape==input.shape regardless of dim
        self.msg[n_id] = torch.gather(msg, 1, msg_perm)
        self.neighbors[n_id] = torch.gather(neighbors, 1, perm)

    def reset_state(self):
        self.cur_e_id = 0
        self.neighbors.fill_(EMPTY_VALUE)
        self.e_id.fill_(EMPTY_VALUE)
        self.t_id.fill_(EMPTY_VALUE)
        self.msg.fill_(EMPTY_VALUE)
        

class LastNeighborLoaderGraphmixer(LastNeighborLoaderTime):
    def __init__(self, num_nodes: int, num_neighbors: int, time_gap: int, edge_feat_dim: int, device=None):
        super().__init__(num_nodes, num_neighbors, time_gap, edge_feat_dim, device) 

    # def __call__(self, f_id: Tensor, p_id: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # n_id = torch.cat([f_id, p_id])
        # return super().__call__(n_id)
    def __call__(self, n_id: Tensor, size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return super().__call__(n_id, size)

    def insert(self, src: Tensor, dst: Tensor, prod: Tensor, t: Tensor, msg: Tensor):
        for i in range(src.size()[0]):  
            super().insert(src[i:i+1], prod[i:i+1], t[i:i+1], msg[i:i+1])
            super().insert(dst[i:i+1], prod[i:i+1], t[i:i+1], msg[i:i+1])
    def reset_state(self):
        return super().reset_state()
