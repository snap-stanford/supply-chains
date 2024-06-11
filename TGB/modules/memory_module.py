"""
Memory Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""


import copy
from typing import Callable, Dict, Tuple
import os

import torch
from torch import Tensor
from torch.nn import GRUCell, RNNCell, Linear, Parameter, Embedding
import torch.nn.functional as F

from torch_geometric.nn.inits import zeros
from torch_geometric.utils import scatter

from modules.time_enc import TimeEncoder


TGNMessageStoreType = Dict[int, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]

class TGNPLMemory(torch.nn.Module):
    r"""Our memory model

    Args:
        num_nodes (int): The number of nodes to save memories for.
        raw_msg_dim (int): The raw message dimensionality.
        memory_dim (int): The hidden memory dimensionality.
        time_dim (int): The time encoding dimensionality.
        message_module (torch.nn.Module): The message function which
            combines source and destination node memory embeddings, the raw
            message and the time encoding.
        aggregator_module (torch.nn.Module): The message aggregator function
            which aggregates messages to the same destination into a single
            representation.
    """

    def __init__(
        self,
        num_nodes: int,
        num_prods: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        message_module: Callable,
        aggregator_module: Callable,
        memory_updater_cell: str = "gru",
        update_penalty: float = 1.,
        init_memory_not_learnable: bool = False,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_prods = num_prods
        self.num_firms = self.num_nodes - self.num_prods  # assume we only have firm and prod nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        
        self.msg_s_module = message_module  # msg module for supplier (src)
        self.msg_d_module = copy.deepcopy(message_module)  # msg module for buyer (dst)
        self.msg_p_module = copy.deepcopy(message_module)  # msg module for product (prod)
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)
        self.update_penalty = update_penalty
        self.init_memory_not_learnable = init_memory_not_learnable
        
        if memory_updater_cell == "gru":  # for TGN
            self.memory_updater = GRUCell(message_module.out_channels, memory_dim)
        elif memory_updater_cell == "rnn":  # for JODIE & DyRep
            self.memory_updater = RNNCell(message_module.out_channels, memory_dim)
        else:
            raise ValueError(
                "Undefined memory updater!!! Memory updater can be either 'gru' or 'rnn'."
            )
        if self.init_memory_not_learnable:
            self.init_memory = torch.zeros(self.num_nodes, self.memory_dim)  # initial memory
        else:
            self.init_memory = Embedding(self.num_nodes, self.memory_dim)  # initial memory
            
        self.register_buffer("memory", torch.empty(self.num_nodes, self.memory_dim))  # current memory
        self.register_buffer("last_update", torch.ones(self.num_nodes, dtype=torch.long).to(self.device) * -1)  # -1 represents no update yet
        self.register_buffer("_assoc", torch.empty(self.num_nodes, dtype=torch.long))            

        self.msg_s_store = {}
        self.msg_d_store = {}
        self.msg_p_store = {}
        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        return self.time_enc.lin.weight.device

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if hasattr(self.msg_s_module, "reset_parameters"):
            self.msg_s_module.reset_parameters()
        if hasattr(self.msg_d_module, "reset_parameters"):
            self.msg_d_module.reset_parameters()
        if hasattr(self.msg_p_module, "reset_parameters"):
            self.msg_p_module.reset_parameters()
        if hasattr(self.aggr_module, "reset_parameters"):
            self.aggr_module.reset_parameters()
        self.time_enc.reset_parameters()
        if not self.init_memory_not_learnable:
            self.init_memory.reset_parameters()
        self.memory_updater.reset_parameters()
        self.reset_state()

    def reset_state(self):
        """Resets the memory to its initial state."""
        zeros(self.memory)
        self.last_update = torch.ones(self.num_nodes, dtype=torch.long).to(self.device) * -1  # -1 represents no update yet
        self._reset_message_store()

    def _reset_message_store(self):
        i = self.memory.new_empty((0,), device=self.device, dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
        # Message store format: (src, dst, prod, t, msg)
        self.msg_s_store = {j: (i, i, i, i, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, i, i, msg) for j in range(self.num_nodes)}
        self.msg_p_store = {j: (i, i, i, i, msg) for j in range(self.num_nodes)}

    def detach(self):
        """Detaches the memory from gradient computation."""
        self.memory.detach_()

    def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp."""
        if self.training:
            memory, last_update, update_loss = self._get_updated_memory(n_id)
        else:
            memory, last_update, update_loss = self.memory[n_id], self.last_update[n_id], 0

        return memory, last_update, update_loss
    
    def update_state(self, src: Tensor, dst: Tensor, prod: Tensor, t: Tensor, raw_msg: Tensor):
        """Updates the memory with newly encountered interactions
        :obj:`(src, dst, prod, t, raw_msg)`."""
        # update memory for all nodes that appeared in this batch
        n_id = torch.cat([src, dst, prod]).unique()  

        if self.training:
            self._update_memory(n_id)
            self._update_msg_store(src, dst, prod, t, raw_msg, self.msg_s_store, key="src")
            self._update_msg_store(src, dst, prod, t, raw_msg, self.msg_d_store, key="dst")
            self._update_msg_store(src, dst, prod, t, raw_msg, self.msg_p_store, key="prod")
        else:
            self._update_msg_store(src, dst, prod, t, raw_msg, self.msg_s_store, key="src")
            self._update_msg_store(src, dst, prod, t, raw_msg, self.msg_d_store, key="dst")
            self._update_msg_store(src, dst, prod, t, raw_msg, self.msg_p_store, key="prod")
            self._update_memory(n_id)
        
    def _update_memory(self, n_id: Tensor):
        """
        Update the stored memory and last update for nodes in n_id.
        """
        memory, last_update, update_loss = self._get_updated_memory(n_id)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    def _get_updated_memory(self, n_id: Tensor, prod_emb: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Get current memory and last update for nodes in n_id, using their 
        current stored interactions.
        """
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)  # used to reindex n_id from 0
        
        # Compute messages to suppliers in interactions where n_id is supplier
        msg_s, t_s, idx_s, _, _ = self._compute_msg(
            n_id, self.msg_s_store, self.msg_s_module
        ) 

        # Compute messages to buyers in interactions where n_id is buyer
        msg_d, t_d, _, idx_d, _ = self._compute_msg(
            n_id, self.msg_d_store, self.msg_d_module
        ) 

        # Compute messages to products in interactions where n_id is product
        msg_p, t_p, _, _, idx_p = self._compute_msg(
            n_id, self.msg_p_store, self.msg_p_module
        )

        # Aggregate messages
        idx = torch.cat([idx_s, idx_d, idx_p], dim=0)
        msg = torch.cat([msg_s, msg_d, msg_p], dim=0)
        t = torch.cat([t_s, t_d, t_p], dim=0)
        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))  # one aggregated msg per node

        # Get old memory
        memory = self.memory[n_id, :self.memory_dim]
        last_update = self.last_update[n_id]
        use_init = last_update == -1  # if this node has never been updated, use initial memory
        if self.init_memory_not_learnable:
            self.init_memory = self.init_memory.to(self.device)
            memory[use_init] = self.init_memory[n_id[use_init]]
        else:
            memory[use_init] = self.init_memory(n_id[use_init])
        
        # Get updated memory
        new_memory = self.memory_updater(aggr, memory.clone())
        has_new_messages = torch.isin(n_id, idx)  # if this node has a new messages, update its memory and last update
        if has_new_messages.sum() > 0:
            delta = torch.linalg.vector_norm(new_memory[has_new_messages] - memory[has_new_messages])  # norm change in memory
            update_loss = self.update_penalty * delta / len(n_id)  # loss is scaled by batch size in tgnpl.py
        else:
            update_loss = 0  # set to 0 so we don't get nan
        memory[has_new_messages] = new_memory[has_new_messages]

        # Get updated last update
        msg_update = scatter(t, idx, 0, self.num_nodes, reduce="max")[n_id]
        last_update[has_new_messages] = msg_update[has_new_messages]        
        return memory, last_update, update_loss
        
    def _compute_msg(
        self, n_id: Tensor, msg_store: TGNMessageStoreType, msg_module: Callable
    ):
        """
        Compute messages for nodes in n_id based on the interactions stored in msg_store.
        The length of the returned msg is the total number of interactions keyed by n_id
        in msg_store.
        """
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, prod, t, raw_msg = list(zip(*data))  # unzip
        src = torch.cat(src, dim=0)
        dst = torch.cat(dst, dim=0)
        prod = torch.cat(prod, dim=0)
        t = torch.cat(t, dim=0)
        raw_msg = torch.cat(raw_msg, dim=0)
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

        msg = msg_module(self.memory[src], self.memory[dst], self.memory[prod], raw_msg, t_enc)

        return msg, t, src, dst, prod
    
    def _update_msg_store(
        self,
        src: Tensor,
        dst: Tensor,
        prod: Tensor,
        t: Tensor,
        raw_msg: Tensor,
        msg_store: TGNMessageStoreType,
        key: str = "src",
    ):
        if key == "src":
            key = src
        elif key == "dst":
            key = dst
        elif key == "prod":
            key = prod
        else:
            raise Exception(f"Invalid key in _update_msg_store: {key}")
        n_id, perm = key.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        # map each node to its interactions where it is in the key role (src, dst, or prod)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], prod[idx], t[idx], raw_msg[idx])
    
    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            self._update_memory(torch.arange(self.num_nodes, device=self.memory.device))
            self._reset_message_store()
        super().train(mode)
 
class StaticMemory(torch.nn.Module):
    """
    Simplest version of memory that just holds onto a learnable, static vector per node.
    """
    def __init__(self, num_nodes: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        # default init: from N(0, 1)
        self.memory = Embedding(self.num_nodes, self.memory_dim)
        self.time_enc = TimeEncoder(time_dim)
        self.register_buffer("last_update", torch.empty(self.num_nodes, dtype=torch.long))
        self.reset_parameters()
        
    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        self.memory.reset_parameters()
        self.time_enc.reset_parameters()
        self.reset_state()
        
    def reset_state(self):
        """Resets last update to its initial state."""
        zeros(self.last_update)
    
    def detach(self):
        """Detaches the memory from gradient computation.
        Nothing to do for this class - can't detach Embedding."""
        pass
        
    def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp."""
        return self.memory(n_id), self.last_update[n_id], 0
    
    def update_state(self, src: Tensor, dst: Tensor, prod: Tensor, t: Tensor, raw_msg: Tensor):
        """Updates the memory with newly encountered interactions.
        Nothing to do for static memory."""
        pass
        
    