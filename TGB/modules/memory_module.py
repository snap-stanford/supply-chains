"""
Memory Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""


import copy
from typing import Callable, Dict, Tuple

import torch
from torch import Tensor
from torch.nn import GRUCell, RNNCell, Linear, Parameter
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
        state_dim: int,
        time_dim: int,
        message_module: Callable,
        aggregator_module: Callable,
        state_updater_cell: str = "gru",
        use_inventory: bool = True,
        learn_att_direct: bool = False,
        debt_penalty: float = 0,
        consumption_reward: float = 0,
        debug: bool = False,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_prods = num_prods
        self.num_firms = self.num_nodes - self.num_prods  # assume we only have firm and prod nodes
        self.raw_msg_dim = raw_msg_dim
        self.state_dim = state_dim
        self.time_dim = time_dim
        
        self.msg_s_module = message_module  # msg module for supplier (src)
        self.msg_d_module = message_module  # msg module for buyer (dst)
        self.msg_p_module = message_module  # msg module for product (prod)
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)
        self.use_inventory = use_inventory
        self.learn_att_direct = learn_att_direct
        self.debt_penalty = debt_penalty
        self.consumption_reward = consumption_reward
        self.debug = debug
        
        if state_updater_cell == "gru":  # for TGN
            print(message_module.out_channels, state_dim)
            self.state_updater = GRUCell(message_module.out_channels, state_dim)
        elif state_updater_cell == "rnn":  # for JODIE & DyRep
            self.state_updater = RNNCell(message_module.out_channels, state_dim)
        else:
            raise ValueError(
                "Undefined state updater!!! State updater can be either 'gru' or 'rnn'."
            )
            
        if self.use_inventory:
            self.memory_dim = self.state_dim + self.num_prods
            if self.learn_att_direct:
                self.att_weights = Parameter(torch.ones(size=(self.num_prods, self.num_prods), requires_grad=True))
            else:
                self.prod_bilinear = Parameter(torch.ones(size=(self.state_dim, self.state_dim), requires_grad=True))
        else:
            self.memory_dim = self.state_dim
        self.register_buffer("memory", torch.empty(self.num_nodes, self.memory_dim))
        self.register_buffer("last_update", torch.empty(self.num_nodes, dtype=torch.long))
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
        self.state_updater.reset_parameters()
        self.reset_state()

    def reset_state(self):
        """Resets the memory to its initial state."""
        zeros(self.memory)
        zeros(self.last_update)
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

    def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp."""
        if self.training:
            memory, last_update, inv_loss = self._get_updated_memory(n_id)
        else:
            memory, last_update, inv_loss = self.memory[n_id], self.last_update[n_id], 0

        return memory, last_update, inv_loss
    
    def update_state(self, src: Tensor, dst: Tensor, prod: Tensor, t: Tensor, raw_msg: Tensor):
        """Updates the memory with newly encountered interactions
        :obj:`(src, dst, prod, t, raw_msg)`."""
        # update state for all nodes that appeared in this batch
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
    
    def get_prod_attention(self, prod_emb: Tensor = None):
        """
        Get attention weights between products.
        """
        if self.learn_att_direct:
            att_weights = self.att_weights
        else:
            if prod_emb is None:  # get product representations from memory
                prod_emb = self.memory[self.num_firms:, :self.state_dim]
            else:  # received product embeddings
                assert prod_emb.size == (self.num_prods, self.state_dim)
            att_weights = prod_emb @ (self.prod_bilinear @ prod_emb.T)  # has gradient issues
            # att_weights = prod_emb @ prod_emb.T  # this works
        return torch.nn.ReLU(inplace=False)(att_weights)
        
    def _update_memory(self, n_id: Tensor):
        """
        Update the stored memory and last update for nodes in n_id.
        """
        memory, last_update, inv_loss = self._get_updated_memory(n_id)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    def _get_updated_memory(self, n_id: Tensor, prod_emb: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Get current memory and last update for nodes in n_id, using their 
        current stored interactions.
        """
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)  # used to reindex n_id from 0
        state, inventory = self.memory[n_id, :self.state_dim], self.memory[n_id, self.state_dim:]
        
        # Compute messages to suppliers in interactions where n_id is supplier
        msg_s, t_s, idx_s, _, _ = self._compute_msg(
            n_id, self.msg_s_store, self.msg_s_module
        ) 
        if self.debug:
            print('msg_s', msg_s)
        
        # Compute messages to buyers in interactions where n_id is buyer
        msg_d, t_d, _, idx_d, _ = self._compute_msg(
            n_id, self.msg_d_store, self.msg_d_module
        ) 
        if self.debug:
            print('msg_d', msg_d)
        
        # Compute messages to products in interactions where n_id is product
        msg_p, t_p, _, _, idx_p = self._compute_msg(
            n_id, self.msg_p_store, self.msg_p_module
        )
        if self.debug:
            print('msg_p', msg_p)
        
        # Aggregate messages
        idx = torch.cat([idx_s, idx_d, idx_p], dim=0)
        msg = torch.cat([msg_s, msg_d, msg_p], dim=0)
        t = torch.cat([t_s, t_d, t_p], dim=0)
        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))  # one aggregated msg per node
        if self.debug:
            print('aggr', aggr)
        
        # Get local copy of updated state
        print("memory-module", aggr.shape, state.shape)
        state = self.state_updater(aggr, state)
            
        # Get local copy of updated `last_update`.
        dim_size = self.last_update.size(0)
        last_update = scatter(t, idx, 0, dim_size, reduce="max")[n_id]
        
        # Get local copy of updated inventory
        if self.use_inventory:
            total_consumed = self._compute_internal_consumption(n_id, self.msg_s_store, prod_emb)
            inv_loss = self._compute_inventory_loss(inventory, total_consumed)
            total_bought = self._compute_new_inputs(n_id, self.msg_d_store)
            if self.debug:
                print('Total consumed:', total_consumed)
                print('Total loss:', inv_loss)
                print('Total bought:', total_bought)
            inventory = inventory - total_consumed + total_bought
        else:
            inv_loss = 0

        print("state = ",state.shape, "inventory = ", inventory.shape)
        memory = torch.cat([state, inventory], dim=1)
        if self.debug:
            print('memory', memory)
        return memory, last_update, inv_loss
    
    def _compute_internal_consumption(
        self, n_id: Tensor, msg_store: TGNMessageStoreType, prod_emb: Tensor = None,
    ):
        """
        Compute the amount consumed by suppliers, for the interactions in msg_store.
        """
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, prod, t, raw_msg = list(zip(*data))  # unzip
        
        # Get total amount supplied per firm, product 
        src = torch.cat(src, dim=0)
        prod = torch.cat(prod, dim=0)
        # we assume first n IDs are firms, followed by products
        assert (prod >= self.num_firms).all()
        prod = prod - self.num_firms
        prod_onehot = F.one_hot(prod, num_classes=self.num_prods)
        amt = torch.cat(raw_msg, dim=0)[:, :1]  # assume first feature in raw_msg is amount
        amt = torch.clip(amt, min=1)  # amt shouldn't be smaller than 1
        prod_onehot = prod_onehot * amt  # scale each row by amt
        total_supplied = scatter(  # num_nodes x num_products
            prod_onehot, src, dim=0, dim_size=self.num_nodes, reduce="sum"
        )
        att_weights = self.get_prod_attention(prod_emb)
        total_consumed = total_supplied @ att_weights  # num_nodes x num_products
        return total_consumed[n_id]
    
    def _compute_inventory_loss(self, inventory: Tensor, consumption: Tensor):
        """
        Compute loss on inventory and consumption. Want to maximize consumption while minimizing
        wherever consumption is larger than inventory.
        """
        diff = inventory - consumption  # num_nodes x num_prod
        total_debt = -torch.sum(diff[diff < 0], dim=-1)  # sum of entries where consumption is greater than inventory
        total_consumption = torch.sum(consumption, dim=-1)
        loss = (self.debt_penalty * total_debt) - (self.consumption_reward * total_consumption)
        return loss.sum()
        
    def _compute_new_inputs(
        self, n_id: Tensor, msg_store: TGNMessageStoreType
    ):
        """
        Compute the amount received by buyers, for the interactions in msg_store.
        """
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, prod, t, raw_msg = list(zip(*data))  # unzip
        
        # Get total amount received per firm, product 
        dst = torch.cat(dst, dim=0)
        prod = torch.cat(prod, dim=0)
        # we assume first n IDs are firms, followed by products
        assert (prod >= self.num_firms).all()
        prod = prod - self.num_firms
        prod_onehot = F.one_hot(prod, num_classes=self.num_prods)
        amt = torch.cat(raw_msg, dim=0)[:, :1]  # assume first feature in raw_msg is amount
        amt = torch.clip(amt, min=1)  # amt shouldn't be smaller than 1
        prod_onehot = prod_onehot * amt  # scale each row by amt
        total_received = scatter(  # num_nodes x num_products
            prod_onehot, dst, dim=0, dim_size=self.num_nodes, reduce="sum"
        )
        return total_received[n_id]
    
        
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

        
class TGNMemory(torch.nn.Module):
    r"""The Temporal Graph Network (TGN) memory model from the
    `"Temporal Graph Networks for Deep Learning on Dynamic Graphs"
    <https://arxiv.org/abs/2006.10637>`_ paper.

    .. note::

        For an example of using TGN, see `examples/tgn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        tgn.py>`_.

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
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        message_module: Callable,
        aggregator_module: Callable,
        memory_updater_cell: str = "gru",
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)
        # self.gru = GRUCell(message_module.out_channels, memory_dim)
        if memory_updater_cell == "gru":  # for TGN
            self.memory_updater = GRUCell(message_module.out_channels, memory_dim)
        elif memory_updater_cell == "rnn":  # for JODIE & DyRep
            self.memory_updater = RNNCell(message_module.out_channels, memory_dim)
        else:
            raise ValueError(
                "Undefined memory updater!!! Memory updater can be either 'gru' or 'rnn'."
            )

        self.register_buffer("memory", torch.empty(num_nodes, memory_dim))
        last_update = torch.empty(self.num_nodes, dtype=torch.long)
        self.register_buffer("last_update", last_update)
        self.register_buffer("_assoc", torch.empty(num_nodes, dtype=torch.long))

        self.msg_s_store = {}
        self.msg_d_store = {}

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
        if hasattr(self.aggr_module, "reset_parameters"):
            self.aggr_module.reset_parameters()
        self.time_enc.reset_parameters()
        self.memory_updater.reset_parameters()
        self.reset_state()

    def reset_state(self):
        """Resets the memory to its initial state."""
        zeros(self.memory)
        zeros(self.last_update)
        self._reset_message_store()

    def detach(self):
        """Detaches the memory from gradient computation."""
        self.memory.detach_()

    def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp."""
        if self.training:
            memory, last_update = self._get_updated_memory(n_id)
        else:
            memory, last_update = self.memory[n_id], self.last_update[n_id]

        return memory, last_update

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor):
        """Updates the memory with newly encountered interactions
        :obj:`(src, dst, t, raw_msg)`."""
        n_id = torch.cat([src, dst]).unique()

        if self.training:
            self._update_memory(n_id)
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
        else:
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
            self._update_memory(n_id)

    def _reset_message_store(self):
        i = self.memory.new_empty((0,), device=self.device, dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
        # Message store format: (src, dst, t, msg)
        self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

    def _update_memory(self, n_id: Tensor):
        memory, last_update = self._get_updated_memory(n_id)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    def _get_updated_memory(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Compute messages (src -> dst).
        msg_s, t_s, src_s, dst_s = self._compute_msg(
            n_id, self.msg_s_store, self.msg_s_module
        )

        # Compute messages (dst -> src).
        msg_d, t_d, src_d, dst_d = self._compute_msg(
            n_id, self.msg_d_store, self.msg_d_module
        )

        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

        # Get local copy of updated memory.
        memory = self.memory_updater(aggr, self.memory[n_id])

        # Get local copy of updated `last_update`.
        dim_size = self.last_update.size(0)
        last_update = scatter(t, idx, 0, dim_size, reduce="max")[n_id]

        return memory, last_update

    def _update_msg_store(
        self,
        src: Tensor,
        dst: Tensor,
        t: Tensor,
        raw_msg: Tensor,
        msg_store: TGNMessageStoreType,
    ):
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

    def _compute_msg(
        self, n_id: Tensor, msg_store: TGNMessageStoreType, msg_module: Callable
    ):
        data = [msg_store[i] for i in n_id.tolist()]
        src, prod, dst, t, raw_msg = list(zip(*data))
        src = torch.cat(src, dim=0)
        dst = torch.cat(dst, dim=0)
        t = torch.cat(t, dim=0)
        raw_msg = torch.cat(raw_msg, dim=0)
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

        msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)

        return msg, t, src, dst

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            self._update_memory(torch.arange(self.num_nodes, device=self.memory.device))
            self._reset_message_store()
        super().train(mode)


class DyRepMemory(torch.nn.Module):
    r"""
    Based on intuitions from TGN Memory...
    Differences with the original TGN Memory:
        - can use source or destination embeddings in message generation
        - can use a RNN or GRU module as the memory updater

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
        memory_updater_type (str): specifies whether the memory updater is GRU or RNN
        use_src_emb_in_msg (bool): whether to use the source embeddings 
            in generation of messages
        use_dst_emb_in_msg (bool): whether to use the destination embeddings 
            in generation of messages
    """
    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int,
                 time_dim: int, message_module: Callable,
                 aggregator_module: Callable, memory_updater_type: str,
                 use_src_emb_in_msg: bool = False, use_dst_emb_in_msg: bool = False):
        super().__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)

        assert memory_updater_type in ['gru', 'rnn'], "Memor updater can be either `rnn` or `gru`."
        if memory_updater_type == 'gru':  # for TGN
            self.memory_updater = GRUCell(message_module.out_channels, memory_dim)
        elif memory_updater_type == 'rnn':  # for JODIE & DyRep
            self.memory_updater = RNNCell(message_module.out_channels, memory_dim)
        else:
            raise ValueError("Undefined memory updater!!! Memory updater can be either 'gru' or 'rnn'.")
        
        self.use_src_emb_in_msg = use_src_emb_in_msg
        self.use_dst_emb_in_msg = use_dst_emb_in_msg

        self.register_buffer('memory', torch.empty(num_nodes, memory_dim))
        last_update = torch.empty(self.num_nodes, dtype=torch.long)
        self.register_buffer('last_update', last_update)
        self.register_buffer('_assoc', torch.empty(num_nodes,
                                                   dtype=torch.long))

        self.msg_s_store = {}
        self.msg_d_store = {}

        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        return self.time_enc.lin.weight.device

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if hasattr(self.msg_s_module, 'reset_parameters'):
            self.msg_s_module.reset_parameters()
        if hasattr(self.msg_d_module, 'reset_parameters'):
            self.msg_d_module.reset_parameters()
        if hasattr(self.aggr_module, 'reset_parameters'):
            self.aggr_module.reset_parameters()
        self.time_enc.reset_parameters()
        self.memory_updater.reset_parameters()
        self.reset_state()

    def reset_state(self):
        """Resets the memory to its initial state."""
        zeros(self.memory)
        zeros(self.last_update)
        self._reset_message_store()

    def detach(self):
        """Detaches the memory from gradient computation."""
        self.memory.detach_()

    def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp."""
        if self.training:
            memory, last_update = self._get_updated_memory(n_id)
        else:
            memory, last_update = self.memory[n_id], self.last_update[n_id]

        return memory, last_update

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor, 
                     embeddings: Tensor = None, assoc: Tensor = None):
        """Updates the memory with newly encountered interactions
        :obj:`(src, dst, t, raw_msg)`."""
        n_id = torch.cat([src, dst]).unique()
        
        if self.training:
            self._update_memory(n_id, embeddings, assoc)
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
        else:
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
            self._update_memory(n_id, embeddings, assoc)

    def _reset_message_store(self):
        i = self.memory.new_empty((0, ), device=self.device, dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
        # Message store format: (src, dst, t, msg)
        self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

    def _update_memory(self, n_id: Tensor, embeddings: Tensor = None, assoc: Tensor = None):
        memory, last_update = self._get_updated_memory(n_id, embeddings, assoc)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    def _get_updated_memory(self, n_id: Tensor, embeddings: Tensor = None, assoc: Tensor = None) -> Tuple[Tensor, Tensor]:
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Compute messages (src -> dst).
        msg_s, t_s, src_s, dst_s = self._compute_msg(n_id, self.msg_s_store,
                                                     self.msg_s_module, embeddings, assoc)                                          

        # Compute messages (dst -> src).
        msg_d, t_d, src_d, dst_d = self._compute_msg(n_id, self.msg_d_store,
                                                     self.msg_d_module, embeddings, assoc)

        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

        # Get local copy of updated memory.
        memory = self.memory_updater(aggr, self.memory[n_id])

        # Get local copy of updated `last_update`.
        dim_size = self.last_update.size(0)
        last_update = scatter(t, idx, 0, dim_size, reduce='max')[n_id]

        return memory, last_update

    def _update_msg_store(self, src: Tensor, dst: Tensor, t: Tensor,
                          raw_msg: Tensor, msg_store: TGNMessageStoreType):
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

    def _compute_msg(self, n_id: Tensor, msg_store: TGNMessageStoreType, msg_module: Callable, 
                     embeddings: Tensor = None, assoc: Tensor = None):
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, t, raw_msg = list(zip(*data))
        src = torch.cat(src, dim=0)
        dst = torch.cat(dst, dim=0)
        t = torch.cat(t, dim=0)
        raw_msg = torch.cat(raw_msg, dim=0)
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

        # source nodes: retrieve embeddings
        source_memory = self.memory[src]
        if self.use_src_emb_in_msg and embeddings != None:
            if src.size(0) > 0:
                curr_src, curr_src_idx = [], []
                for s_idx, s in enumerate(src):
                    if s in n_id:
                        curr_src.append(s.item())
                        curr_src_idx.append(s_idx)

                source_memory[curr_src_idx] = embeddings[assoc[curr_src]]

        # destination nodes: retrieve embeddings
        destination_memory = self.memory[dst]
        if self.use_dst_emb_in_msg and embeddings != None:
            if dst.size(0) > 0:
                curr_dst, curr_dst_idx = [], []
                for d_idx, d in enumerate(dst):
                    if d in n_id:
                        curr_dst.append(d.item())
                        curr_dst_idx.append(d_idx)
                destination_memory[curr_dst_idx] = embeddings[assoc[curr_dst]]
            
        msg = msg_module(source_memory, destination_memory, raw_msg, t_enc)

        return msg, t, src, dst

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            self._update_memory(
                torch.arange(self.num_nodes, device=self.memory.device))
            self._reset_message_store()
        super().train(mode)