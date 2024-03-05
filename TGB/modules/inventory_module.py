import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.utils import scatter


class TGNPLInventory(torch.nn.Module):
    def __init__(
        self,
        num_firms : int,
        num_prods : int,
        debt_penalty : float = 10.0,
        consumption_reward : float = 1.0,
        device = None,
        learn_att_direct: bool = False,
        emb_dim : int = 0,
    ):
        super().__init__()
        if not learn_att_direct:
            assert emb_dim > 0
        self.num_firms = num_firms
        self.num_prods = num_prods
        self.debt_penalty = debt_penalty
        self.consumption_reward = consumption_reward
        self.emb_dim = emb_dim
        self.learn_att_direct = learn_att_direct
        self.device = device
            
        self.inventory = torch.ones(size=(self.num_firms, self.num_prods), requires_grad=False, device=device)
        if self.learn_att_direct:
            # learn attention weights directly 
            self.att_weights = Parameter(torch.rand(size=(self.num_prods, self.num_prods), requires_grad=True, device=device))
        else:
            # learn attention weights using product embeddings
            self.prod_bilinear = Parameter(torch.rand(size=(self.emb_dim, self.emb_dim), requires_grad=True, device=device))
                
    
    def forward(self, src, dst, prod, raw_msg, prod_emb: Tensor = None):
        """
        Main function: updates inventory based on new transactions and returns inventory loss.
        """
        total_supplied = self._compute_totals_per_firm_and_product(src, prod, raw_msg)
        att_weights = self._get_prod_attention(prod_emb)
        total_consumed = total_supplied @ att_weights  # num_firms x num_products
        assert self.inventory.shape == total_consumed.shape
        inv_loss, debt_loss, consump_rwd_loss = self._compute_inventory_loss(self.inventory, total_consumed)
        total_bought = self._compute_totals_per_firm_and_product(dst, prod, raw_msg)
        self.inventory = torch.clip(self.inventory - total_consumed + total_bought, 0, None)
        return inv_loss / len(src), debt_loss / len(src), consump_rwd_loss / len(src)  # loss is scaled by batch size in tgnpl.py
    
    def detach(self):
        """
        Detaches inventory from gradient computation.
        """
        self.inventory.detach_()
        
    def _compute_totals_per_firm_and_product(self, firm_ids, prod_ids, raw_msg):
        """
        Take vector of firm IDs and product IDs and return matrix of (n_firms x n_prod),
        indicating totals per pair. Used to get total supplied and total bought.
        """
        assert (prod_ids >= self.num_firms).all()
        prod_ids = prod_ids - self.num_firms
        prod_onehot = F.one_hot(prod_ids, num_classes=self.num_prods)
        amt = raw_msg[:, :1]  # assume first feature in raw_msg is amount
        amt = torch.clip(amt, min=1)  # amt shouldn't be smaller than 1
        prod_onehot = prod_onehot * amt  # scale each row by amt
        totals = scatter(  # num_firms x num_products
            prod_onehot, firm_ids, dim=0, dim_size=self.num_firms, reduce="sum"
        )
        return totals
    
    def _get_prod_attention(self, prod_emb: Tensor = None):
        """
        Get attention weights between products. Returns a Tensor, (n_prod x n_prod).
        """
        if self.learn_att_direct:
            att_weights = self.att_weights
        else:
            assert prod_emb is not None
            assert prod_emb.shape == (self.num_prods, self.emb_dim), prod_emb.shape
            att_weights = prod_emb @ (self.prod_bilinear @ prod_emb.T)
        return torch.nn.ReLU(inplace=False)(att_weights)  # weights must be non-negative
    
    def _compute_inventory_loss(self, inventory: Tensor, consumption: Tensor):
        """
        Compute loss on inventory and consumption. Want to maximize consumption while minimizing
        wherever consumption is larger than inventory.
        """
        diff = torch.maximum(consumption - inventory, torch.zeros_like(inventory, device=self.device))  # entrywise max
        total_debt = torch.sum(diff, dim=-1)  # n_firms, sum of entries where consumption is greater than inventory
        total_consumption = torch.sum(consumption, dim=-1)  # n_firms
        debt_loss = self.debt_penalty * total_debt
        consump_rwd_loss = self.consumption_reward * total_consumption
        loss = debt_loss - consump_rwd_loss
        return loss.sum(), debt_loss.sum(), consump_rwd_loss.sum()