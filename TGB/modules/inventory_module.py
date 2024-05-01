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
        debt_penalty : float = 5.,
        consumption_reward : float = 4.,
        adjust_penalty: float = 1.,
        device = None,
        learn_att_direct: bool = False,
        init_weights = None,
        emb_dim : int = 0,
        seed: int = 0,
    ):
        super().__init__()
        if not learn_att_direct:
            assert emb_dim > 0
        self.num_firms = num_firms
        self.num_prods = num_prods
        self.debt_penalty = debt_penalty
        self.consumption_reward = consumption_reward
        self.adjust_penalty = adjust_penalty
        self.emb_dim = emb_dim
        self.learn_att_direct = learn_att_direct

        self.device = device
        self.seed = seed
        torch.manual_seed(seed)
            
        self.reset()
        if init_weights is None:
            init_weights = torch.rand(size=(self.num_prods, self.num_prods), requires_grad=True, device=device)
        else:  # initial weights provided, eg, from correlations
            assert init_weights.shape == (self.num_prods, self.num_prods)
            init_weights = torch.Tensor(init_weights).to(device)
        
        if self.learn_att_direct:  # learn attention weights directly  
            self.att_weights = Parameter(init_weights)
        else:  # learn attention weights using product embeddings, treat weights as adjustments
            self.prod_bilinear = Parameter(torch.eye(self.emb_dim, requires_grad=True, device=device))
            self.adjustments = Parameter(init_weights * 0.01)
                
    
    def forward(self, src: Tensor, dst: Tensor, prod: Tensor, t: Tensor, amt: Tensor, prod_emb: Tensor = None):
        """
        Main function: updates inventory based on new transactions and returns inventory loss.
        """
        unique_timesteps = sorted(torch.unique(t))
        inv_loss = 0
        debt_loss = 0
        consump_rwd = 0
        att_weights = self._get_prod_attention(prod_emb)  # num_products x num_products
        for ts in unique_timesteps:
            if ts != self.curr_t:  # we've reached a new timestep
                assert ts > self.curr_t, f'ts={ts}, curr_t={self.curr_t}'  # we should only move forward in time
                self.inventory = self.inventory + self.received  # add received products from previous t to inventory
                self.reset_received()  # set received products to 0
                self.curr_t = ts  # update current timestep
            
            in_ts = t == ts
            total_supplied_t = self._compute_totals_per_firm_and_product(src[in_ts], prod[in_ts], amt[in_ts])
            total_consumed_t = total_supplied_t @ att_weights
            inv_loss_t, debt_loss_t, consump_rwd_t = self._compute_inventory_loss(self.inventory, total_consumed_t)
            inv_loss += inv_loss_t
            debt_loss += debt_loss_t
            consump_rwd += consump_rwd_t
            self.inventory = torch.clip(self.inventory - total_consumed_t, 0, None)  # subtract consumption from inventory
            
            total_bought_t = self._compute_totals_per_firm_and_product(dst[in_ts], prod[in_ts], amt[in_ts])
            self.received += total_bought_t  # add received products at t

        return inv_loss / len(src), debt_loss / len(src), consump_rwd / len(src)  # loss is scaled by batch size in tgnpl.py
    
    
    def link_pred_penalties(self, src: Tensor, prod: Tensor, prod_emb: Tensor = None):
        """
        Penalize transactions that would go into inventory debt.
        """
        att_weights = self._get_prod_attention(prod_emb)  # num_products x num_products
        prod_ids = prod - self.num_firms
        parts_per_prod = att_weights[prod_ids]  # parts needed to make one unit of product
        assert parts_per_prod.shape == (len(prod), self.num_prods)
        inventories = self.inventory[src]
        assert inventories.shape == (len(src), self.num_prods)
        diff = parts_per_prod - inventories          
        penalties = torch.maximum(diff, torch.zeros_like(diff, device=self.device))  # wherever necessary consumption > inventory
        return penalties.sum(axis=1)
        
        
    def detach(self):
        """
        Detaches inventory from gradient computation.
        """
        self.inventory.detach_()
        
    def reset(self):
        """
        Reset all time-varying parameters.
        """
        self.reset_inventory()
        self.reset_received()
        self.curr_t = -1
        
    def reset_inventory(self):
        """
        Reset inventory for all firms.
        """
        self.inventory = torch.ones(size=(self.num_firms, self.num_prods), requires_grad=False, device=self.device)
        
    def reset_received(self):
        """
        Reset received for all firms.
        """
        self.received = torch.zeros(size=(self.num_firms, self.num_prods), requires_grad=False, device=self.device)
        
    def _compute_totals_per_firm_and_product(self, firm_ids, prod_ids, amt):
        """
        Take vector of firm IDs and product IDs and return matrix of (n_firms x n_prod),
        indicating totals per pair. Used to get total supplied and total bought.
        """
        assert (prod_ids >= self.num_firms).all()
        prod_ids = prod_ids - self.num_firms
        prod_onehot = F.one_hot(prod_ids, num_classes=self.num_prods)
        neg_amt = (amt < 0).sum() / len(amt)
        if neg_amt > 0.01:
            print(f'Warning: {neg_amt:0.3f} of amt in batch is negative')
        amt = torch.clip(amt, min=0)  # amt shouldn't be smaller than 0
        prod_onehot = prod_onehot * amt.reshape(-1, 1)  # scale each row by amt
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
            att_weights = (prod_emb @ (self.prod_bilinear @ prod_emb.T)) + self.adjustments
        att_weights = att_weights * (1-torch.eye(self.num_prods, device=self.device))  # set diagonal to 0
        return torch.nn.ReLU(inplace=False)(att_weights)  # weights must be non-negative
    
    def _compute_inventory_loss(self, inventory: Tensor, consumption: Tensor):
        """
        Compute loss on inventory and consumption. Want to maximize consumption while minimizing
        wherever consumption is larger than inventory. Also penalize adjustments if they are being used.
        """
        diff = torch.maximum(consumption - inventory, torch.zeros_like(inventory, device=self.device))  # entrywise max
        total_debt = torch.sum(diff, dim=-1)  # n_firms, sum of entries where consumption is greater than inventory
        debt_loss = (self.debt_penalty * total_debt).mean()  # mean over firms
        total_consumption = torch.sum(consumption, dim=-1)  # n_firms
        consump_rwd = (self.consumption_reward * total_consumption).mean()  # mean over firms
        loss = debt_loss - consump_rwd
        if not self.learn_att_direct:
            loss += self.adjust_penalty * torch.sqrt(torch.sum(self.adjustments ** 2))
        return loss, debt_loss, consump_rwd