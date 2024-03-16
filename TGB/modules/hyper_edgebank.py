import torch
from tqdm import tqdm
import numpy as np

class HyperEdgeBankPredictor(object):
    """
    EdgeBank is a simple strong baseline for dynamic link prediction, which
    predicts future edges based on their count / existence in the train data.
    """
    def __init__(self,
        num_firms,
        num_prods,
        consecutive
    ):
        self.num_firms = num_firms
        self.num_prods = num_prods
        self.consecutive = consecutive
        self.fitted = False
        
    def convert_triplet_to_index(self, src, dst, prod):
        """
        Convert (source, destination, product) triplets to single index.
        """
        if self.consecutive:
            prod = prod - self.num_firms
        assert (src >= 0).all() and (src < self.num_firms).all()
        assert (dst >= 0).all() and (dst < self.num_firms).all()
        assert (prod >= 0).all() and (prod < self.num_prods).all()
        idx = src * (self.num_firms * self.num_prods)
        idx += dst * self.num_prods
        idx += prod
        return idx
    
    def convert_index_to_triplet(self, idx):
        """
        Convert single index back to (source, destination, product) triplets.
        """
        src = idx // (self.num_firms * self.num_prods)
        dst = (idx % (self.num_firms * self.num_prods)) // self.num_prods
        prod = idx % self.num_prods
        if self.consecutive:
            prod = prod + self.num_firms
        return src, dst, prod

    def fit(self, src, dst, prod):
        """
        Fit HyperEdgeBank on train triplets, storing which triplets were seen and
        their counts.
        """
        train_idx = self.convert_triplet_to_index(src, dst, prod)
        train_idx, _ = torch.sort(train_idx)
        unique_idx, counts = torch.unique_consecutive(train_idx, return_counts=True)
        print(f'Fit on {len(src)} edges; found {len(unique_idx)} unique')
        self.seen_idx = unique_idx
        self.idx_counts = dict(zip(list(unique_idx.detach().numpy()), list(counts.detach().numpy())))
        self.fitted = True
        
    def predict(self, src, dst, prod, use_counts=True):
        """
        Predict future edges based on stored train data.
        If use_counts is True, return the number of times each edge was seen in train.
        If use_counts is False, return 1 if the edge appeared in train; otherwise 0.
        """
        if not self.fitted:
            raise Exception("Cannot predict until fitted on train data.")
        test_idx = self.convert_triplet_to_index(src, dst, prod)
        if use_counts:
            scores = torch.Tensor([self.idx_counts.get(int(idx.detach()), 0) for idx in test_idx])
        else:
            scores = torch.isin(test_idx, self.seen_idx).to(int)
        return scores