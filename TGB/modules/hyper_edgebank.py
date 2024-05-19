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

    def fit(self, src, dst, prod, msg):
        """
        Fit HyperEdgeBank on train triplets, storing which triplets were seen and
        their counts and their mean, median edge weights.
        """
        train_idx = self.convert_triplet_to_index(src, dst, prod)
        train_idx, ordering = torch.sort(train_idx)
        msg = msg[ordering]
        # get count
        unique_idx, counts = torch.unique_consecutive(train_idx, return_counts=True)

        # get average weight
        split_msg = torch.split(msg, counts.tolist())
        weights_mean = torch.tensor([segment.float().mean() for segment in split_msg])
        weights_median = torch.tensor([segment.float().median() for segment in split_msg])
        assert weights_mean.shape[0] == unique_idx.shape[0]

        print(f'Fit on {len(src)} edges; found {len(unique_idx)} unique')
        self.seen_idx = unique_idx
        self.idx_counts = dict(zip(list(unique_idx.detach().numpy()), list(counts.detach().numpy())))
        self.idx_weights_mean = dict(zip(list(unique_idx.detach().numpy()), list(weights_mean.detach().numpy())))
        self.idx_weights_median = dict(zip(list(unique_idx.detach().numpy()), list(weights_median.detach().numpy())))
        self.fitted = True
        
    def predict(self, src, dst, prod, use_counts=True, use_median=True):
        """
        Predict future edges based on stored train data.
        If use_counts is True, return the number of times each edge was seen in train.
        If use_counts is False, return 1 if the edge appeared in train; otherwise 0.
        If use_median is True, return the median of edge weights seen in train. 
        If use_median is False, return the mean of edge weights seen in train. 
        """
        if not self.fitted:
            raise Exception("Cannot predict until fitted on train data.")
        test_idx = self.convert_triplet_to_index(src, dst, prod)
        if use_counts:
            scores = torch.Tensor([self.idx_counts.get(int(idx.detach()), 0) for idx in test_idx])
        else:
            scores = torch.isin(test_idx, self.seen_idx).to(int)
        if use_median:
            weights = torch.Tensor([self.idx_weights_median.get(int(idx.detach()), 0) for idx in test_idx])
        else:
            weights = torch.Tensor([self.idx_weights_mean.get(int(idx.detach()), 0) for idx in test_idx])
        return scores, weights
