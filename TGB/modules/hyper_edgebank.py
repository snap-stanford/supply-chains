"""
EdgeBank is a simple strong baseline for dynamic link prediction
it predicts the existence of edges based on their history of occurrence
"""

import torch
from tqdm import tqdm
import numpy as np

class HyperEdgeBankPredictor(object):
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
        src = idx // (self.num_firms * self.num_prods)
        dst = (idx % (self.num_firms * self.num_prods)) // self.num_prods
        prod = idx % self.num_prods
        if self.consecutive:
            prod = prod + self.num_firms
        return src, dst, prod

    def fit(self, src, dst, prod):
        train_idx = self.convert_triplet_to_index(src, dst, prod)
        train_idx, _ = torch.sort(train_idx)
        unique_idx, counts = torch.unique_consecutive(train_idx, return_counts=True)
        print(f'Fit on {len(src)} edges; found {len(unique_idx)} unique')
        self.seen_idx = unique_idx
        self.idx_counts = dict(zip(list(unique_idx.detach().numpy()), list(counts.detach().numpy())))
        self.fitted = True
        
    def predict(self, src, dst, prod, use_counts=True):
        if not self.fitted:
            raise Exception("Cannot predict until fitted on train data.")
        test_idx = self.convert_triplet_to_index(src, dst, prod)
        if use_counts:
            # return the number of times the edge was seen in train
            scores = torch.Tensor([self.idx_counts.get(int(idx.detach()), 0) for idx in test_idx])
        else:
            # return 1 if edge was seen in train, 0 otherwise
            scores = torch.isin(test_idx, self.seen_idx).to(int)
        return scores
    

def test_edgebank(loader, neg_sampler, split_mode, evaluator, metric, edgebank, use_counts=True,
                  use_prev_sampling=False, ns_samples=6):
    r"""
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges
    Code is similar to test() in tgnpl.py

    Parameters:
        loader: an object containing positive attributes of the positive edges of the evaluation set
        neg_sampler: an object that gives the negative edges corresponding to each positive edge
        split_mode: specifies whether it is the 'validation' or 'test' set to correctly load the negatives
        edgebank: the fitted HyperEdgeBankPredictor objective
        use_counts: whether to use train count or train existence in predict()
    Returns:
        perf_metric: the result of the performance evaluaiton
    """
    perf_list = []  # mean metric per batch
    batch_size = []
    for pos_batch in tqdm(loader):
        
        pos_src, pos_prod, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.prod,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )
        bs = len(pos_src)
        
        if use_prev_sampling == True:
            #using the negative sampling procedure prior to Oct 24 (no loose negatives)
            neg_batch_list = neg_sampler.query_batch(pos_src, pos_prod, pos_dst, pos_t, split_mode=split_mode)
            assert len(neg_batch_list) == bs
            neg_batch_list = torch.Tensor(neg_batch_list)
            ns_samples = neg_batch_list.size(1) // 3  # num negative samples per src/prod/dst
            neg_src = neg_batch_list[:, :ns_samples]   # we assume neg batch is ordered by neg_src, neg_prod, neg_dst
            neg_prod = neg_batch_list[:, ns_samples:(2*ns_samples)]  
            neg_dst = neg_batch_list[:, (2*ns_samples):] 
            
            num_samples = (3*ns_samples)+1  # total num samples per data point
            batch_src = pos_src.reshape(bs, 1).repeat(1, num_samples)  # [[src1, src1, ...], [src2, src2, ...]]
            batch_src[:, 1:ns_samples+1] = neg_src  # replace pos_src with negatives
            batch_prod = pos_prod.reshape(bs, 1).repeat(1, num_samples)
            batch_prod[:, ns_samples+1:(2*ns_samples)+1] = neg_prod  # replace pos_prod with negatives
            batch_dst = pos_dst.reshape(bs, 1).repeat(1, num_samples)
            batch_dst[:, (2*ns_samples)+1:] = neg_dst  # replace pos_dst with negatives
        else:
            #using the current negative sampling for hypergraph
            neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_prod, pos_t, split_mode=split_mode)
            assert len(neg_batch_list) == bs
            neg_batch_list = torch.Tensor(np.array(neg_batch_list)).int()
            num_samples = neg_batch_list.size(1) + 1
            neg_src = neg_batch_list[:,:,0]
            neg_dst = neg_batch_list[:,:,1]
            neg_prod = neg_batch_list[:,:,2]
            batch_src = torch.cat((torch.unsqueeze(pos_src,-1), neg_src), dim = -1)
            batch_dst = torch.cat((torch.unsqueeze(pos_dst,-1), neg_dst), dim = -1)
            batch_prod = torch.cat((torch.unsqueeze(pos_prod,-1), neg_prod), dim = -1)
                 
        y_pred = edgebank.predict(batch_src.flatten(), batch_dst.flatten(), batch_prod.flatten(),
                                  use_counts=use_counts)
        y_pred = y_pred.reshape(bs, 1+(3*ns_samples))
        input_dict = {
            "y_pred_pos": y_pred[:, :1],
            "y_pred_neg": y_pred[:, 1:],
            "eval_metric": [metric]
        }
        perf_list.append(evaluator.eval(input_dict)[metric])
        batch_size.append(len(pos_src))
        
    num = (torch.tensor(perf_list) * torch.tensor(batch_size)).sum()
    denom = torch.tensor(batch_size).sum()
    perf_metrics = float(num/denom)
    return perf_metrics