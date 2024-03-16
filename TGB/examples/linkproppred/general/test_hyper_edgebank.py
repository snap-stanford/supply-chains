import wandb
import math
import timeit
from tqdm import tqdm
import json
import argparse 
import sys

import os
import os.path as osp
from pathlib import Path
import numpy as np

import torch
from torch_geometric.loader import TemporalDataLoader

# internal imports
from tgb.linkproppred.logger import TensorboardLogger
from tgb.utils.utils import *
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset, PyGLinkPropPredDatasetHyper

from modules.hyper_edgebank import HyperEdgeBankPredictor

    
# ===========================================
# == Functions to run complete experiments
# ===========================================
# Global variables
MODEL_NAME = 'TGNPL'
NUM_NEIGHBORS = 10
#PATH_TO_DATASETS = f'/lfs/turing1/0/{os.getlogin()}/supply-chains/TGB/tgb/datasets/'
TGB_DIRECTORY = "/".join(str(__file__).split("/")[:-4])
PATH_TO_DATASETS = os.path.join(TGB_DIRECTORY, "tgb/datasets/")

def test_edgebank(loader, neg_sampler, split_mode, evaluator, metric, edgebank, use_counts=True,
                  use_prev_sampling=False, ns_samples=6):
    """
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges
    Code is similar to test() in tgnpl.py

    Parameters:
        loader: an object containing positive attributes of the positive edges of the evaluation set
        neg_sampler: an object that gives the negative edges corresponding to each positive edge
        split_mode: specifies whether it is the 'validation' or 'test' set to correctly load the negatives
        edgebank: the fitted HyperEdgeBankPredictor object
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


def test_edgebank_on_dataset(args):
    """
    Run a full experiment: initialize data, test HyperEdgeBank (binary and count) on validation and test sets.
    """
    with open(os.path.join(PATH_TO_DATASETS, f"{args.dataset.replace('-', '_')}/{args.dataset}_meta.json"), "r") as f:
        metadata = json.load(f)
    # set global data variables
    num_nodes = len(metadata['id2entity'])  
    NUM_FIRMS = metadata['product_threshold']
    NUM_PRODUCTS = num_nodes - NUM_FIRMS  
    
    dataset = PyGLinkPropPredDatasetHyper(name=args.dataset, root='datasets')
    data = dataset.get_TemporalData()
    neg_sampler = dataset.negative_sampler
    metric = dataset.eval_metric
    evaluator = Evaluator(name=args.dataset)
    
    train_data = data[dataset.train_mask]
    val_data = data[dataset.val_mask]
    test_data = data[dataset.test_mask]
    BATCH_SIZE = 10000  # batch size doesn't matter for hyperedgebank
    train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)
    
    # initialize and fit HyperEdgeBank
    print(f'HyperEdgeBank results on {args.dataset}...')
    edgebank = HyperEdgeBankPredictor(NUM_FIRMS, NUM_PRODUCTS, consecutive=True)
    edgebank.fit(train_data.src, train_data.dst, train_data.prod)
    
    dataset.load_val_ns()  # load validation negative samples
    val_bin = test_edgebank(val_loader, neg_sampler, "val", evaluator, metric, edgebank, use_counts=False,
                            use_prev_sampling=args.use_prev_sampling)  # binary on validation set
    val_cnt = test_edgebank(val_loader, neg_sampler, "val", evaluator, metric, edgebank, use_counts=True,
                            use_prev_sampling=args.use_prev_sampling)  # count on validation set
    print(f'Validation set MRRs: binary = {val_bin:0.4f}, count = {val_cnt:0.4f}')
    
    dataset.load_test_ns()  # load test negative samples
    test_bin = test_edgebank(test_loader, neg_sampler, "test", evaluator, metric, edgebank, use_counts=False,
                             use_prev_sampling=args.use_prev_sampling)  # binary on test set
    test_cnt = test_edgebank(test_loader, neg_sampler, "test", evaluator, metric, edgebank, use_counts=True,
                             use_prev_sampling=args.use_prev_sampling)  # binary on test set
    print(f'Test set MRRs: binary = {test_bin:0.4f}, count = {test_cnt:0.4f}')  
    return val_bin, val_cnt, test_bin, test_cnt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset name', default='tgbl-hypergraph')
    parser.add_argument('--use_prev_sampling', action='store_true')
    args = parser.parse_args()
    
    test_edgebank_on_dataset(args)
