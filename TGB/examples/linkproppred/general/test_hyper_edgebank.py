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

# Global variables
TGB_DIRECTORY = "/".join(str(__file__).split("/")[:-4])
PATH_TO_DATASETS = os.path.join(TGB_DIRECTORY, "tgb/datasets/")

def test_edgebank(loader, neg_sampler, split_mode, evaluator, metric, edgebank, use_counts=True, use_median=True, ns_samples=6):
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
        use_median: whether to use median over mean of edge weights in predict()
    Returns:
        perf_metric: the result of the performance evaluaiton
    """
    perf_list = []  # mean metric per batch
    perf_list_rmse = [] # mean rmse per batch 
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
        
        # link prediction performance
        y_pred, _ = edgebank.predict(batch_src.flatten(), batch_dst.flatten(), batch_prod.flatten(),
                                  use_counts=use_counts, use_median=use_median)
        y_pred = y_pred.reshape(bs, 1+(3*ns_samples))
        input_dict = {
            "y_pred_pos": y_pred[:, :1],
            "y_pred_neg": y_pred[:, 1:],
            "eval_metric": [metric]
        }
        perf_list.append(evaluator.eval(input_dict)[metric])
    
        # amount prediction performance
        _, y_amt_pred = edgebank.predict(pos_src.flatten(), pos_dst.flatten(), pos_prod.flatten(),
                                  use_counts=use_counts, use_median=use_median)
        y_amt_pred = y_amt_pred.reshape(bs, 1)
        criterion = torch.nn.MSELoss()
        rmse = torch.sqrt(criterion(y_amt_pred, pos_msg))
        perf_list_rmse.append(rmse)

        batch_size.append(len(pos_src))
        
    num = (torch.tensor(perf_list) * torch.tensor(batch_size)).sum()
    num_rmse = (torch.tensor(perf_list_rmse) * torch.tensor(batch_size)).sum()
    denom = torch.tensor(batch_size).sum()
    perf_metrics, perf_rmse = float(num/denom), float(num_rmse/denom)
    return perf_metrics, perf_rmse


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

    # apply log scaling and standard scaling to edge features    
    for d in range(data.msg.shape[1]):
        vals = data.msg[:, d]
        assert (vals >= 0).all()  # if we are logging, all values need to be positive
        min_val = torch.min(vals[vals > 0])  # minimum value greater than 0
        vals = torch.clip(vals, min_val, None)  # clip so we don't take log of 0
        vals = torch.log(vals)  # log scale
        mean = torch.mean(vals)
        std = torch.std(vals)
        if d == 0:  # save so that we can use to scale inventory module caps in get_y_pred 
            AMT_MEAN = mean
            AMT_STD = std
        vals = (vals - mean) / std  # standard scaling
        data.msg[:, d] = vals
    print(f"after log scaling, mean={data.msg.mean()}, std={data.msg.std()}") 
    
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
    edgebank.fit(train_data.src, train_data.dst, train_data.prod, train_data.msg)
    
    dataset.load_val_ns()  # load validation negative samples
    val_bin, val_rmse_mean = test_edgebank(val_loader, neg_sampler, "val", evaluator, metric, edgebank, use_counts=False, use_median=False)  # binary on validation set
    val_cnt, val_rmse_median = test_edgebank(val_loader, neg_sampler, "val", evaluator, metric, edgebank, use_counts=True, use_median=True)  # count on validation set
    print(f'Validation set MRRs: binary = {val_bin:0.4f}, count = {val_cnt:0.4f}; Val set RMSE: {val_rmse_mean: 0.4f}')
    print(f'Validation set RMSE: mean = {val_rmse_mean: 0.4f}, median = {val_rmse_median: 0.4f}')  

    dataset.load_test_ns()  # load test negative samples
    test_bin, test_rmse_mean = test_edgebank(test_loader, neg_sampler, "test", evaluator, metric, edgebank, use_counts=False, use_median=False)  # binary on test set
    test_cnt, test_rmse_median = test_edgebank(test_loader, neg_sampler, "test", evaluator, metric, edgebank, use_counts=True, use_median=True)  # binary on test set
    print(f'Test set MRRs: binary = {test_bin:0.4f}, count = {test_cnt:0.4f}')
    print(f'Test set RMSE: mean = {test_rmse_mean: 0.4f}, median = {test_rmse_median: 0.4f}')  
    return val_bin, val_cnt, val_rmse_mean, val_rmse_median, test_bin, test_cnt, test_rmse_mean, test_rmse_median


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset name', default='tgbl-hypergraph')
    args = parser.parse_args()
    
    test_edgebank_on_dataset(args)

