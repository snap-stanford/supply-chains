"""
file for testing the supply-chains hypergraph data

"""

import math
import timeit

import os
import os.path as osp
from pathlib import Path
import numpy as np

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader

from torch_geometric.nn import TransformerConv

# internal imports
from tgb.utils.utils import get_args, set_random_seed, save_results
from tgb.linkproppred.evaluate import Evaluator
from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import IdentityMessage
from modules.msg_agg import LastAggregator
from modules.neighbor_loader import LastNeighborLoader
from modules.memory_module import TGNMemory
from modules.early_stopping import  EarlyStopMonitor
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset, PyGLinkPropPredDatasetHyper

import json
import argparse 
import sys

def get_tgn_args():
    parser = argparse.ArgumentParser('*** TGB ***')
    parser.add_argument('--dataset', type=str, help='Dataset name', default='tgbl-hypergraph')
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--bs', type=int, help='Batch size', default=200)
    parser.add_argument('--k_value', type=int, help='k_value for computing ranking metrics', default=10)
    parser.add_argument('--num_epoch', type=int, help='Number of epochs', default=50)
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--mem_dim', type=int, help='Memory dimension', default=100)
    parser.add_argument('--time_dim', type=int, help='Time dimension', default=100)
    parser.add_argument('--emb_dim', type=int, help='Embedding dimension', default=100)
    parser.add_argument('--tolerance', type=float, help='Early stopper tolerance', default=1e-6)
    parser.add_argument('--patience', type=float, help='Early stopper patience', default=10)
    parser.add_argument('--num_run', type=int, help='Number of iteration runs', default=1)
    parser.add_argument('--wandb', type=bool, help='Wandb support', default=False)

    try:
        args = parser.parse_args()
        defaults = parser.parse_args([])
    
    except:
        parser.print_help()
        sys.exit(0)
    return args, defaults, sys.argv

def compare_args(args, defaults): 
    #compares deviations of the parsed arguments from the defaults (for labeling the checkpoints
    #& results folder succinctly) 
    args_dict = vars(args)
    defaults_dict = vars(defaults)
    return {key: value for key,value in args_dict.items() if (
        key not in defaults_dict or defaults_dict[key] != args_dict[key])}
    
# ==========
# ==========
# ==========

# Start...
start_overall = timeit.default_timer()

# ========== set parameters...
args, defaults, _ = get_tgn_args()
labeling_args = compare_args(args, defaults)
MODEL_NAME = 'TGN'

FOLDER_NAME_HDR = f"model={MODEL_NAME}"
for arg_name in sorted(list(labeling_args.keys())):
    arg_value = labeling_args[arg_name]
    FOLDER_NAME_HDR += f"_{arg_name}={arg_value}"

DATA = args.dataset
LR = args.lr
BATCH_SIZE = args.bs
K_VALUE = args.k_value  
NUM_EPOCH = args.num_epoch
SEED = args.seed
MEM_DIM = args.mem_dim
TIME_DIM = args.time_dim
EMB_DIM = args.emb_dim
TOLERANCE = args.tolerance
PATIENCE = args.patience
NUM_RUNS = args.num_run
NUM_NEIGHBORS = 10

#preprocessing for the bipartite graph 
DATA_FOLDER_NAME = DATA.replace("-","_")
metadata_path = f"./tgb/datasets/{DATA_FOLDER_NAME}/{DATA}_meta.json"
with open(metadata_path,"r") as file:
    metadata_supplychains = json.load(file) 
    product_min_idx = metadata_supplychains["product_threshold"]

print(product_min_idx)

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data loading
dataset = PyGLinkPropPredDatasetHyper(name=DATA, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
data = dataset.get_TemporalData()
data = data.to(device)
metric = dataset.eval_metric

train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]

train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

for batch in train_loader:
    batch = batch.to(device)
    pos_src, pos_prod, pos_dst, t, msg = batch.src, batch.prod, batch.dst, batch.t, batch.msg
    print("Source", pos_src)
    print("Product", pos_prod)
    print("Destination", pos_dst)
    print("Time Stamp", t)
    print("Message", msg)
    break

dataset.load_val_ns()
neg_sampler = dataset.negative_sampler
for pos_batch in val_loader:
    pos_src, pos_prod, pos_dst, pos_t, pos_msg = (
        pos_batch.src, pos_batch.prod, pos_batch.dst, pos_batch.t, pos_batch.msg,)

    neg_batch_list = neg_sampler.query_batch(pos_src, pos_prod, pos_dst, pos_t, split_mode="val")

    for idx, neg_batch in enumerate(neg_batch_list):
        p_src, p_prod, p_dst = (p.cpu().numpy()[idx] for p in [pos_src, pos_prod, pos_dst])
        ns_samples = len(neg_batch) // 3
        src = torch.tensor([p_src] + neg_batch[:ns_samples] + [p_src for _ in range(ns_samples * 2)], device = device)
        prod = torch.tensor([p_prod] + [p_prod for _ in range(ns_samples)] + neg_batch[ns_samples:ns_samples * 2] + [p_prod for _ in range(ns_samples)], device = device)
        dest = torch.tensor([p_dst] + [p_dst for _ in range(ns_samples * 2)] + neg_batch[ns_samples * 2:], device = device)

        print(src)
        print(prod)
        print(dest)
        print(idx, [(s.item(),p.item(),d.item()) for s, p, d in zip(src, prod, dest)], len([(s,p,d) for s, p, d in zip(src, prod, dest)]))
        break
    break



