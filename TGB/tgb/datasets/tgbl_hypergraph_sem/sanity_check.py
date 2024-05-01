# check basic properties
import pandas as pd
sem = pd.read_csv('sem.csv')
sem_trans = pd.read_csv('sem_transactions.csv')

print("SEM and SEM transactions shapes are", sem.shape, sem_trans.shape)
print("Print neg amount_sum if any (SEM)", sem.amount_sum[sem.amount_sum < 0])
print("Print neg amount_sum if any (SEM_trans)", sem_trans.total_amount[sem_trans.total_amount < 0])

# Check chronological batch order
import torch
import numpy as np
from tgb.utils.utils import *
from tgb.linkproppred.evaluate import Evaluator
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import TGNPLMessage
from modules.msg_agg import MeanAggregator
from modules.neighbor_loader import LastNeighborLoaderTGNPL
from modules.memory_module import TGNPLMemory, StaticMemory
from modules.early_stopping import  EarlyStopMonitor
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset, PyGLinkPropPredDatasetHyper
from examples.linkproppred.general.tgnpl import *
from scipy.stats import pearsonr

# recreate args
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
args = Namespace(dataset='tgbl-hypergraph_sem',
                 lr=1e-4,
                 bs=2000,  # use larger batch size since we're not training
                 k_value=10,
                 num_epoch=100,
                 seed=1,
                 mem_dim=100,
                 time_dim=10,
                 emb_dim=100,
                 tolerance=1e-6,
                 patience=100,
                 num_run=1,
                 wandb=False,
                 bipartite=False,
                 memory_name='static',
                 emb_name='sum',
                 use_inventory=False,
                 debt_penalty=0,
                 consum_rwd=0,
                 gpu=0,
                 num_train_days=-1,
                 use_prev_sampling=False,
                 batch_by_t=False)
device = torch.device("cpu")

dataset = PyGLinkPropPredDatasetHyper(name=args.dataset, root="datasets", 
                                      use_prev_sampling = args.use_prev_sampling)
data = dataset.get_TemporalData().to(device)
print("Print data.msg negative values, if any", data.msg[data.msg < 0])
train_loader, val_loader, test_loader = set_up_data(args, data, dataset)

# Check by assertion (timestamp value always increasing): seems correct! 
print("check ordering by assertion: correct if nothing fails")
print("checking train_loader")
prev_max_t = 0
for batch in tqdm(train_loader):
    current_min_t, current_max_t = min(set(batch.t)), max(set(batch.t))
    assert(prev_max_t <= current_min_t) # previous max should be <= current minimum
    prev_max_t = current_max_t

print("checking val_loader")
for batch in tqdm(val_loader):
    current_min_t, current_max_t = min(set(batch.t)), max(set(batch.t))
    assert(prev_max_t <= current_min_t) # previous max should be <= current minimum
    prev_max_t = current_max_t

print("checking test_loader")
for batch in tqdm(test_loader):
    current_min_t, current_max_t = min(set(batch.t)), max(set(batch.t))
    assert(prev_max_t <= current_min_t) # previous max should be <= current minimum
    prev_max_t = current_max_t

print("assertion checking done")
# Check by eye-balling
# print("check ordering by eye-balling")
# for batch in tqdm(train_loader):
#     print(batch.t)
