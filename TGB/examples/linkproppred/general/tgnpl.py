"""
Dynamic Link Prediction with a TGN model with Early Stopping
Reference: 
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py

command for an example run:
    python examples/linkproppred/tgbl-wiki/tgn.py --data "tgbl-wiki" --num_run 1 --seed 1
"""
import wandb
import math
import timeit
from tqdm import tqdm

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
from tgb.utils.utils import *
from tgb.linkproppred.evaluate import Evaluator
from modules.decoder import LinkPredictorTGNPL
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import TGNPLMessage
from modules.msg_agg import MeanAggregator
from modules.neighbor_loader import LastNeighborLoaderTGNPL
from modules.memory_module import TGNPLMemory, StaticMemory
from modules.early_stopping import  EarlyStopMonitor
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset, PyGLinkPropPredDatasetHyper

# ==========
# ========== Define helper function...
# ==========

import json

def repeat_tensor(t, k):
    """
    When k = 2, tensor([1, 2, 3]) becomes tensor([1, 1, 2, 2, 3, 3]).
    Used to align the ordering of neighbor loader 'e_id' and data
    """
    if len(t.shape)==1:
        return t.reshape(-1, 1).repeat(1, k).reshape(t.shape[0]*k)
    elif len(t.shape)==2:
        return t.reshape(-1, 1).repeat(1, k).reshape(t.shape[0]*k, 1)
    else:
        raise Exception("repeat_tensor: Not Applicable")

def train():
    r"""
    Training procedure for TGN model
    This function uses some objects that are globally defined in the current scrips 

    Parameters:
        None
    Returns:
        None
            
    """

    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()

    model['memory'].reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss, total_logits_loss, total_inv_loss = 0, 0, 0
    for batch in tqdm(train_loader):

        torch.autograd.set_detect_anomaly(True)
        
        batch = batch.to(device)
        optimizer.zero_grad()

        src, prod, dst, t, msg = batch.src, batch.prod, batch.dst, batch.t, batch.msg # Note: msg is amount
        
        # Sample negative source, product, destination nodes.
        neg_src = torch.randint(
            min_src_idx,
            max_src_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )
         
        neg_prod = torch.randint(
            min_prod_idx,
            max_prod_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )

        neg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )
        
        f_id = torch.cat([src, dst, neg_src, neg_dst]).unique()
        p_id = torch.cat([prod, neg_prod]).unique()
        # 'n_id' indexes both firm and product nodes (positive and negative),'edge_index' are relevant firm-product and product-firm edges
        n_id, edge_index, e_id = neighbor_loader(f_id, p_id)

        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        
        # Get updated memory of all nodes involved in the computation.
        memory, last_update, inv_loss = model['memory'](n_id)
        
        z = model['gnn'](
            memory,
            last_update,
            edge_index,
            repeat_tensor(data.t, 2)[e_id].to(device),
            repeat_tensor(data.msg, 2)[e_id].to(device),
        )

        pos_out = model['link_pred'](z[assoc[src]], z[assoc[dst]], z[assoc[prod]])
        neg_out_src = model['link_pred'](z[assoc[neg_src]], z[assoc[dst]], z[assoc[prod]])
        neg_out_prod = model['link_pred'](z[assoc[src]], z[assoc[dst]], z[assoc[neg_prod]])
        neg_out_dst = model['link_pred'](z[assoc[src]], z[assoc[neg_dst]], z[assoc[prod]])

        logits_loss = criterion(pos_out, torch.ones_like(pos_out))
        logits_loss += criterion(neg_out_src, torch.zeros_like(neg_out_src))
        logits_loss += criterion(neg_out_prod, torch.zeros_like(neg_out_prod))
        logits_loss += criterion(neg_out_dst, torch.zeros_like(neg_out_dst))
        loss = logits_loss + inv_loss

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(src, dst, prod, t, msg) # handle inventory
        neighbor_loader.insert(src, dst, prod)

        loss.backward()
        optimizer.step()
        model['memory'].detach()
        total_loss += float(loss) * batch.num_events
        total_logits_loss += float(logits_loss) * batch.num_events
        total_inv_loss += float(inv_loss) * batch.num_events

    return total_loss / train_data.num_events, total_logits_loss / train_data.num_events, total_inv_loss / train_data.num_events

@torch.no_grad()
def test(loader, neg_sampler, split_mode):
    r"""
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges

    Parameters:
        loader: an object containing positive attributes of the positive edges of the evaluation set
        neg_sampler: an object that gives the negative edges corresponding to each positive edge
        split_mode: specifies whether it is the 'validation' or 'test' set to correctly load the negatives
    Returns:
        perf_metric: the result of the performance evaluaiton
    """
    model['memory'].eval()
    model['gnn'].eval()
    model['link_pred'].eval()

    perf_list = []
    batch_size = []
    
    for pos_batch in tqdm(loader):
        pos_src, pos_prod, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.prod,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )
        bs = len(pos_src)  # batch size

        neg_batch_list = neg_sampler.query_batch(pos_src, pos_prod, pos_dst, pos_t, split_mode=split_mode)
        assert len(neg_batch_list) == bs
        neg_batch_list = torch.Tensor(neg_batch_list)
        ns_samples = neg_batch_list.size(1) // 3 
        batch_src = pos_src.reshape(bs, 1).repeat(1, 1+(3*ns_samples))  # [[src1, src1, ...], [src2, src2, ...]]
        batch_src[:, 1:ns_samples+1] = neg_batch_list[:, :ns_samples]  # replace pos_src with negatives
        batch_prod = pos_prod.reshape(bs, 1).repeat(1, 1+(3*ns_samples))
        batch_prod[:, ns_samples+1:(2*ns_samples)+1] = neg_batch_list[:, ns_samples:(2*ns_samples)]  # replace pos_prod with negatives
        batch_dst = pos_dst.reshape(bs, 1).repeat(1, 1+(3*ns_samples))
        batch_dst[:, (2*ns_samples)+1:] = neg_batch_list[:, (2*ns_samples):]  # replace pos_dst with negatives
        
        src, dst, prod = batch_src.flatten(), batch_dst.flatten(), batch_prod.flatten()  # row-wise
        f_id = torch.cat([src, dst]).unique()
        p_id = torch.cat([prod]).unique()
        
        n_id, edge_index, e_id = neighbor_loader(f_id, p_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
            
        # Get updated memory of all nodes involved in the computation.
        memory, last_update, inv_loss = model['memory'](n_id)
        z = model['gnn'](
            memory,
            last_update,
            edge_index,
            repeat_tensor(data.t, 2)[e_id].to(device),
            repeat_tensor(data.msg, 2)[e_id].to(device),
        )

        y_pred = model['link_pred'](z[assoc[src]], z[assoc[dst]], z[assoc[prod]])
        y_pred = y_pred.reshape(bs, 1+(3*ns_samples))
        input_dict = {
            "y_pred_pos": y_pred[:, :1],
            "y_pred_neg": y_pred[:, 1:],
            "eval_metric": [metric]
        }
        perf_list.append(evaluator.eval(input_dict)[metric])
        batch_size.append(len(pos_src))

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(pos_src, pos_dst, pos_prod, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst, pos_prod)
    
    num = (torch.tensor(perf_list) * torch.tensor(batch_size)).sum()
    denom = torch.tensor(batch_size).sum()
    perf_metrics = float(num/denom)
    return perf_metrics

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
    parser.add_argument('--bipartite', type=bool, help='Whether to use bipartite graph', default=False)
    parser.add_argument('--memory_name', type=str, help='Name of memory module', default='tgnpl', choices=['tgnpl', 'static'])
    parser.add_argument('--use_inventory', type=bool, help='Whether to use inventory in TGNPL memory', default=False)
    parser.add_argument('--debt_penalty', type=float, help='Debt penalty weight for calculating TGNPL memory inventory loss', default=0)
    parser.add_argument('--consum_rwd', type=float, help='Consumption reward weight for calculating TGNPL memory inventory loss', default=0)
    
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
print(args)
labeling_args = compare_args(args, defaults)
MODEL_NAME = 'TGNPL'

#print(args.wandb)

# start a new wandb run to track this script
WANDB = args.wandb
if WANDB:
    wandb.init(
        # set the wandb project where this run will be logged
        project=WANDB_PROJECT,
        entity=WANDB_TEAM,
        resume="allow",

        # track hyperparameters and run metadata
        config=args
    )
    config = wandb.config

DATA = config.dataset if WANDB else args.dataset
LR = config.lr if WANDB else args.lr
BATCH_SIZE = config.bs if WANDB else args.bs
K_VALUE = config.k_value if WANDB else args.k_value  
NUM_EPOCH = config.num_epoch if WANDB else args.num_epoch
SEED = config.seed if WANDB else args.seed
MEM_DIM = config.mem_dim if WANDB else args.mem_dim
TIME_DIM = config.time_dim if WANDB else args.time_dim
EMB_DIM = config.emb_dim if WANDB else args.emb_dim
TOLERANCE = config.tolerance if WANDB else args.tolerance
PATIENCE = config.patience if WANDB else args.patience
NUM_RUNS = config.num_run if WANDB else args.num_run
MEMORY_NAME = config.memory_name if WANDB else args.memory_name
USE_INVENTORY = config.use_inventory if WANDB else args.use_inventory
DEBT_PENALTY = config.debt_penalty if WANDB else args.debt_penalty
CONSUM_RWD = config.consum_rwd if WANDB else args.consum_rwd
assert (NUM_RUNS == 1)

NUM_NEIGHBORS = 10
if WANDB:
    wandb.summary["num_neighbors"] = NUM_NEIGHBORS
    wandb.summary["model_name"] = MODEL_NAME

UNIQUE_TIME = f"{current_pst_time().strftime('%Y_%m_%d-%H_%M_%S')}"
UNIQUE_NAME = f"{MODEL_NAME}_{DATA}_{LR}_{BATCH_SIZE}_{K_VALUE}_{NUM_EPOCH}_{SEED}_{MEM_DIM}_{TIME_DIM}_{EMB_DIM}_{TOLERANCE}_{PATIENCE}_{NUM_RUNS}_{NUM_NEIGHBORS}_{MEMORY_NAME}_{USE_INVENTORY}_{DEBT_PENALTY}_{CONSUM_RWD}_{UNIQUE_TIME}"

# ==========

# open the meta file (change to flexible path later)
import json
with open(f"/lfs/turing1/0/{os.getlogin()}/supply-chains/TGB/tgb/datasets/{DATA.replace('-', '_')}/{DATA}_meta.json","r") as file:
    METADATA = json.load(file)
    NUM_NODES = len(METADATA["id2entity"])

print("Starting")
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

# Ensure to only sample actual source, product, or destination nodes as negatives.
min_src_idx, max_src_idx = int(data.src.min()), int(data.src.max())
min_prod_idx, max_prod_idx = int(data.prod.min()), int(data.prod.max())
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

# neighhorhood sampler
neighbor_loader = LastNeighborLoaderTGNPL(NUM_NODES, size=NUM_NEIGHBORS, device=device)
NUM_FIRMS = METADATA["product_threshold"]
NUM_PRODUCTS = NUM_NODES - NUM_FIRMS

# define the model end-to-end
if MEMORY_NAME == 'tgnpl':
    memory = TGNPLMemory(
        use_inventory = USE_INVENTORY,
        num_nodes = NUM_NODES,
        num_prods = NUM_PRODUCTS,
        raw_msg_dim = data.msg.size(-1),
        state_dim = MEM_DIM,
        time_dim = TIME_DIM,
        message_module=TGNPLMessage(data.msg.size(-1), MEM_DIM+(NUM_PRODUCTS if USE_INVENTORY else 0), TIME_DIM),
        aggregator_module=MeanAggregator(),
        debt_penalty=DEBT_PENALTY,
        consumption_reward=CONSUM_RWD,
    ).to(device)
else:
    assert MEMORY_NAME == 'static'
    assert not USE_INVENTORY
    memory = StaticMemory(num_nodes = NUM_NODES, memory_dim = MEM_DIM, time_dim = TIME_DIM).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=MEM_DIM+(NUM_PRODUCTS if USE_INVENTORY else 0),
    out_channels=EMB_DIM,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictorTGNPL(in_channels=EMB_DIM).to(device)

model = {'memory': memory,
         'gnn': gnn,
         'link_pred': link_pred}

optimizer = torch.optim.Adam(
    set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters()),
    lr=LR,
)
criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(NUM_NODES, dtype=torch.long, device=device)

print("==========================================================")
print(f"=================*** {MODEL_NAME}: LinkPropPred: {DATA} ***=============")
print("==========================================================")

evaluator = Evaluator(name=DATA)
neg_sampler = dataset.negative_sampler

# for saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{UNIQUE_NAME}_results.json'

for run_idx in range(NUM_RUNS):    
    print('-------------------------------------------------------------------------------')
    print(f"INFO: >>>>> Run: {run_idx} <<<<<")
    start_run = timeit.default_timer()

    # set the seed for deterministic results...
    torch.manual_seed(run_idx + SEED)
    set_random_seed(run_idx + SEED)

    # define an early stopper
    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
    save_model_id = f'{UNIQUE_NAME}_{run_idx}'

    early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_id, 
                                    tolerance=TOLERANCE, patience=PATIENCE)

    # ==================================================== Train & Validation
    # loading the validation negative samples
    dataset.load_val_ns()

    train_loss_list = []
    val_perf_list = []
    start_train_val = timeit.default_timer()
    for epoch in range(1, NUM_EPOCH + 1):
        # training
        start_epoch_train = timeit.default_timer()
        loss, logits_loss, inv_loss = train()
        TIME_TRAIN = timeit.default_timer() - start_epoch_train
        print(
            f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Logits_Loss: {logits_loss:.4f}, Inv_Loss: {inv_loss:.4f}, Training elapsed Time (s): {TIME_TRAIN: .4f}"
        )
        train_loss_list.append(float(loss))

        # validation
        start_val = timeit.default_timer()
        perf_metric_val = test(val_loader, neg_sampler, split_mode="val")
        TIME_VAL = timeit.default_timer() - start_val
        print(f"\tValidation {metric}: {perf_metric_val: .4f}")
        print(f"\tValidation: Elapsed time (s): {TIME_VAL: .4f}")
        val_perf_list.append(perf_metric_val)

        # log metric to wandb
        if WANDB:
            wandb.log({"loss": loss, 
                       "logits_loss": logits_loss,
                       "inv_loss": inv_loss,
                       "perf_metric_val": perf_metric_val, 
                       "elapsed_time_train": TIME_TRAIN, 
                       "elapsed_time_val": TIME_VAL
                       })

        # check for early stopping
        if early_stopper.step_check(perf_metric_val, model):
            break

    train_val_time = timeit.default_timer() - start_train_val
    print(f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")

    # ==================================================== Test
    # first, load the best model
    early_stopper.load_checkpoint(model)

    # loading the test negative samples
    dataset.load_test_ns()

    # final testing
    start_test = timeit.default_timer()
    perf_metric_test = test(test_loader, neg_sampler, split_mode="test")

    print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
    print(f"\tTest: {metric}: {perf_metric_test: .4f}")
    test_time = timeit.default_timer() - start_test
    print(f"\tTest: Elapsed Time (s): {test_time: .4f}")
    if WANDB:
        wandb.summary["metric"] = metric
        wandb.summary["best_epoch"] = early_stopper.best_epoch
        wandb.summary["perf_metric_test"] = perf_metric_test
        wandb.summary["elapsed_time_test"] = test_time


    save_results({'model': MODEL_NAME,
                  'data': DATA,
                  'run': run_idx,
                  'seed': SEED,
                  'train loss': train_loss_list,
                  f'val {metric}': val_perf_list,
                  f'test {metric}': perf_metric_test,
                  'test_time': test_time,
                  'tot_train_val_time': train_val_time
                  }, 
    results_filename)

    print(f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
    print('-------------------------------------------------------------------------------')

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
print("==============================================================")
wandb.finish()
