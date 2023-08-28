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
from modules.msg_func import IdentityMessageTGNPL
from modules.msg_agg import LastAggregator
from modules.neighbor_loader import LastNeighborLoaderTGNPL
from modules.memory_module import TGNPLMemory
from modules.early_stopping import  EarlyStopMonitor
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

# ==========
# ========== Define helper function...
# ==========

import json

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

    total_loss = 0
    for batch in train_loader:
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

<<<<<<< HEAD:TGB/examples/linkproppred/tgbl-supplychains/tgn.py
        
        
        if (bipartite == False):
            # Sample negative destination nodes.
            neg_dst = torch.randint(
                min_dst_idx,
                max_dst_idx + 1,
                (src.size(0),),
                dtype=torch.long,
                device=device,
            )
        else:
            neg_dst = torch.zeros((src.size(0),), dtype = torch.long, device = device)
            num_products = int(torch.sum(src >= product_min_idx).item())
            num_firms = int(torch.sum(src < product_min_idx).item())
            neg_dst[src >= product_min_idx] = torch.randint(min_dst_idx, product_min_idx, (num_products,), dtype = torch.long,
                                                           device = device)
            neg_dst[src < product_min_idx] = torch.randint(product_min_idx, max_dst_idx + 1, (num_firms,), dtype = torch.long,
                                                           device = device)

            
            #print(src >= product_min_idx) #get products, (min_dst_idx, max_dst_idx + 1)
            #print(product_min_idx)
            #print(neg_dst)
            #print(src)
            #print(pos_dst)
            #raise ValueError("bruh")


        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
=======
        neg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )
        
        f_id = torch.cat([src, dst, neg_src, neg_dst]).unique()
        p_id = torch.cat([prod, neg_prod]).unique()
        # 'n_id' indexes both firm and product nodes (positive and negative)
        # 'edge_index' are relevant firm-product and product-firm edges
        n_id, edge_index, e_id = neighbor_loader(f_id, p_id)
>>>>>>> c3802ac9aa11a7b57e9fc91900a1ce018b802b3f:TGB/examples/linkproppred/general/tgnpl.py
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        
        # Get updated memory of all nodes involved in the computation.
        memory, last_update = model['memory'](n_id)

        z = model['gnn'](
            memory,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
        )

        pos_out = model['link_pred'](z[assoc[src]], z[assoc[dst]], z[assoc[prod]])
        neg_out_src = model['link_pred'](z[assoc[neg_src]], z[assoc[dst]], z[assoc[prod]])
        neg_out_prod = model['link_pred'](z[assoc[src]], z[assoc[dst]], z[assoc[neg_prod]])
        neg_out_dst = model['link_pred'](z[assoc[src]], z[assoc[neg_dst]], z[assoc[prod]])

        # TODO: add inventory constraint
        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out_src, torch.zeros_like(neg_out_src))
        loss += criterion(neg_out_prod, torch.zeros_like(neg_out_prod))
        loss += criterion(neg_out_dst, torch.zeros_like(neg_out_dst))

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(src, dst, prod, t, msg) # handle inventory
        neighbor_loader.insert(src, dst, prod)

        loss.backward()
        optimizer.step()
        model['memory'].detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events

# TODO: modify test()
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

    for pos_batch in loader:
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )

        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)

        for idx, neg_batch in enumerate(neg_batch_list):
            src = torch.full((1 + len(neg_batch),), pos_src[idx], device=device)
            dst = torch.tensor(
                np.concatenate(
                    ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                    axis=0,
                ),
                device=device,
            )

            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # Get updated memory of all nodes involved in the computation.
            z, last_update = model['memory'](n_id)
            z = model['gnn'](
                z,
                last_update,
                edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )

            y_pred = model['link_pred'](z[assoc[src]], z[assoc[dst]])

            # compute MRR
            input_dict = {
                "y_pred_pos": np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
                "y_pred_neg": np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(pos_src, pos_dst, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst)

    perf_metrics = float(torch.tensor(perf_list).mean())

    return perf_metrics

import argparse 
import sys

def get_tgn_args():
    parser = argparse.ArgumentParser('*** TGB ***')
    parser.add_argument('--dataset', type=str, help='Dataset name', default='tgbl-supplychains')
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

<<<<<<< HEAD:TGB/examples/linkproppred/tgbl-supplychains/tgn.py
FOLDER_NAME_HDR = f"model={MODEL_NAME}"
for arg_name in sorted(list(labeling_args.keys())):
    arg_value = labeling_args[arg_name]
    FOLDER_NAME_HDR += f"_{arg_name}={arg_value}"
    
#print("INFO: Arguments:", args)
#print(FOLDER_NAME_HDR)
#raise ValueError("bruh")

""" modify this to accept addtional args for dataset name """
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
bipartite = args.bipartite 

#preprocessing for the bipartite graph 
DATA_FOLDER_NAME = DATA.replace("-","_")
metadata_path = f"./tgb/datasets/{DATA_FOLDER_NAME}/{DATA}_meta.json"
with open(metadata_path,"r") as file:
    metadata_supplychains = json.load(file) 
    product_min_idx = metadata_supplychains["product_threshold"]

=======
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

DATA = config.data if WANDB else args.data
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
assert (NUM_RUNS == 1)

NUM_NEIGHBORS = 10
MODEL_NAME = 'TGN'
if WANDB:
    wandb.summary["num_neighbors"] = NUM_NEIGHBORS
    wandb.summary["model_name"] = MODEL_NAME

UNIQUE_TIME = f"{current_pst_time().strftime('%Y_%m_%d-%H_%M_%S')}"
UNIQUE_NAME = f"{MODEL_NAME}_{DATA}_{LR}_{BATCH_SIZE}_{K_VALUE}_{NUM_EPOCH}_{SEED}_{MEM_DIM}_{TIME_DIM}_{EMB_DIM}_{TOLERANCE}_{PATIENCE}_{NUM_RUNS}_{NUM_NEIGHBORS}_{UNIQUE_TIME}"
>>>>>>> c3802ac9aa11a7b57e9fc91900a1ce018b802b3f:TGB/examples/linkproppred/general/tgnpl.py
# ==========

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data loading
dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
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
# TODO: @ Ben, check data.num_nodes accounts for both firm and product
neighbor_loader = LastNeighborLoaderTGNPL(data.num_nodes, size=NUM_NEIGHBORS, device=device)

# define the model end-to-end
memory = TGNPLMemory(
    data.num_nodes,
    data.msg.size(-1),
    MEM_DIM,
    TIME_DIM,
    message_module=IdentityMessageTGNPL(data.msg.size(-1), MEM_DIM, TIME_DIM),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=MEM_DIM,
    out_channels=EMB_DIM,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictorTGNPL(in_channels=EMB_DIM).to(device)

model = {'memory': memory,
         'gnn': gnn,
         'link_pred': link_pred}

# TODO: if inventory is trainable, add inventory.parameters() here too
optimizer = torch.optim.Adam(
    set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters()),
    lr=LR,
)
criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

print("==========================================================")
print(f"=================*** {MODEL_NAME}: LinkPropPred: {DATA} ***=============")
print("==========================================================")

evaluator = Evaluator(name=DATA)
neg_sampler = dataset.negative_sampler

# for saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/{FOLDER_NAME_HDR}_saved_results'
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
<<<<<<< HEAD:TGB/examples/linkproppred/tgbl-supplychains/tgn.py
    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/{FOLDER_NAME_HDR}_saved_models/'
    save_model_id = f'{MODEL_NAME}_{DATA}_{SEED}_{run_idx}'
=======
    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
    save_model_id = f'{UNIQUE_NAME}_{run_idx}'
>>>>>>> c3802ac9aa11a7b57e9fc91900a1ce018b802b3f:TGB/examples/linkproppred/general/tgnpl.py
    early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_id, 
                                    tolerance=TOLERANCE, patience=PATIENCE)

    # ==================================================== Train & Validation
    # loading the validation negative samples
    dataset.load_val_ns()

    val_perf_list = []
    start_train_val = timeit.default_timer()
    for epoch in range(1, NUM_EPOCH + 1):
        # training
        start_epoch_train = timeit.default_timer()
        loss = train()
        TIME_TRAIN = timeit.default_timer() - start_epoch_train
        print(
            f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {TIME_TRAIN: .4f}"
        )

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
