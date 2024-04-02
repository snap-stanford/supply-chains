import wandb
import math
import timeit
from tqdm import tqdm
import json
import argparse 
import sys
from torch.utils.tensorboard import SummaryWriter

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
from tgb.linkproppred.logger import TensorboardLogger
from tgb.utils.utils import *
from tgb.linkproppred.evaluate import Evaluator
from modules.graphmixer import GraphMixer
from modules.decoder import LinkPredictorTGNPL
from modules.emb_module import *
# from modules.msg_func import TGNPLMessage
# from modules.msg_agg import MeanAggregator
from modules.neighbor_loader import LastNeighborLoaderGraphmixer
# from modules.memory_module import TGNPLMemory, StaticMemory
from modules.inventory_module import TGNPLInventory
from modules.early_stopping import  EarlyStopMonitor
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset, PyGLinkPropPredDatasetHyper

# To prevent CUDA device side assertion bug
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# ===========================================
# == Main functions to train and test model
# ===========================================
def _get_y_pred_for_batch(batch, model, neighbor_loader, data, device,
                          ns_samples=6, neg_sampler=None, split_mode="val",
                          num_firms=None, num_products=None, use_prev_sampling = False):
    """
    Get model scores for a batch's positive edges and its corresponding negative samples.
    If neg_sampler is None, sample negative samples at random.
    Parameters:
        batch: a batch from data loader
        model: a dict of model modules (memory, gnn, link_pred)
        neighbor_loader: stores and loads temporal graph
        data: object holding onto all data
        device: current device
        ns_samples: how many negative samples to draw per src/prod/dst; only used if neg_sampler is None
        neg_sampler: a sampler with fixed negative samples
        split_mode: in ['val', 'test'], used for neg_sampler
        num_firms: total number of firms. Assumed that firm indices are [0 ... num_firms-1].
        num_products: total number of products. Assumed that product indices are [num_firms ... num_products+num_firms-1].
        use_prev_sampling: whether the negative hyperedges were sampled using the prior negative sampling process
                            (that is, without correction to the loose negatives on Oct 24)
    Returns:
        y_pred: shape is (batch size) x (1 + 3*ns_samples)
    """
    # use global variables when arguments are not specified
    if num_firms is None:
        num_firms = NUM_FIRMS
    if num_products is None:
        num_products = NUM_PRODUCTS
    num_nodes = num_firms + num_products
    # Helper vector to map global node indices to local ones
    assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
    
    pos_src, pos_prod, pos_dst, pos_t, pos_msg = batch.src, batch.prod, batch.dst, batch.t, batch.msg

    bs = len(pos_src)  # batch size
    
    if neg_sampler is None:
        # sample negatives        
        neg_src = torch.randint(
            0,  # min firm idx
            num_firms,  # max firm idx+1
            (bs, ns_samples),
            dtype=torch.long,
            device=device,
        )        
        neg_prod = torch.randint(
            num_firms,  # min product idx
            num_firms+num_products,  # max product idx+1
            (bs, ns_samples),
            dtype=torch.long,
            device=device,
        )
        neg_dst = torch.randint(
            0,
            num_firms,
            (bs, ns_samples),
            dtype=torch.long,
            device=device,
        )

    elif (use_prev_sampling == True):
        #using the negative sampling procedure prior to Oct 24 (no loose negatives)
        neg_batch_list = neg_sampler.query_batch(pos_src, pos_prod, pos_dst, pos_t, split_mode=split_mode)
        assert len(neg_batch_list) == bs
        neg_batch_list = torch.Tensor(neg_batch_list)
        ns_samples = neg_batch_list.size(1) // 3  # num negative samples per src/prod/dst
        neg_src = neg_batch_list[:, :ns_samples]   # we assume neg batch is ordered by neg_src, neg_prod, neg_dst
        neg_prod = neg_batch_list[:, ns_samples:(2*ns_samples)]  
        neg_dst = neg_batch_list[:, (2*ns_samples):]  
        # TODO: graphmixer might not support this - since we also need to have time 
        
    else:
        #using the current negative sampling for hypergraph
        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_prod, pos_t, split_mode=split_mode)
        assert len(neg_batch_list) == bs
        neg_batch_list = torch.Tensor(np.array(neg_batch_list)).to(device).int()
        num_samples = neg_batch_list.size(1) + 1
        neg_src = neg_batch_list[:,:,0]
        neg_dst = neg_batch_list[:,:,1]
        neg_prod = neg_batch_list[:,:,2]
        batch_src = torch.cat((torch.unsqueeze(pos_src,-1), neg_src), dim = -1)
        batch_dst = torch.cat((torch.unsqueeze(pos_dst,-1), neg_dst), dim = -1)
        batch_prod = torch.cat((torch.unsqueeze(pos_prod,-1), neg_prod), dim = -1)
        
    if (neg_sampler is None or use_prev_sampling == True):
        num_samples = (3*ns_samples)+1  # total num samples per data point
        batch_src = pos_src.reshape(bs, 1).repeat(1, num_samples)  # [[src1, src1, ...], [src2, src2, ...]]
        batch_src[:, 1:ns_samples+1] = neg_src  # replace pos_src with negatives
        batch_prod = pos_prod.reshape(bs, 1).repeat(1, num_samples)
        batch_prod[:, ns_samples+1:(2*ns_samples)+1] = neg_prod  # replace pos_prod with negatives
        batch_dst = pos_dst.reshape(bs, 1).repeat(1, num_samples)
        batch_dst[:, (2*ns_samples)+1:] = neg_dst  # replace pos_dst with negatives
    
    batch_t = pos_t.reshape(bs, 1).repeat(1, num_samples)
    src, dst, prod, t = batch_src.flatten(), batch_dst.flatten(), batch_prod.flatten(), batch_t.flatten()  # row-wise flatten
    # Note: we don't call .unique() here over the batch, so we input the format (batch_size, ?)
    batch_src_node_embeddings, batch_dst_node_embeddings, batch_prod_node_embeddings = \
                model['graphmixer'].compute_src_dst_prod_node_temporal_embeddings(src_node_ids=src,
                                                                                dst_node_ids=dst,
                                                                                prod_node_ids=prod,
                                                                                node_interact_times=t,
                                                                                neighbor_loader=neighbor_loader)
    # f_id = torch.cat([src, dst]).unique()
    # p_id = torch.cat([prod]).unique()
    # n_id, edge_index, e_id = neighbor_loader(f_id, p_id)
    # assoc[n_id] = torch.arange(n_id.size(0), device=device)

    # # Get updated memory of all nodes involved in the computation.
    # memory, last_update, update_loss = model['memory'](n_id)
    # z = model['gnn'](
    #     memory,
    #     last_update,
    #     edge_index,
    #     repeat_tensor(data.t, 2)[e_id].to(device),
    #     repeat_tensor(data.msg, 2)[e_id].to(device),
    # )
    # y_pred = model['link_pred'](z[assoc[src]], z[assoc[dst]], z[assoc[prod]])

    y_pred = model['link_pred'](batch_src_node_embeddings, 
                                batch_dst_node_embeddings,
                                batch_prod_node_embeddings).squeeze(dim=-1)
    y_pred = y_pred.reshape(bs, num_samples)
    update_loss = 0 # TODO: what's update loss for graphmixer? 
    return y_pred, update_loss
    
def _update_inventory_and_compute_loss(batch, model, neighbor_loader, data, device,
                                       num_firms=None, num_products=None):
#     """
#     Update inventory per firm based on latest batch and compute losses.
#     """
#     # use global variables when arguments are not specified
#     if num_firms is None:
#         num_firms = NUM_FIRMS
#     if num_products is None:
#         num_products = NUM_PRODUCTS
#     num_nodes = num_firms + num_products
#     # Helper vector to map global node indices to local ones
#     assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
    
#     prod, t = batch.prod.flatten(), batch.t.flatten()
#     # TODO: think about how to .unique() since we can have different timestamps in the same batch for a fixed node
    
#     # Only compute product node embedding to save time
#     model['graphmixer'].neighbor_sampler = neighbor_loader
#     prod_embs = model['graphmixer'].compute_node_temporal_embeddings(node_ids=prod, node_interact_times=t)
   
#     # prod_embs has shape (num_products, num_products)
#     inv_loss, debt_loss, consump_rwd_loss = model['inventory'](batch.src, batch.dst, batch.prod, batch.msg, prod_embs)
    inv_loss, debt_loss, consump_rwd_loss = 0, 0, 0
    return inv_loss, debt_loss, consump_rwd_loss

def train(model, optimizer, neighbor_loader, data, data_loader, device, 
          loss_name='ce-softmax', update_params=True, 
          ns_samples=6, neg_sampler=None, split_mode="val",
          num_firms=None, num_products=None, use_prev_sampling = False):
    """
    Training procedure for TGN-PL model.
    Parameters:
        model: a dict of model modules (memory, gnn, link_pred)
        optimizer: torch optimizer linked to model parameters
        neighbor_loader: stores and loads temporal graph
        data: object holding onto all data
        data_loader: loader for the train data
        device: current device
        loss_name: in ['ce-softmax', 'bce-logits']
        update_params: bool, whether to update model params
        ns_samples: how many negative samples to draw per src/prod/dst; only used if neg_sampler is None
        neg_sampler: usually None. Provide if you want to train on fixed negative samples.
        split_mode: in ['val', 'test'], used for neg_sampler
        num_firms: total number of firms. Assumed that firm indices are [0 ... num_firms-1].
        num_products: total number of products. Assumed that product indices are [num_firms ... num_products+num_firms-1].
        use_prev_sampling: whether the negative hyperedges were sampled using the prior negative sampling process
                            (that is, without correction to the loose negatives on Oct 24)
    Returns:
        total loss, logits loss (from dynamic link prediction), inventory loss, memory update loss
    """
    assert loss_name in ['ce-softmax', 'bce-logits']    
    if update_params:
        for module in model.values():
            module.train()
    else:
        for module in model.values():
            module.eval()

    # model['memory'].reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.
    
    if loss_name == 'ce-softmax':
        # softmax then cross entropy
        criterion = torch.nn.CrossEntropyLoss() 
    else:
        # sigmoid then binary cross entropy, used in TGB
        criterion = torch.nn.BCEWithLogitsLoss()
    
    total_loss, total_logits_loss, total_inv_loss, total_debt_loss, total_consump_rwd_loss, total_update_loss, total_num_events = 0, 0, 0, 0, 0, 0, 0
    for batch in tqdm(data_loader):        
        batch = batch.to(device)
        if update_params:
            optimizer.zero_grad()

        y_pred, update_loss = _get_y_pred_for_batch(batch, model, neighbor_loader, data, device,
                                       ns_samples=ns_samples, neg_sampler=neg_sampler, split_mode=split_mode,
                                       num_firms=num_firms, num_products=num_products, use_prev_sampling = use_prev_sampling)
        if 'inventory' in model:  # will be in model if args.use_inventory = True
            inv_loss, debt_loss, consump_rwd_loss = _update_inventory_and_compute_loss(batch, model, neighbor_loader, data, device,
                                                      num_firms=num_firms, num_products=num_products)
        else:
            inv_loss, debt_loss, consump_rwd_loss = 0, 0, 0
      
        if loss_name == 'ce-softmax':
            target = torch.zeros(y_pred.size(0), device=device).long()  # positive always in first position
            logits_loss = criterion(y_pred, target)
        else:
            # original binary loss from TGB
            assert y_pred.size(1) == (3*ns_samples)+1
            pos_out = y_pred[:, :1]
            neg_out = y_pred[:, 1:]            
            logits_loss = criterion(pos_out, torch.ones_like(pos_out))
            logits_loss += criterion(neg_out, torch.zeros_like(neg_out))
        
        # three losses: logits loss from link prediction, inventory loss, and memory update loss
        loss = logits_loss + inv_loss + update_loss
        total_loss += float(loss) * batch.num_events  # scale by batch size
        total_logits_loss += float(logits_loss) * batch.num_events
        total_inv_loss += float(inv_loss) * batch.num_events
        total_debt_loss += float(debt_loss) * batch.num_events
        total_consump_rwd_loss += float(consump_rwd_loss) * batch.num_events
        total_update_loss += float(update_loss) * batch.num_events
        total_num_events += batch.num_events
        
        # Update memory and neighbor loader with ground-truth transactions
        # model['memory'].update_state(batch.src, batch.dst, batch.prod, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst, batch.prod, batch.t, batch.msg)
                
        # Update model parameters with backprop
        if update_params:
            loss.backward()
            optimizer.step()
#             model['memory'].detach()
            if 'inventory' in model:
                model['inventory'].detach()

    return total_loss / total_num_events, total_logits_loss / total_num_events, total_inv_loss / total_num_events, total_debt_loss / total_num_events, total_consump_rwd_loss / total_num_events, total_update_loss / total_num_events

@torch.no_grad()
def test(model, neighbor_loader, data, data_loader, neg_sampler, evaluator, device,
         split_mode="val", metric="mrr", num_firms=None, num_products=None, use_prev_sampling = False):
    """
    Evaluation procedure for TGN-PL model.
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges

    Parameters:
        model: a dict of model modules (memory, gnn, link_pred)
        neighbor_loader: stores and loads temporal graph
        data: object holding onto all data
        data_loader: loader for the test data
        neg_sampler: a sampler with fixed negative samples
        evaluator: evaluator object that evaluates one vs many performance
        device: current device
        split_mode: in ['val', 'test'], specifies which set we are in to correctly load negatives
        metric: in ['mrr', 'hits@'], which metric to use in evaluator
        num_firms: total number of firms. Assumed that firm indices are [0 ... num_firms-1].
        num_products: total number of products. Assumed that product indices are [num_firms ... num_products+num_firms-1].
        use_prev_sampling: whether the negative hyperedges were sampled using the prior negative sampling process
                            (that is, without correction to the loose negatives on Oct 24)
    Returns:
        perf_metric: the result of the performance evaluaiton
    """
    assert split_mode in ['val', 'test']
    assert metric in ['mrr', 'hits@']
        
    model['graphmixer'].eval()
    model['link_pred'].eval()

    perf_list = []
    batch_size = []
    
    for batch in tqdm(data_loader):
        y_pred, _ = _get_y_pred_for_batch(batch, model, neighbor_loader, data, device,
                                       neg_sampler=neg_sampler, split_mode=split_mode,
                                       num_firms=num_firms, num_products=num_products,
                                       use_prev_sampling=use_prev_sampling)
        input_dict = {
            "y_pred_pos": y_pred[:, :1],
            "y_pred_neg": y_pred[:, 1:],
            "eval_metric": [metric]
        }
        perf_list.append(evaluator.eval(input_dict)[metric])
        batch_size.append(y_pred.size(0))

        # Update memory and neighbor loader with ground-truth state.
        # model['memory'].update_state(batch.src, batch.dst, batch.prod, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst, batch.prod, batch.t, batch.msg)
    
    num = (torch.tensor(perf_list) * torch.tensor(batch_size)).sum()
    denom = torch.tensor(batch_size).sum()
    perf_metrics = float(num/denom)
    return perf_metrics


# ===========================================
# == Helper functions
# ===========================================
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

def get_graphmixer_args():
    """
    Parse args from command line.
    """
    parser = argparse.ArgumentParser('*** GraphMixer ***')
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
    parser.add_argument('--num_run', type=int, help='Number of iteration runs', default=1, choices=[1])
    parser.add_argument('--wandb', type=bool, help='Wandb support', default=False)
    parser.add_argument('--tensorboard', type=bool, help='Tensorboard support', default=False)
    parser.add_argument('--bipartite', type=bool, help='Whether to use bipartite graph', default=False)
    parser.add_argument('--memory_name', type=str, help='Name of memory module', default='tgnpl', choices=['tgnpl', 'static'])
    parser.add_argument('--emb_name', type=str, help='Name of embedding module', default='attn', choices=['attn', 'sum', 'id'])
    parser.add_argument('--use_inventory', type=bool, help='Whether to use inventory module', default=False)
    parser.add_argument('--debt_penalty', type=float, help='Debt penalty weight for inventory loss', default=10)
    parser.add_argument('--consum_rwd', type=float, help='Consumption reward weight for inventory loss', default=1)
    parser.add_argument('--update_penalty', type=float, help='Regularization of TGNPL memory updates by penalizing change in memory', default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--weights', type=str, default='', help='Saved weights to initialize model with')
    parser.add_argument('--num_train_days', type=int, default=-1, help='How many days to use for training; used for debugging and faster training')
    parser.add_argument('--train_on_val', type=bool, default=False, help='If true, train on validation set with fixed negative sampled; used for debugging')
    parser.add_argument('--use_prev_sampling', action='store_true', help = "if true, use previous hypergraph sampling method")

    # Additional args for graphmixer
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_neighbors', type=int, default=10, help='number of neighbors')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap')

    try:
        args = parser.parse_args()
        defaults = parser.parse_args([])
    
    except:
        parser.print_help()
        sys.exit(0)
    return args, defaults, sys.argv

def compare_args(args, defaults): 
    """
    Compares deviations of the parsed arguments from the defaults (for labeling the checkpoints
    & results folder succinctly) 
    """
    args_dict = vars(args)
    defaults_dict = vars(defaults)
    return {key: value for key,value in args_dict.items() if (
        key not in defaults_dict or defaults_dict[key] != args_dict[key])}

def get_unique_id_for_experiment(args):
    """
    Returns a unique ID for an experiment that encodes all of its args.
    """
    curr_time = f"{current_pst_time().strftime('%Y_%m_%d-%H_%M_%S')}"
    id_elements = [MODEL_NAME]
    for arg in vars(args):  # iterate over Namespace
        if arg != 'weights':  # drop weight argument; otherwise, ID gets too long
            id_elements.append(str(getattr(args, arg)))
    id_elements.append(curr_time)
    return '_'.join(id_elements)

    
# ===========================================
# == Functions to run complete experiments
# ===========================================
# Global variables
MODEL_NAME = 'graphmixer'
#PATH_TO_DATASETS = f'/lfs/turing1/0/{os.getlogin()}/supply-chains/TGB/tgb/datasets/'
TGB_DIRECTORY = "/".join(str(__file__).split("/")[:-4])
PATH_TO_DATASETS = os.path.join(TGB_DIRECTORY, "tgb/datasets/")
NUM_FIRMS = -1
NUM_PRODUCTS = -1    
    
def set_up_model(args, data, device, num_firms=None, num_products=None, mimic_static_debug=False): # TODO: delete debug flag
    """
    Initialize model modules based on args.
    """
    # use global variables when arguments are not specified
    if num_firms is None:
        num_firms = NUM_FIRMS
    if num_products is None:
        num_products = NUM_PRODUCTS
    num_nodes = num_firms + num_products

    # initialize graphmixer module
    node_raw_features = torch.eye(num_nodes).to(device) # since node features is None here, one_hot encoding of node id
    edge_feat_dim = data.msg.shape[1]
    graphmixer = GraphMixer(node_raw_features=node_raw_features, edge_feat_dim=edge_feat_dim,
                            time_feat_dim=args.time_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, time_gap=args.time_gap, mimic_static_debug=mimic_static_debug).to(device) # TODO: delete debug flag

    # initialize 
#     link_pred = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1], input_dim3=node_raw_features.shape[1], hidden_dim=node_raw_features.shape[1], output_dim=1).to(device)
    link_pred = LinkPredictorTGNPL(node_raw_features.shape[1]).to(device) # same as MergeLayer, but without .sigmoid() at the outmost layer

    # put together in model and initialize optimizer
    model = {'graphmixer': graphmixer,
             'link_pred': link_pred}
    all_params = set(model['graphmixer'].parameters()) | set(model['link_pred'].parameters())
    
    # add inventory module if required
    if args.use_inventory:
        inventory = TGNPLInventory(
            num_firms = num_firms,
            num_prods = num_products,
            debt_penalty = args.debt_penalty,
            consumption_reward = args.consum_rwd,
            device = device,
            emb_dim = node_raw_features.shape[1],
        ).to(device)
        model['inventory'] = inventory
        all_params = all_params | set(model['inventory'].parameters())
    
    optimizer = torch.optim.Adam(all_params,lr=args.lr)
    return model, optimizer

  
def set_up_data(args, data, dataset):
    """
    Normalize edge features; split data into train, val, test.
    """
    # apply log scaling and standard scaling to edge features
    msg = data.msg
    min_val = torch.min(msg[msg > 0])
    msg = torch.clip(msg, min_val, None)  # so that we don't take log of 0
    assert (msg > 0).all()
    log_msg = torch.log(msg)  # log scale
    mean = torch.mean(log_msg)
    std = torch.std(log_msg)
    data.msg = (log_msg - mean) / std  # standard scaling
    
    if args.num_train_days == -1:
        train_data = data[dataset.train_mask]
    else:
        assert args.num_train_days > 0
        train_days = data.t[dataset.train_mask].unique()  # original set of train days
        days_to_keep = train_days[-args.num_train_days:]  # keep train days from the end, since val follows train
        new_train_mask = torch.isin(data.t, days_to_keep)
        train_data = data[new_train_mask]
    val_data = data[dataset.val_mask]
    test_data = data[dataset.test_mask]
    print('Train: N=%d, %d days; Val: N=%d, %d days; Test: N=%d, %d days' % 
          (len(train_data), len(train_data.t.unique()), len(val_data), len(val_data.t.unique()),
           len(test_data), len(test_data.t.unique()))
    )
        
    train_loader = TemporalDataLoader(train_data, batch_size=args.bs)
    val_loader = TemporalDataLoader(val_data, batch_size=args.bs)
    test_loader = TemporalDataLoader(test_data, batch_size=args.bs)
    return train_loader, val_loader, test_loader
    
def run_experiment(args):
    """
    Run a complete experiment.
    """
    global NUM_FIRMS, NUM_PRODUCTS  # add this to modify global variables
    print('Starting...')
    start_overall = timeit.default_timer()
    exp_id = get_unique_id_for_experiment(args)
    print('Experiment ID:', exp_id)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    # start tensorboard to track this script
    if args.tensorboard:
        writer = SummaryWriter(comment=exp_id)
    
    # start a new wandb run to track this script
    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project=WANDB_PROJECT,
            entity=WANDB_TEAM,
            resume="allow",
            # track hyperparameters and run metadata
            config=args
        )
        config = wandb.config
    if args.wandb:
        wandb.summary["num_neighbors"] = args.num_neighbors
        wandb.summary["model_name"] = MODEL_NAME
        
    # Set up paths for saving results and model weights
    results_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
    if not osp.exists(results_dir):
        os.mkdir(results_dir)
        print('INFO: Create directory {}'.format(results_dir))
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    results_filename = f'{results_dir}/{exp_id}_results.json'
    
    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
    if not osp.exists(save_model_dir):
        os.mkdir(save_model_dir)
        print('INFO: Create directory {}'.format(save_model_dir))
    Path(save_model_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset
    with open(os.path.join(PATH_TO_DATASETS, f"{args.dataset.replace('-', '_')}/{args.dataset}_meta.json"), "r") as f:
        metadata = json.load(f)
    # set global data variables
    num_nodes = len(metadata["id2entity"])  
    NUM_FIRMS = metadata["product_threshold"]
    NUM_PRODUCTS = num_nodes - NUM_FIRMS        
    dataset = PyGLinkPropPredDatasetHyper(name=args.dataset, root="datasets", 
                                          use_prev_sampling = args.use_prev_sampling)
    print(f"There are {NUM_FIRMS} firms and {NUM_PRODUCTS} products")
    
    metric = dataset.eval_metric
    neg_sampler = dataset.negative_sampler
    evaluator = Evaluator(name=args.dataset)
    data = dataset.get_TemporalData().to(device)
    train_loader, val_loader, test_loader = set_up_data(args, data, dataset)
    print('Converted amount to log-scale and applied standard scaling: mean = %.2f' % torch.mean(data.msg))
    if args.train_on_val:
        print('Warning: ignoring train set, training on validation set and its fixed negative samples')

    # Initialize model
    model, opt = set_up_model(args, data, device)
    if args.weights != "":
        print('Initializing model with weights from', args.weights)
        model_path = os.path.join(save_model_dir, args.weights)
        assert os.path.isfile(model_path)
        saved_model = torch.load(model_path)
        try:
            # try initializing for all modules
            for module_name, module in model.items():
                module.load_state_dict(saved_model[module_name])
            print('Success: matched all model modules and loaded weights')
        except:
            # if args.memory_name == 'tgnpl' and 'static' in args.weights:
            #     # try initializing TGN-PL initial memory with static memory
            #     model['memory'].init_memory.weight.data = saved_model['memory']['memory.weight'].to(device)
            #     print('Success: initialized init_memory with static memory')
            # else:
            # TODO: special init for graphmixer?
            raise Exception('Failed to initialize model with weights')
            
    # Initialize neighbor loader
    neighbor_loader = LastNeighborLoaderGraphmixer(num_nodes, num_neighbors=args.num_neighbors, time_gap=args.time_gap, edge_feat_dim=data.msg.shape[1], device=device)

    print("==========================================================")
    print(f"=================*** {MODEL_NAME}: LinkPropPred: {args.dataset} ***=============")
    print("==========================================================")    
    for run_idx in range(args.num_run):    
        print('-------------------------------------------------------------------------------')
        print(f"INFO: >>>>> Run: {run_idx} <<<<<")
        start_run = timeit.default_timer()
        # set the seed for deterministic results...
        torch.manual_seed(run_idx + args.seed)
        set_random_seed(run_idx + args.seed)
        # define an early stopper
        save_model_id = f'{exp_id}_{run_idx}'
        early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_id, 
                                         tolerance=args.tolerance, patience=args.patience)

        # ==================================================== Train & Validation
        dataset.load_val_ns()  # load validation negative samples
        train_loss_list = []
        val_perf_list = []
        start_train_val = timeit.default_timer()
        for epoch in range(1, args.num_epoch + 1):
            start_epoch_train = timeit.default_timer()
            if args.train_on_val:
                # used for debugging: train on validation, with fixed negative samples

                loss, logits_loss, inv_loss, debt_loss, consump_rwd_loss, update_loss = train(model, opt, neighbor_loader, data, val_loader, device, 
                                                    neg_sampler=neg_sampler, split_mode="val", 
                                                    use_prev_sampling = args.use_prev_sampling)
                # Reset memory and graph for beginning of val
                # model['memory'].reset_state()  
                neighbor_loader.reset_state()
            else:
              
                loss, logits_loss, inv_loss, debt_loss, consump_rwd_loss, update_loss = train(model, opt, neighbor_loader, data, train_loader, device,
                                                    use_prev_sampling = args.use_prev_sampling)
                # Don't reset memory and graph since val is a continuation of train
            time_train = timeit.default_timer() - start_epoch_train
            print(
                f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Logits_Loss: {logits_loss:.4f}, Inv_Loss: {inv_loss:.4f}, Debt_Loss: {debt_loss:.4f}, Consumption_Reward_Loss: {consump_rwd_loss:.4f}, Update_Loss: {update_loss:.4f}, Training elapsed Time (s): {time_train: .4f}"
            )
            train_loss_list.append(float(loss))

            # validation
            start_val = timeit.default_timer()
            perf_metric_val = test(model, neighbor_loader, data, val_loader, neg_sampler, evaluator,
                                   device, split_mode="val", metric=metric, use_prev_sampling = args.use_prev_sampling)
            time_val = timeit.default_timer() - start_val
            print(f"\tValidation {metric}: {perf_metric_val: .4f}")
            print(f"\tValidation: Elapsed time (s): {time_val: .4f}")
            val_perf_list.append(perf_metric_val)

            # log metrics to tensorboard
            if args.tensorboard:
                writer.add_scalar("loss", loss, epoch)
                writer.add_scalar("logits_loss", logits_loss, epoch)
                writer.add_scalar("inv_loss", inv_loss, epoch)
                writer.add_scalar("debt_loss", debt_loss, epoch)
                writer.add_scalar("consump_rwd_loss", consump_rwd_loss, epoch)
                writer.add_scalar("update_loss", update_loss, epoch)
                writer.add_scalar("perf_metric_val", perf_metric_val, epoch)
                writer.add_scalar("elapsed_time_train", time_train, epoch)
                writer.add_scalar("elapsed_time_val", time_val, epoch)
            
            # log metric to wandb
            if args.wandb:
                wandb.log({"loss": loss, 
                           "logits_loss": logits_loss,
                           "inv_loss": inv_loss,
                           "debt_loss": debt_loss,
                           "consump_rwd_loss": consump_rwd_loss,
                           "update_loss": update_loss,
                           "perf_metric_val": perf_metric_val, 
                           "elapsed_time_train": time_train, 
                           "elapsed_time_val": time_val
                           })
            # save train+val results after each epoch
            save_results({'model': MODEL_NAME,
                  'data': args.dataset,
                  'run': run_idx,
                  'seed': args.seed,
                  'train loss': train_loss_list,
                  f'val {metric}': val_perf_list,}, 
                    results_filename, replace_file=True)

            # check if best on val so far, save if so, stop if no improvement observed for a while
            if early_stopper.step_check(perf_metric_val, model):
                break
                
        # also save final model
        model_path = os.path.join(save_model_dir, save_model_id + '_final.pth')
        print("INFO: save final model to {}".format(model_path))
        model_names = list(model.keys())
        model_components = list(model.values())
        torch.save({model_names[i]: model_components[i].state_dict() for i in range(len(model_names))}, 
                   model_path)

        train_val_time = timeit.default_timer() - start_train_val
        print(f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")

        # ==================================================== Test
        early_stopper.load_checkpoint(model)  # load the best model
        dataset.load_test_ns()  # load test negatives
        start_test = timeit.default_timer()
        perf_metric_test = test(model, neighbor_loader, data, test_loader, neg_sampler, evaluator,
                                device, split_mode="test", metric=metric, use_prev_sampling = args.use_prev_sampling)

        print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
        print(f"\tTest: {metric}: {perf_metric_test: .4f}")
        test_time = timeit.default_timer() - start_test
        print(f"\tTest: Elapsed Time (s): {test_time: .4f}")
        if args.tensorboard:
            hparam_dict = vars(args)
            hparam_dict.update({"num_neighbors": args.num_neighbors,
                                "model_name": MODEL_NAME, 
                                "metric": metric})
            metric_dict = {
                "best_epoch": early_stopper.best_epoch,
                "perf_metric_test": perf_metric_test
            }
            writer.add_hparams(hparam_dict, metric_dict)
            writer.add_scalar("elapsed_time_test", test_time)
            
        if args.wandb:
            wandb.summary["metric"] = metric
            wandb.summary["best_epoch"] = early_stopper.best_epoch
            wandb.summary["perf_metric_test"] = perf_metric_test
            wandb.summary["elapsed_time_test"] = test_time

        save_results({'model': MODEL_NAME,
                      'data': args.dataset,
                      'run': run_idx,
                      'seed': args.seed,
                      'train loss': train_loss_list,
                      f'val {metric}': val_perf_list,
                      f'test {metric}': perf_metric_test,
                      'test_time': test_time,
                      'tot_train_val_time': train_val_time
                      }, 
            results_filename, replace_file=True)

        print(f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
        print('-------------------------------------------------------------------------------')

    print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
    print("==============================================================")
    if args.tensorboard:
        writer.close()
    if args.wandb:
        wandb.finish()
        

if __name__ == "__main__":
    # Parse parameters
    args, defaults, _ = get_graphmixer_args()
    labeling_args = compare_args(args, defaults)
    run_experiment(args)
