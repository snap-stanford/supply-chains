import wandb
import math
import timeit
import time
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
from modules.decoder import DecoderTGNPL
from modules.emb_module import *
from modules.msg_func import TGNPLMessage
from modules.msg_agg import MeanAggregator
from modules.neighbor_loader import LastNeighborLoaderTGNPL, LastNeighborLoaderGraphmixer
from modules.memory_module import TGNPLMemory, StaticMemory
from modules.inventory_module import TGNPLInventory, mean_average_precision
from modules.early_stopping import  EarlyStopMonitor
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDatasetHyper, TimeSpecificDataLoader

# ===========================================
# == Main functions to train and test model
# ===========================================
def get_y_pred_for_batch(batch, model, neighbor_loader, data, device,
                         ns_samples=6, neg_sampler=None, split_mode="val",
                         num_firms=None, num_products=None, use_prev_sampling=False,
                         include_inventory=True):
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
        include_inventory: whether to include the inventory module 
    Returns:
        y_link_pred: shape is (batch size) x (1 + 3*ns_samples)
        y_amt_pred: shape is (batch size) x 1 if predict_amount is True; else, None
        update_loss: Tensor float
    """
    assert MODEL_NAME in {'TGNPL', 'GRAPHMIXER'} or include_inventory
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

    elif use_prev_sampling: # not supported by graphmixer
        #using the negative sampling procedure prior to Oct 24 (no loose negatives)
        neg_batch_list = neg_sampler.query_batch(pos_src, pos_prod, pos_dst, pos_t, split_mode=split_mode)
        assert len(neg_batch_list) == bs
        neg_batch_list = torch.Tensor(neg_batch_list)
        ns_samples = neg_batch_list.size(1) // 3  # num negative samples per src/prod/dst
        neg_src = neg_batch_list[:, :ns_samples]   # we assume neg batch is ordered by neg_src, neg_prod, neg_dst
        neg_prod = neg_batch_list[:, ns_samples:(2*ns_samples)]  
        neg_dst = neg_batch_list[:, (2*ns_samples):]  
        
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
        
    if (neg_sampler is None or use_prev_sampling):
        num_samples = (3*ns_samples)+1  # total num samples per data point
        batch_src = pos_src.reshape(bs, 1).repeat(1, num_samples)  # [[src1, src1, ...], [src2, src2, ...]]
        batch_src[:, 1:ns_samples+1] = neg_src  # replace pos_src with negatives
        batch_prod = pos_prod.reshape(bs, 1).repeat(1, num_samples)
        batch_prod[:, ns_samples+1:(2*ns_samples)+1] = neg_prod  # replace pos_prod with negatives
        batch_dst = pos_dst.reshape(bs, 1).repeat(1, num_samples)
        batch_dst[:, (2*ns_samples)+1:] = neg_dst  # replace pos_dst with negatives
    
    batch_t = pos_t.reshape(bs, 1).repeat(1, num_samples)
    src, dst, prod, t = batch_src.flatten(), batch_dst.flatten(), batch_prod.flatten(), batch_t.flatten()  # row-wise flatten

    # get predictions from TGNPL or GraphMixer
    if MODEL_NAME in {'TGNPL', 'GRAPHMIXER'}:
        if MODEL_NAME == 'TGNPL':
            # Get updated memory of all nodes involved in the computation.
            f_id = torch.cat([src, dst]).unique()
            p_id = torch.cat([prod]).unique()
            n_id, edge_index, e_id = neighbor_loader(f_id, p_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)
            
            memory, last_update, update_loss = model['memory'](n_id)
            z = model['gnn'](
                memory,
                last_update,
                edge_index,
                repeat_tensor(data.t, 2)[e_id].to(device),
                repeat_tensor(data.msg, 2)[e_id].to(device),
            )
            y_link_pred = model['link_pred'](z[assoc[src]], z[assoc[dst]], z[assoc[prod]]).reshape(bs, num_samples)
        elif MODEL_NAME == 'GRAPHMIXER':
            batch_src_node_embeddings, batch_dst_node_embeddings, batch_prod_node_embeddings = \
                model['graphmixer'].compute_src_dst_prod_node_temporal_embeddings(src_node_ids=src,
                                                                                  dst_node_ids=dst,
                                                                                  prod_node_ids=prod,
                                                                                  node_interact_times=t,
                                                                                  neighbor_loader=neighbor_loader)
            y_link_pred = model['link_pred'](batch_src_node_embeddings,
                                             batch_dst_node_embeddings,
                                             batch_prod_node_embeddings).squeeze(dim=-1).reshape(bs, num_samples)
            update_loss = 0 # TODO: is this true? 
        if 'amount_pred' in model:
            if MODEL_NAME == 'TGNPL':
                y_amt_pred = model['amount_pred'](z[assoc[pos_src]], z[assoc[pos_dst]], z[assoc[pos_prod]])
            elif MODEL_NAME == 'GRAPHMIXER':
                batch_pos_src_node_embeddings, batch_pos_dst_node_embeddings, batch_pos_prod_node_embeddings = \
                    model['graphmixer'].compute_src_dst_prod_node_temporal_embeddings(src_node_ids=pos_src,
                                                                                      dst_node_ids=pos_dst,
                                                                                      prod_node_ids=pos_prod,
                                                                                      node_interact_times=pos_t,
                                                                                      neighbor_loader=neighbor_loader)
                y_amt_pred = model['amount_pred'](batch_pos_src_node_embeddings, 
                                                  batch_pos_dst_node_embeddings,
                                                  batch_pos_prod_node_embeddings)
            assert y_amt_pred.shape == (bs, 1)
    else:
        update_loss = 0
    
    # get predictions from inventory module
    if 'inventory' in model and include_inventory:
        assert (batch.t >= model['inventory'].curr_t).all()
        to_skip = batch.t > model['inventory'].curr_t  # inventory module can't help with future timesteps
        if model['inventory'].learn_att_direct:
            penalties = model['inventory'].link_pred_penalties(src, prod).reshape(bs, num_samples)
            caps = model['inventory'].amount_caps(pos_src, pos_prod)
        else:  # need product embeddings
            prod_embs = get_product_embeddings(model, neighbor_loader, num_firms, num_products, data, device)
            penalties = model['inventory'].link_pred_penalties(src, prod, prod_emb=prod_embs).reshape(bs, num_samples)
            caps = model['inventory'].amount_caps(pos_src, pos_prod, prod_emb=prod_embs)
        
        # use inventory module to penalize "impossible" transactions
        assert (penalties >= 0).all()
        keep_pred = (~to_skip).int().reshape(-1, 1)
        penalties = keep_pred * penalties  # set to 0 wherever t > curr_t
        
        # use inventory module to cap maximum amount possible
        # we use -1 to indicate products without parts; can't compute cap
        neg_ones = torch.full(caps.shape, -1, dtype=caps.dtype).to(device)
        no_parts = torch.isclose(caps, neg_ones)  
        assert (caps[~no_parts] >= 0).all()
        skip_amt = to_skip | no_parts
        amt_exceeds_cap = (batch.amt[~skip_amt] > caps[~skip_amt]).float().mean()  # real amount should rarely exceed cap
        if amt_exceeds_cap > 0.1:
            print(f'Warning: batch ts={batch.t.unique()}; amount exceeds inventory caps {amt_exceeds_cap:0.3f} of the time')
        caps = (torch.log(caps)-AMT_MEAN)/AMT_STD  # apply log and standard scaling, -1 will become nan
        caps = caps.reshape(-1, 1)
        
        if MODEL_NAME == 'INVENTORY':
            y_link_pred = -penalties
            caps[skip_amt] = torch.zeros_like(caps[skip_amt])  # replace with 0 as neutral prediction
            y_amt_pred = caps
        else:  # adjust TGNPL or GraphMixer predictions with inventory module
            y_link_pred -= penalties
            if 'amount_pred' in model:
                caps[skip_amt] = y_amt_pred[skip_amt]  # replace with original prediction
                y_amt_pred = torch.minimum(y_amt_pred, caps)  # can't predict more than inventory caps    
            else:
                y_amt_pred = None
        
    return y_link_pred, y_amt_pred, update_loss
    
    
def update_inventory_and_compute_loss(batch, model, neighbor_loader, data, device,
                                      num_firms=None, num_products=None):
    """
    Update inventory per firm based on latest batch and compute losses.
    """
    if model['inventory'].learn_att_direct:
        inv_loss, debt_loss, consump_rwd = model['inventory'](batch.src, batch.dst, batch.prod, batch.t, batch.amt)
    else:
        # get product embeddings
        if num_firms is None:
            num_firms = NUM_FIRMS
        if num_products is None:
            num_products = NUM_PRODUCTS
        num_nodes = num_firms + num_products
        prod_embs = get_product_embeddings(model, neighbor_loader, num_firms, num_products, data, device)
        inv_loss, debt_loss, consump_rwd = model['inventory'](batch.src, batch.dst, batch.prod, batch.t, 
                                                              batch.amt, prod_emb=prod_embs)
    return inv_loss, debt_loss, consump_rwd


def get_product_embeddings(model, neighbor_loader, num_firms, num_products, data, device):
    """
    Helper function to get current embeddings for all products.
    """
    assert MODEL_NAME in {'TGNPL', 'GRAPHMIXER'}
    if MODEL_NAME == 'TGNPL':
        # Helper vector to map global node indices to local ones
        assoc = torch.empty(num_firms+num_products, dtype=torch.long, device=device)
        f_id = torch.Tensor([]).long().to(device)  # we only need embeddings for products, not firms
        p_id = torch.arange(num_firms, num_firms+num_products, device=device).long()  # all product IDs
        n_id, edge_index, e_id = neighbor_loader(f_id, p_id)  # n_id contains p_id and its neighbors
        assoc[n_id] = torch.arange(n_id.size(0), device=device)  # maps original ID to row in z
        memory, last_update, update_loss = model['memory'](n_id)
        z = model['gnn'](
            memory,
            last_update,
            edge_index,
            repeat_tensor(data.t, 2)[e_id].to(device),
            repeat_tensor(data.msg, 2)[e_id].to(device),
        )
        prod_embs = z[assoc[p_id]]
    else:
        # TODO
        src, prod, dst, t, msg = data.src, data.prod, data.dst, data.t, data.msg
        src_node_embeddings, dst_node_embeddings, prod_node_embeddings = \
                model['graphmixer'].compute_src_dst_prod_node_temporal_embeddings(src_node_ids=src,
                                                                                  dst_node_ids=dst,
                                                                                  prod_node_ids=prod,
                                                                                  node_interact_times=t,
                                                                                  neighbor_loader=neighbor_loader)
        prod_embs = prod_node_embeddings # TODO: check if this is prob_embs for graphmixer? 
    return prod_embs

    
def train(model, optimizer, neighbor_loader, data, data_loader, device, 
          loss_name='ce-softmax', amt_loss_weight=1, inv_loss_weight=1, 
          update_params=True, ns_samples=6, neg_sampler=None, split_mode="val",
          num_firms=None, num_products=None, use_prev_sampling=False):
    """
    Training procedure.
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

    neighbor_loader.reset_state()  # Start with an empty graph.
    if MODEL_NAME == 'TGNPL':
        model['memory'].reset_state()  # Start with a fresh memory.
    if 'inventory' in model:
        model['inventory'].reset()
    
    loss_types = ['loss', 'link_loss', 'amt_loss', 'inv_loss', 'debt_loss', 'consump_rwd', 'update_loss']
    total_loss_dict = {l:0 for l in loss_types}
    total_num_events = 0
    for batch in tqdm(data_loader):        
        batch = batch.to(device)
        if update_params:
            optimizer.zero_grad()
            
        y_link_pred, y_amt_pred, update_loss = get_y_pred_for_batch(
            batch, model, neighbor_loader, data, device, ns_samples=ns_samples, neg_sampler=neg_sampler, 
            split_mode=split_mode, num_firms=num_firms, num_products=num_products, use_prev_sampling=use_prev_sampling,
            include_inventory=(MODEL_NAME=='INVENTORY'))
        assert y_link_pred.size(1) == (3*ns_samples)+1

        if loss_name == 'ce-softmax':
            criterion = torch.nn.CrossEntropyLoss()  # softmax then cross entropy
            target = torch.zeros(y_link_pred.size(0), device=device).long()  # positive always in first position
            link_loss = criterion(y_link_pred, target)
        else:
            criterion = torch.nn.BCEWithLogitsLoss()  # original loss from TGB: sigmoid then binary cross entropy
            pos_out = y_link_pred[:, :1]
            neg_out = y_link_pred[:, 1:]            
            link_loss = criterion(pos_out, torch.ones_like(pos_out))
            link_loss += criterion(neg_out, torch.zeros_like(neg_out))

        if 'amount_pred' in model:  # will be in model if args.skip_amount = False
            criterion = torch.nn.MSELoss()
            target = batch.msg[:, :1]  # amount is always first feature in msg
            amt_loss = amt_loss_weight * torch.sqrt(criterion(y_amt_pred, target))  # RMSE  
        else:
            amt_loss = 0            
        
        if 'inventory' in model:
            inv_loss, debt_loss, consump_rwd = update_inventory_and_compute_loss(
                batch, model, neighbor_loader, data, device, num_firms=num_firms, num_products=num_products)
        else:
            inv_loss, debt_loss, consump_rwd = 0, 0, 0
        
        loss = link_loss + amt_loss + update_loss
        if 'inventory' in model and model['inventory'].trainable:
            loss += (inv_loss_weight*inv_loss)
        total_loss_dict['loss'] += float(loss) * batch.num_events  # scale by batch size
        total_loss_dict['link_loss'] += float(link_loss) * batch.num_events
        total_loss_dict['amt_loss'] += float(amt_loss) * batch.num_events
        total_loss_dict['inv_loss'] += inv_loss_weight * float(inv_loss) * batch.num_events
        total_loss_dict['debt_loss'] += inv_loss_weight * float(debt_loss) * batch.num_events
        total_loss_dict['consump_rwd'] += inv_loss_weight * float(consump_rwd) * batch.num_events
        total_loss_dict['update_loss'] += float(update_loss) * batch.num_events
        total_num_events += batch.num_events
        
        # Update time-varying states with ground-truth transactions
        if MODEL_NAME == 'TGNPL':
            neighbor_loader.insert(batch.src, batch.dst, batch.prod)
            model['memory'].update_state(batch.src, batch.dst, batch.prod, batch.t, batch.msg)
        elif MODEL_NAME == 'GRAPHMIXER':
            neighbor_loader.insert(batch.src, batch.dst, batch.prod, batch.t, batch.msg)
                
        # Update model parameters with backprop
        if update_params:
            loss.backward()
            optimizer.step()
            if MODEL_NAME == 'TGNPL':
                model['memory'].detach()
            if 'inventory' in model:
                model['inventory'].detach()
    
    for l, total in total_loss_dict.items():
        total_loss_dict[l] = total / total_num_events
    return total_loss_dict


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
    for module in model.values():
        module.eval()

    total_perf_dict = {'link_pred':0, 'amount_pred':0}
    total_num_events = 0
    for batch in tqdm(data_loader):
        y_link_pred, y_amt_pred, _ = get_y_pred_for_batch(
            batch, model, neighbor_loader, data, device, neg_sampler=neg_sampler, split_mode=split_mode,
            num_firms=num_firms, num_products=num_products, use_prev_sampling=use_prev_sampling, 
            include_inventory=True)
#         print(y_link_pred.sum(axis=0))
        input_dict = {
            "y_pred_pos": y_link_pred[:, :1],
            "y_pred_neg": y_link_pred[:, 1:],
            "eval_metric": [metric]
        }
        link_perf = evaluator.eval(input_dict)[metric]  # link prediction performance
        total_perf_dict['link_pred'] += link_perf * batch.num_events
        
        if 'amount_pred' in model:  # amount prediction perforamance
            criterion = torch.nn.MSELoss()
            target = batch.msg[:, :1]  # amount is always first feature in msg
            rmse = torch.sqrt(criterion(y_amt_pred, target))  # RMSE
            total_perf_dict['amount_pred'] += float(rmse) * batch.num_events
        total_num_events += batch.num_events
        
        # Update time-varying states with ground-truth transactions
        
        if MODEL_NAME == 'TGNPL':
            neighbor_loader.insert(batch.src, batch.dst, batch.prod)
            model['memory'].update_state(batch.src, batch.dst, batch.prod, batch.t, batch.msg)
        elif MODEL_NAME == 'GRAPHMIXER':
            neighbor_loader.insert(batch.src, batch.dst, batch.prod, batch.t, batch.msg)
        if 'inventory' in model:
            update_inventory_and_compute_loss(batch, model, neighbor_loader, data, device, 
                num_firms=num_firms, num_products=num_products)  # ignore loss
    
    for p, total in total_perf_dict.items():
        total_perf_dict[p] = total / total_num_events
    return total_perf_dict


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

def parse_args():
    """
    Parse args from command line.
    """
    parser = argparse.ArgumentParser()
    # general parameters
    parser.add_argument('--model', type=str, help='Model name', default='tgnpl', choices=['tgnpl', 'graphmixer', 'inventory'])
    parser.add_argument('--dataset', type=str, help='Dataset name', default='tgbl-hypergraph')
    parser.add_argument('--skip_amount', action="store_true", help='If true, skip amount prediction')
    parser.add_argument('--sweep', action='store_true', help='Launch hyperparameter sweep')
    parser.add_argument('--num_neighbors', type=int, help='Number of neighbors to store in NeighborLoader', default=10)
    
    # TGN-PL model parameters
    parser.add_argument('--memory_name', type=str, help='Name of memory module', default='tgnpl', choices=['tgnpl', 'static'])
    parser.add_argument('--emb_name', type=str, help='Name of embedding module', default='attn', choices=['attn', 'sum', 'id'])
    parser.add_argument('--mem_dim', type=int, help='Memory dimension', default=100)
    parser.add_argument('--emb_dim', type=int, help='Embedding dimension', default=100)
    parser.add_argument('--time_dim', type=int, help='Time dimension', default=100)
    parser.add_argument('--update_penalty', type=float, help='Regularization of TGNPL memory updates by penalizing change in memory', default=1)
    parser.add_argument('--weights', type=str, help='Saved weights to initialize model with')
    
    # GraphMixer model parameters
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap')
    parser.add_argument('--token_dim_expansion_factor', type=float, default=0.5, help='token dimension expansion factor in MLPMixer')
    parser.add_argument('--channel_dim_expansion_factor', type=float, default=4.0, help='channel dimension expansion factor in MLPMixer')
    parser.add_argument('--node_features_dim', type=int, help='Node features dimension', default=10)

    # inventory module parameters
    parser.add_argument('--use_inventory', action='store_true', help='Whether to use inventory module')
    parser.add_argument('--inv_loss_weight', type=float, help='How much to weigh inventory loss; only used with use_inventory', default=0.001)
    parser.add_argument('--learn_att_direct', action='store_true', help='Whether to learn pairwise attention')
    parser.add_argument('--att_weights', type=str, help='Saved attention weights for inventory module')
    parser.add_argument('--fix_inventory', action="store_true", help='Treat inventory module as fixed, don\'t update')
    parser.add_argument('--prod_graph', type=str, default='synthetic_prod_graph.pkl')
    
    # training parameters
    parser.add_argument('--num_epoch', type=int, help='Number of epochs', default=100)
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--bs', type=int, help='Batch size', default=100)
    parser.add_argument('--batch_by_t', action="store_true", help='Batch by t instead of fixed batch size; overrides --bs')
    parser.add_argument('--tolerance', type=float, help='Early stopper tolerance', default=1e-6)
    parser.add_argument('--patience', type=float, help='Early stopper patience', default=10)
    parser.add_argument('--ignore_patience_num_epoch', type=int, default=20, help='how many epochs we run before considering patience')
    parser.add_argument('--num_run', type=int, help='Number of iteration runs', default=1)
    parser.add_argument('--num_train_days', type=int, help='How many days to use for training; used for debugging and faster training', default=-1)
    parser.add_argument('--train_on_val', action='store_true', help='Train on validation set with fixed negative sampled; used for debugging')
    parser.add_argument('--train_with_fixed_samples', action="store_true", help='Use fixed negative samples for training')
    parser.add_argument('--test_per_epoch', action="store_true", help='Evaluate MRR on test every epoch; used for debugging')
    parser.add_argument('--use_prev_sampling', action='store_true', help = "Use previous negative sampling method")
    
    # system parameters
    parser.add_argument('--wandb', action='store_true', help='Wandb support')
    parser.add_argument('--tensorboard', action='store_true', help='Tensorboard support')
    parser.add_argument('--gpu', type=int, help='Which GPU to use', default=0)
    
    try:
        args = parser.parse_args()
        defaults = parser.parse_args([])
    except:
        parser.print_help()
        sys.exit(0)
    return args, defaults, sys.argv

def get_unique_id_for_experiment(args):
    """
    Returns a unique ID for an experiment.
    """
    curr_time = f"{current_pst_time().strftime('%Y_%m_%d-%H_%M_%S')}"
    # include important parameters + time
    exp_id = f'{args.model.upper()}_{args.dataset}_{args.use_inventory}_{curr_time}'
    return exp_id

def do_hyperparameter_sweep(hyperparameters, fixed_args, gpus=None):
    """
    Helper function to launch hyperparameter sweep.
    hyperparameters is a list of dicts.
    """
    assert 'dataset' in fixed_args
    dataset = fixed_args['dataset']
    if gpus is None:
        gpus = range(torch.cuda.device_count())
    print(hyperparameters)
    for idx, hyps in enumerate(hyperparameters):
        cmd = 'nohup python -u model_experiments.py '
        out_elements = [dataset]
        for arg,val in hyps.items():
            cmd += f'--{arg} {val} '
            out_elements.append(f'{arg}={val}')
        for arg,val in fixed_args.items():
            if val is None:
                cmd += f'--{arg} '
            else:
                cmd += f'--{arg} {val} '
        cmd += f'--gpu {gpus[idx % len(gpus)]} '
        out_file = '_'.join(out_elements) + '.out'
        cmd += f'> {out_file} 2>&1 &'
        print(cmd)
        os.system(cmd)
        time.sleep(5)
            
    
# ===========================================
# == Functions to run complete experiments
# ===========================================
# Global variables
TGB_DIRECTORY = "/".join(str(__file__).split("/")[:-4])
PATH_TO_DATASETS = os.path.join(TGB_DIRECTORY, "tgb/datasets/")
NUM_FIRMS = -1
NUM_PRODUCTS = -1
AMT_MEAN = None
AMT_STD = None
    
def set_up_model(args, data, device, num_firms=None, num_products=None):
    """
    Initialize model modules based on args.
    """
    # use global variables when arguments are not specified
    if num_firms is None:
        num_firms = NUM_FIRMS
    if num_products is None:
        num_products = NUM_PRODUCTS
    num_nodes = num_firms + num_products

    model = {}
    assert MODEL_NAME in {'TGNPL', 'GRAPHMIXER', 'INVENTORY'}
    if MODEL_NAME == 'TGNPL':
        # initialize memory module
        if args.memory_name == 'tgnpl':
            memory = TGNPLMemory(
                num_nodes = num_nodes,
                num_prods = num_products,
                raw_msg_dim = data.msg.size(-1),
                memory_dim = args.mem_dim,
                time_dim = args.time_dim,
                message_module=TGNPLMessage(data.msg.size(-1), args.mem_dim, args.time_dim),
                aggregator_module=MeanAggregator(),
                update_penalty=args.update_penalty,
            ).to(device)
        else:
            assert args.memory_name == 'static'
            memory = StaticMemory(num_nodes = num_nodes, memory_dim = args.mem_dim, time_dim = args.time_dim).to(device)
        model['memory'] = memory

        # initialize GNN
        if args.emb_name == 'attn':
            gnn = GraphAttentionEmbedding(
                in_channels=args.mem_dim,
                out_channels=args.emb_dim,
                msg_dim=data.msg.size(-1),
                time_enc=memory.time_enc,
            ).to(device)
        elif args.emb_name == 'sum':
            gnn = GraphSumEmbedding(
                in_channels=args.mem_dim,
                out_channels=args.emb_dim
            ).to(device)
        else:
            assert args.emb_name == 'id'
            gnn = IdentityEmbedding().to(device)
        model['gnn'] = gnn

        # initialize link predictor
        emb_dim = args.mem_dim if args.emb_name == 'id' else args.emb_dim
        link_pred = DecoderTGNPL(in_channels=emb_dim).to(device)
        model['link_pred'] = link_pred

        # put together in model and initialize optimizer
        all_params = set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters())

    elif MODEL_NAME == 'GRAPHMIXER':
        # initialize graphmixer layer
        edge_feat_dim = data.msg.shape[1]
        graphmixer = GraphMixer(num_nodes=num_nodes, edge_feat_dim=edge_feat_dim,
                                time_feat_dim=args.time_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, 
                                node_feat_dim=args.node_features_dim, dropout=args.dropout, time_gap=args.time_gap, 
                                token_dim_expansion_factor=args.token_dim_expansion_factor, 
                                channel_dim_expansion_factor=args.channel_dim_expansion_factor).to(device)
        model['graphmixer'] = graphmixer

        # initialize decoder layer
        emb_dim = graphmixer.node_feat_dim
        link_pred = DecoderTGNPL(in_channels=emb_dim).to(device) # same as MergeLayer, but without .sigmoid() at the outmost layer
        model['link_pred'] = link_pred

        # put together in model and initialize optimizer
        all_params = set(model['graphmixer'].parameters()) | set(model['link_pred'].parameters())
        
    else:
        all_params = set()
        emb_dim = None
    
    # add amount predictor if required
    if not args.skip_amount:
        amt_pred = DecoderTGNPL(in_channels=emb_dim).to(device)  # can use the same architecture
        model['amount_pred'] = amt_pred

    # add inventory module if required
    if MODEL_NAME == 'INVENTORY' or args.use_inventory:
        if args.att_weights is not None:
            with open(args.att_weights, 'rb') as f:
                att_weights = pickle.load(f)
            print(f'Initializing inventory module with {args.att_weights} (trainable = {not args.fix_inventory})')
        else:
            att_weights = None
        inventory = TGNPLInventory(
            num_firms = num_firms,
            num_prods = num_products,
            learn_att_direct = args.learn_att_direct,
            device = device,
            emb_dim = emb_dim,
            init_weights = att_weights,
            trainable = not args.fix_inventory,
        ).to(device)
        model['inventory'] = inventory
        all_params = all_params | set(model['inventory'].parameters())
    
    if len(all_params) > 0:
        optimizer = torch.optim.Adam(all_params,lr=args.lr)
    else:
        optimizer = None
    return model, optimizer


def set_up_data(args, data, dataset):
    """
    Normalize edge features; split data into train, val, test.
    """
    global AMT_MEAN, AMT_STD
    raw_amts = torch.clone(data.msg[:, 0])
    nan_count = torch.isnan(raw_amts).sum()
    zero_count = len(raw_amts)-torch.count_nonzero(raw_amts)
    print('Raw amounts: %.3f are nan, %.3f are 0' % (nan_count/len(raw_amts), zero_count/len(raw_amts)))
    if MODEL_NAME == 'INVENTORY' or args.use_inventory:
        data.amt = raw_amts  # need to store raw amounts for inventory module
    
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
        
    if args.batch_by_t:
        train_loader = TimeSpecificDataLoader(train_data)
        val_loader = TimeSpecificDataLoader(val_data)
        test_loader = TimeSpecificDataLoader(test_data)
    else:
        train_loader = TemporalDataLoader(train_data, batch_size=args.bs)
        val_loader = TemporalDataLoader(val_data, batch_size=args.bs)
        test_loader = TemporalDataLoader(test_data, batch_size=args.bs)

    return train_loader, val_loader, test_loader
    
def run_experiment(args):
    """
    Run a complete experiment.
    """
    global MODEL_NAME, NUM_FIRMS, NUM_PRODUCTS  # add this to modify global variables
    print('Args:', args)
    start_overall = timeit.default_timer()
    MODEL_NAME = args.model.upper()
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
    edge_feat_means = torch.mean(data.msg, axis=0)  # check that standard scaling worked
    assert torch.isclose(edge_feat_means, torch.zeros_like(edge_feat_means).to(device), atol=1e-5).all(), edge_feat_means
    if args.train_on_val:
        print('Warning: ignoring train set, training on validation set and its fixed negative samples')
    if args.train_with_fixed_samples:
        print('Using fixed negative samples for train')
    
    # Initialize neighbor loader
    if MODEL_NAME in {'TGNPL', 'INVENTORY'}:
        neighbor_loader = LastNeighborLoaderTGNPL(num_nodes, size=args.num_neighbors, device=device)
    elif MODEL_NAME == 'GRAPHMIXER':
        neighbor_loader = LastNeighborLoaderGraphmixer(num_nodes, num_neighbors=args.num_neighbors, 
                              time_gap=args.time_gap, edge_feat_dim=data.msg.shape[1], device=device)

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
        results_filename = f'{results_dir}/{exp_id}_{run_idx}_results.json'
        save_model_id = f'{exp_id}_{run_idx}'
        early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_id, 
                                         tolerance=args.tolerance, patience=args.patience,
                                         ignore_patience_num_epoch=args.ignore_patience_num_epoch)
        
        # Initialize model
        model, opt = set_up_model(args, data, device)
        if args.weights is not None:
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
                raise Exception('Failed to initialize model with weights')

        # Load ground-truth production functions, if they are provided
        if 'inventory' in model and args.prod_graph is not None:
            with open(args.prod_graph, 'rb') as f:
                prod_graph, products = pickle.load(f)
                prod2idx = {p:i for i,p in enumerate(products)}
            assert len(products) == NUM_PRODUCTS
        else:
            prod_graph = None

        # ==================================================== Train & Validation
        train_loss_list = []
        val_link_perf_list = []
        val_amt_perf_list = []
        test_link_perf_list = []
        test_amt_perf_list = []
        
        start_train_val = timeit.default_timer()
        for epoch in range(1, args.num_epoch + 1):
            # train
            start_epoch_train = timeit.default_timer()
            if args.train_on_val:
                # used for debugging: train on validation, with fixed negative samples
                dataset.load_val_ns()  # load val negative samples
                loss_dict = train(model, opt, neighbor_loader, data, val_loader, device, 
                    neg_sampler=neg_sampler, split_mode="val", use_prev_sampling=args.use_prev_sampling,
                    inv_loss_weight=args.inv_loss_weight, update_params=(opt is not None))
                # reset for beginning of val
                neighbor_loader.reset_state()
                if MODEL_NAME == 'TGNPL':
                    model['memory'].reset_state()  
                if 'inventory' in model:
                    model['inventory'].reset()
            elif args.train_with_fixed_samples:
                # train on fixed negative samples instead of randomly drawn per epoch
                dataset.load_train_ns()  # load train negative samples
                loss_dict = train(model, opt, neighbor_loader, data, train_loader, device, 
                    neg_sampler=neg_sampler, split_mode="train", use_prev_sampling=args.use_prev_sampling,
                    inv_loss_weight=args.inv_loss_weight, update_params=(opt is not None))
                # Don't reset since val is a continuation of train
            else:
                loss_dict = train(model, opt, neighbor_loader, data, train_loader, device,
                    inv_loss_weight=args.inv_loss_weight, update_params=(opt is not None))
                # Don't reset since val is a continuation of train
            if 'inventory' in model:
                model['inventory'].update_to_new_timestep()  # train ends at the end of t
            time_train = timeit.default_timer() - start_epoch_train
            loss_str = ', '.join([f'{s}: {l:.4f}' for s,l in loss_dict.items()])
            print(f'Epoch: {epoch:02d}, {loss_str}; Training elapsed Time (s): {time_train:.4f}')
            train_loss_list.append(loss_dict['loss'])

            # validation
            start_val = timeit.default_timer()
            dataset.load_val_ns()  # load val negative samples
            val_dict = test(model, neighbor_loader, data, val_loader, neg_sampler, evaluator,
                            device, split_mode="val", metric=metric, use_prev_sampling=args.use_prev_sampling)
            if 'inventory' in model:
                model['inventory'].update_to_new_timestep()  # val ends at the end of t
            print(f"\tValidation {metric}: {val_dict['link_pred']:.4f}, RMSE: {val_dict['amount_pred']:.4f}")
            val_link_perf_list.append(val_dict['link_pred'])
            val_amt_perf_list.append(val_dict['amount_pred'])
                
            if prod_graph is not None:  # this means model has inventory AND production functions were provided
                if model['inventory'].learn_att_direct:
                    pred_mat = model['inventory']._get_prod_attention().cpu().detach().numpy()
                else:
                    prod_embs = get_product_embeddings(model, neighbor_loader, num_firms, num_products, data, device)
                    pred_mat = model['inventory']._get_prod_attention(prod_emb=prod_embs).cpu().detach().numpy()
                prod_map = mean_average_precision(prod_graph, prod2idx, pred_mat)
                print(f'\tProduction function MAP: {prod_map:.4f}')
            time_val = timeit.default_timer() - start_val
            print(f"\tValidation: Elapsed time (s): {time_val: .4f}")
            
            # log metrics
            log_dict = loss_dict.copy()
            log_dict[f'val_link_pred_{metric}'] = val_dict['link_pred']
            log_dict['val_amount_pred_RMSE'] = val_dict['amount_pred']
            log_dict['elapsed_time_train'] = time_train
            log_dict['elapsed_time_val'] = time_val
            if args.tensorboard:
                for k, v in log_dict:
                    writer.add_scalar(k, v, epoch)            
            if args.wandb:
                wandb.log(log_dict)
            
            if args.test_per_epoch:  # evaluate on test per epoch - used for debugging
                start_test = timeit.default_timer()
                dataset.load_test_ns()  # load test negative samples
                test_dict = test(model, neighbor_loader, data, test_loader, neg_sampler, evaluator,
                                       device, split_mode="test", metric=metric, use_prev_sampling=args.use_prev_sampling)
                time_test = timeit.default_timer() - start_test
                print(f"\tTest {metric}: {test_dict['link_pred']:.4f}, RMSE: {test_dict['amount_pred']:.4f}")
                print(f"\tTest: Elapsed time (s): {time_test: .4f}")
                test_link_perf_list.append(test_dict['link_pred'])
                test_amt_perf_list.append(test_dict['amount_pred'])
            
            # save results after each epoch
            save_results({'model': MODEL_NAME,
                  'data': args.dataset,
                  'run': run_idx,
                  'seed': args.seed,
                  'train loss': train_loss_list,
                  f'val {metric}': val_link_perf_list,
                  'val RMSE': val_amt_perf_list,
                  f'test {metric}': test_link_perf_list,
                  'test RMSE': test_amt_perf_list}, 
                  results_filename, replace_file=True)

            # check if best on val so far, save if so, stop if no improvement observed for a while
            if early_stopper.step_check(val_dict['link_pred'], model):
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
        if args.test_per_epoch:
            # we already ran test every epoch, just get test metric from best val
            best_epoch = np.argmax(val_link_perf_list)
            test_dict = {'link_pred': test_link_perf_list[best_epoch],
                         'amount_pred': test_amt_perf_list[best_epoch]}
        else:
            early_stopper.load_checkpoint(model)  # load the best model
            dataset.load_test_ns()  # load test negatives
            start_test = timeit.default_timer()
            test_dict = test(model, neighbor_loader, data, test_loader, neg_sampler, evaluator,
                                    device, split_mode="test", metric=metric, 
                                    use_prev_sampling=args.use_prev_sampling)
            save_results({'model': MODEL_NAME,
                  'data': args.dataset,
                  'run': run_idx,
                  'seed': args.seed,
                  'train loss': train_loss_list,
                  f'val {metric}': val_link_perf_list,
                  'val RMSE': val_amt_perf_list,
                  f'test {metric}': test_dict['link_pred'],
                  'test RMSE': test_dict['amount_pred']}, 
                  results_filename, replace_file=True)

        print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
        print(f"\tTest {metric}: {test_dict['link_pred']:.4f}, RMSE: {test_dict['amount_pred']:.4f}")
        test_time = timeit.default_timer() - start_test
        print(f"\tTest: Elapsed Time (s): {test_time: .4f}")
        if args.tensorboard:
            hparam_dict = vars(args)
            hparam_dict.update({"num_neighbors": args.num_neighbors,
                                "model_name": MODEL_NAME, 
                                "metric": metric})
            metric_dict = {
                'best_epoch': early_stopper.best_epoch,
                f'test {metric}': test_dict['link_pred'],
                'test RMSE': test_dict['amount_pred']
            }
            writer.add_hparams(hparam_dict, metric_dict)
            writer.add_scalar("elapsed_time_test", test_time)
        if args.wandb:
            wandb.summary["metric"] = metric
            wandb.summary["best_epoch"] = early_stopper.best_epoch
            wandb.summary[f'test {metric}'] = test_dict['link_pred']
            wandb.summary['test RMSE'] = test_dict['amount_pred']
            wandb.summary["elapsed_time_test"] = test_time

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
    args, defaults, _ = parse_args()
    if args.sweep:
        hyps_list = []
#         for num_neighbors in [5, 20, 50]:
#             hyperparameters = {'num_neighbors': num_neighbors}
#             hyps_list.append(hyperparameters)
#         fixed_args = {'model': 'tgnpl', 
#                       'dataset': 'tgbl-hypergraph_synthetic_std', 
#                       'train_with_fixed_samples': None,
#                       'mem_dim': 500,
#                       'emb_dim': 500,
#                       'bs': 30}
        for dim in [100, 1000, 1500]:
            hyperparameters = {'node_features_dim': dim}
            hyps_list.append(hyperparameters)
        fixed_args = {'model': 'graphmixer',
                      'dataset': 'tgbl-hypergraph_synthetic_std',
                      'train_with_fixed_samples': None,
                      'bs': 30,
                      'time_gap': 10}
        gpus = [5,6,7]
        do_hyperparameter_sweep(hyps_list, fixed_args, gpus=gpus)
#     labeling_args = compare_args(args, defaults)
    else:
        run_experiment(args)
