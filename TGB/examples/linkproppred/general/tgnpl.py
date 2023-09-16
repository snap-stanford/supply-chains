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

# ===========================================
# == Main functions to train and test model
# ===========================================
def _get_y_pred_for_batch(batch, model, neighbor_loader, data, device,
                          ns_samples=6, neg_sampler=None, split_mode="val",
                          num_firms=None, num_products=None):
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
    else:
        # use fixed negatives
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

    src, dst, prod = batch_src.flatten(), batch_dst.flatten(), batch_prod.flatten()  # row-wise flatten
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
    y_pred = y_pred.reshape(bs, num_samples)
    return y_pred
    

def train(model, optimizer, neighbor_loader, data, data_loader, device, 
          loss_name='ce-softmax', update_params=True, 
          ns_samples=6, neg_sampler=None, split_mode="val",
          num_firms=None, num_products=None):
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
    Returns:
        total loss, logits loss (from dynamic link prediction), inventory loss
    """
    assert loss_name in ['ce-softmax', 'bce-logits']    
    if update_params:
        model['memory'].train()
        model['gnn'].train()
        model['link_pred'].train()
    else:
        model['memory'].eval()
        model['gnn'].eval()
        model['link_pred'].eval()

    model['memory'].reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.
    
    if loss_name == 'ce-softmax':
        # softmax then cross entropy
        criterion = torch.nn.CrossEntropyLoss() 
    else:
        # sigmoid then binary cross entropy, used in TGB
        criterion = torch.nn.BCEWithLogitsLoss()
    
    total_loss, total_logits_loss, total_inv_loss, total_num_events = 0, 0, 0, 0
    for batch in tqdm(data_loader):        
        batch = batch.to(device)
        if update_params:
            optimizer.zero_grad()

        y_pred = _get_y_pred_for_batch(batch, model, neighbor_loader, data, device,
                                       ns_samples=ns_samples, neg_sampler=neg_sampler, split_mode=split_mode,
                                       num_firms=num_firms, num_products=num_products)
        
        if loss_name == 'ce-softmax':
            target = torch.zeros(y_pred.size(0), device=device).long()  # positive always in first position
            logits_loss = criterion(y_pred, target)
        else:
            # original loss from TGB
            assert y_pred.size(1) == (3*ns_samples)+1
            pos_out = y_pred[:, :1]
            neg_out = y_pred[:, 1:]            
            logits_loss = criterion(pos_out, torch.ones_like(pos_out))
            logits_loss += criterion(neg_out, torch.zeros_like(neg_out))
        
        inv_loss = 0  # TODO
        loss = logits_loss + inv_loss
        total_loss += float(loss) * batch.num_events  # scale by batch size
        total_logits_loss += float(logits_loss) * batch.num_events
        total_inv_loss += float(inv_loss) * batch.num_events
        total_num_events += batch.num_events
        
        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(batch.src, batch.dst, batch.prod, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst, batch.prod)
        
        # Update model parameters with backprop
        if update_params:
            loss.backward()
            optimizer.step()
            model['memory'].detach()

    return total_loss / total_num_events, total_logits_loss / total_num_events, total_inv_loss / total_num_events

@torch.no_grad()
def test(model, neighbor_loader, data, data_loader, neg_sampler, evaluator, device,
         split_mode="val", metric="mrr", num_firms=None, num_products=None):
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
    Returns:
        perf_metric: the result of the performance evaluaiton
    """
    assert split_mode in ['val', 'test']
    assert metric in ['mrr', 'hits@']
        
    model['memory'].eval()
    model['gnn'].eval()
    model['link_pred'].eval()

    perf_list = []
    batch_size = []
    
    for batch in tqdm(data_loader):
        y_pred = _get_y_pred_for_batch(batch, model, neighbor_loader, data, device,
                                       neg_sampler=neg_sampler, split_mode=split_mode,
                                       num_firms=num_firms, num_products=num_products)
        input_dict = {
            "y_pred_pos": y_pred[:, :1],
            "y_pred_neg": y_pred[:, 1:],
            "eval_metric": [metric]
        }
        perf_list.append(evaluator.eval(input_dict)[metric])
        batch_size.append(y_pred.size(0))

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(batch.src, batch.dst, batch.prod, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst, batch.prod)
    
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

def get_tgnpl_args():
    """
    Parse args from command line.
    """
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
    parser.add_argument('--num_run', type=int, help='Number of iteration runs', default=1, choices=[1])
    parser.add_argument('--wandb', type=bool, help='Wandb support', default=False)
    parser.add_argument('--bipartite', type=bool, help='Whether to use bipartite graph', default=False)
    parser.add_argument('--memory_name', type=str, help='Name of memory module', default='tgnpl', choices=['tgnpl', 'static'])
    parser.add_argument('--use_inventory', type=bool, help='Whether to use inventory in TGNPL memory', default=False)
    parser.add_argument('--debt_penalty', type=float, help='Debt penalty weight for calculating TGNPL memory inventory loss', default=0)
    parser.add_argument('--consum_rwd', type=float, help='Consumption reward weight for calculating TGNPL memory inventory loss', default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--weights', type=str, default='', help='Saved weights to initialize model with')
    
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
        id_elements.append(str(getattr(args, arg)))
    id_elements.append(curr_time)
    return '_'.join(id_elements)

    
# ===========================================
# == Functions to run complete experiments
# ===========================================
# Global variables
MODEL_NAME = 'TGNPL'
NUM_NEIGHBORS = 10
PATH_TO_DATASETS = f'/lfs/turing1/0/{os.getlogin()}/supply-chains/TGB/tgb/datasets/'
NUM_FIRMS = -1
NUM_PRODUCTS = -1    
    
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

    # initialize memory module
    mem_out = args.mem_dim+num_products if args.use_inventory else args.mem_dim
    if args.memory_name == 'tgnpl':
        memory = TGNPLMemory(
            use_inventory = args.use_inventory,
            num_nodes = num_nodes,
            num_prods = num_products,
            raw_msg_dim = data.msg.size(-1),
            state_dim = args.mem_dim,
            time_dim = args.time_dim,
            message_module=TGNPLMessage(data.msg.size(-1), mem_out, args.time_dim),
            aggregator_module=MeanAggregator(),
            debt_penalty=args.debt_penalty,
            consumption_reward=args.consum_rwd,
        ).to(device)
    else:
        assert args.memory_name == 'static'
        assert not args.use_inventory
        memory = StaticMemory(num_nodes = num_nodes, memory_dim = args.mem_dim, time_dim = args.time_dim).to(device)

    # initialize GNN
    gnn = GraphAttentionEmbedding(
        in_channels=mem_out,
        out_channels=args.emb_dim,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device)

    # initialize decoder
    link_pred = LinkPredictorTGNPL(in_channels=args.emb_dim).to(device)

    # put together in model and initialize optimizer
    model = {'memory': memory,
             'gnn': gnn,
             'link_pred': link_pred}
    optimizer = torch.optim.Adam(
        set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters()),
        lr=args.lr,
    )
    return model, optimizer
    
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
        wandb.summary["num_neighbors"] = NUM_NEIGHBORS
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
    dataset = PyGLinkPropPredDatasetHyper(name=args.dataset, root="datasets")
    metric = dataset.eval_metric
    neg_sampler = dataset.negative_sampler
    evaluator = Evaluator(name=args.dataset)
    # split into train/val/test
    data = dataset.get_TemporalData().to(device)
    train_loader = TemporalDataLoader(data[dataset.train_mask], batch_size=args.bs)
    val_loader = TemporalDataLoader(data[dataset.val_mask], batch_size=args.bs)
    test_loader = TemporalDataLoader(data[dataset.test_mask], batch_size=args.bs)
    
    # Initialize model
    model, opt = set_up_model(args, data, device)
    if args.weights != "":
        print('Initializing model with weights from', args.weights)
        model_path = os.path.join(save_model_dir, args.weights)
        assert os.path.isfile(model_path)
        saved_model = torch.load(model_path)
        for module_name, module in model.items():
            module.load_state_dict(saved_model[module_name])
    
    # Initialize neighbor loader
    neighbor_loader = LastNeighborLoaderTGNPL(num_nodes, size=NUM_NEIGHBORS, device=device)

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
            loss, logits_loss, inv_loss = train(model, opt, neighbor_loader, data, train_loader, device)
            time_train = timeit.default_timer() - start_epoch_train
            print(
                f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Logits_Loss: {logits_loss:.4f}, Inv_Loss: {inv_loss:.4f}, Training elapsed Time (s): {time_train: .4f}"
            )
            train_loss_list.append(float(loss))

            # validation
            start_val = timeit.default_timer()
            perf_metric_val = test(model, neighbor_loader, data, val_loader, neg_sampler, evaluator,
                                   device, split_mode="val", metric=metric)
            time_val = timeit.default_timer() - start_val
            print(f"\tValidation {metric}: {perf_metric_val: .4f}")
            print(f"\tValidation: Elapsed time (s): {time_val: .4f}")
            val_perf_list.append(perf_metric_val)

            # log metric to wandb
            if args.wandb:
                wandb.log({"loss": loss, 
                           "logits_loss": logits_loss,
                           "inv_loss": inv_loss,
                           "perf_metric_val": perf_metric_val, 
                           "elapsed_time_train": TIME_TRAIN, 
                           "elapsed_time_val": TIME_VAL
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
                                device, split_mode="test", metric=metric)

        print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
        print(f"\tTest: {metric}: {perf_metric_test: .4f}")
        test_time = timeit.default_timer() - start_test
        print(f"\tTest: Elapsed Time (s): {test_time: .4f}")
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
    if args.wandb:
        wandb.finish()
        

if __name__ == "__main__":
    # Parse parameters
    args, defaults, _ = get_tgnpl_args()
    labeling_args = compare_args(args, defaults)
    run_experiment(args)