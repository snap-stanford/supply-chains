import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json

from models.EdgeBank import edge_bank_link_prediction
from utils.metrics import get_link_prediction_metrics, get_node_classification_metrics
from utils.utils import set_random_seed
from utils.utils import NeighborSampler
from utils.DataLoader import Data

# additional required imports
from tgb.linkproppred.evaluate import Evaluator


def query_pred_edge_batch(model_name: str, model: nn.Module, 
                          src_node_ids: int, dst_node_ids: int, node_interact_times: float, edge_ids: int,
                          edges_are_positive: bool, num_neighbors: int, time_gap: int):
    """
    query the prediction probabilities for a batch of edges
    """
    if model_name in ['TGAT', 'CAWN', 'TCL']:
        # get temporal embedding of source and destination nodes
        # two Tensors, with shape (batch_size, node_feat_dim)
        batch_src_node_embeddings, batch_dst_node_embeddings = \
            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                              dst_node_ids=dst_node_ids,
                                                              node_interact_times=node_interact_times,
                                                              num_neighbors=num_neighbors)

    elif model_name in ['JODIE', 'DyRep', 'TGN']:
        # get temporal embedding of source and destination nodes
        # two Tensors, with shape (batch_size, node_feat_dim)
        batch_src_node_embeddings, batch_dst_node_embeddings = \
            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                                dst_node_ids=dst_node_ids,
                                                                node_interact_times=node_interact_times,
                                                                edge_ids=edge_ids,
                                                                edges_are_positive=edges_are_positive,
                                                                num_neighbors=num_neighbors)

    elif model_name in ['GraphMixer']:
        # get temporal embedding of source and destination nodes
        # two Tensors, with shape (batch_size, node_feat_dim)
        batch_src_node_embeddings, batch_dst_node_embeddings = \
            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                                dst_node_ids=dst_node_ids,
                                                                node_interact_times=node_interact_times,
                                                                num_neighbors=num_neighbors,
                                                                time_gap=time_gap)

    elif model_name in ['DyGFormer']:
        # get temporal embedding of source and destination nodes
        # two Tensors, with shape (batch_size, node_feat_dim)
        batch_src_node_embeddings, batch_dst_node_embeddings = \
            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                                dst_node_ids=dst_node_ids,
                                                                node_interact_times=node_interact_times)

    else:
        raise ValueError(f"Wrong value for model_name {model_name}!")
        batch_src_node_embeddings, batch_dst_node_embeddings = None, None
        
    return batch_src_node_embeddings, batch_dst_node_embeddings



def eval_LPP_TGB(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                evaluate_data: Data,  negative_sampler: object, evaluator: Evaluator, metric: str = 'mrr',
                split_mode: str = 'test', k_value: int = 10, num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the link prediction task based on TGB NegativeSampler and Evaluator
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :param split_mode: str, specifies whether the evaluation is performed for test or validation
    :param evaluator: Evaluator, dynamic link prediction evaluator
    "param k_value: int, the desired `k` for calculation of metrics @ k
    :return:
    """
    perf_list = []

    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            pos_src_orig = batch_src_node_ids - 1
            pos_dst_orig = batch_dst_node_ids - 1
            pos_t = np.array([int(ts) for ts in batch_node_interact_times])
            neg_batch_list = negative_sampler.query_batch(pos_src_orig, pos_dst_orig, 
                                                    pos_t, split_mode=split_mode)

            
            for idx, neg_batch in enumerate(neg_batch_list):
                neg_batch = np.array(neg_batch) + 1  # due to the special data loading processing ...
                batch_neg_src_node_ids = np.array([int(batch_src_node_ids[idx]) for _ in range(len(neg_batch))])
                batch_neg_dst_node_ids = np.array(neg_batch)
                batch_neg_node_interact_times = np.array([batch_node_interact_times[idx] for _ in range(len(neg_batch))])

                # negative edges
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    query_pred_edge_batch(model_name=model_name, model=model, 
                          src_node_ids=batch_neg_src_node_ids, dst_node_ids=batch_neg_dst_node_ids, 
                          node_interact_times=batch_neg_node_interact_times, edge_ids=None,
                          edges_are_positive=False, num_neighbors=num_neighbors, time_gap=time_gap)
                
                # one positive edge
                batch_pos_src_node_embeddings, batch_pos_dst_node_embeddings = \
                    query_pred_edge_batch(model_name=model_name, model=model, 
                          src_node_ids=np.array([batch_src_node_ids[idx]]), dst_node_ids=np.array([batch_dst_node_ids[idx]]), 
                          node_interact_times=np.array([batch_node_interact_times[idx]]), edge_ids=np.array([batch_edge_ids[idx]]),
                          edges_are_positive=True, num_neighbors=num_neighbors, time_gap=time_gap)

                
                # get positive and negative probabilities
                positive_probabilities = model[1](input_1=batch_pos_src_node_embeddings, 
                                                  input_2=batch_pos_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, 
                                                  input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

                # compute MRR
                input_dict = {
                    'y_pred_pos': np.array(positive_probabilities.cpu()),
                    'y_pred_neg': np.array(negative_probabilities.cpu()),
                    'eval_metric': [metric]
                }
                perf_list.append(evaluator.eval(input_dict)[metric])
            
    avg_perf_metric = float(np.mean(np.array(perf_list)))

    return avg_perf_metric


