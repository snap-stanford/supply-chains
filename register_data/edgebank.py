"""
also termed the "historical baseline" (ranks candidate edges by frequency or binary existence in the training set)
"""

import argparse
import pandas as pd
import pickle
import json
from tqdm import tqdm
import os 
import multiprocessing as mp
import matplotlib.pyplot as plt 
import numpy as np

def get_args():
    #make these into argparse
    parser = argparse.ArgumentParser(description='Extracting graph data from the transactions in logistic_data')
    parser.add_argument('--dataset_name', nargs='?', default = "tgbl-supplychains", help = "name to be assigned to dataset")
    parser.add_argument('--dir', nargs='?', default = "./tgb_data", help = "directory to save data")
    parser.add_argument('--use_freq', action='store_true', help = "use frequency instead of binary existence to rank candidates")
    args = parser.parse_args()
    return args

def calc_mean_MRR(ranks):
    if (len(ranks) == 0): raise ValueError("len(ranks) >= 1")
    total = 0
    for rank in ranks:
        total += 1 / rank 
    return total / len(ranks)

def search_frequency(query, frequency_dict):
    if (query in frequency_dict):
        return frequency_dict[query]
    return 0

def count_node_matches(edge1, edge2):
    return sum([edge1[j] == edge2[j] for j in range(len(edge1))])

def get_hist_breakdown(train_edges_set, pos_edge, neg_edges): 
    num_hist = 0
    num_loose_hist = 0
    num_rand = 0
    for neg_edge in neg_edges: 
        if (neg_edge in train_edges_set):
            num_rand += 1
            continue 
        num_node_matches = count_node_matches(pos_edge, neg_edge)
        if (num_node_matches == 2):
            num_hist += 1
        else:
            num_loost_hist += 1

    return num_hist, num_loose_hist, num_rand
            
#def get_prop_hist(train_frequency_dict, pos_edge, ):
#    num_hist = len([dest for dest in negative_dest if (src, dest) in train_freq])
#    return num_hist / len(negative_dest)

def get_positive_rank(train_frequency_dict, pos_edge, neg_edges, isBinary = False):
    all_edges = [pos_edge] + list(neg_edges)
    frequencies = [search_frequency(edge, train_frequency_dict) for edge in all_edges]
    if (isBinary):
        frequencies = [1 if freq > 0 else 0 for freq in frequencies]
    frequency_of_pos_edge = frequencies[0]
    sorted_frequencies = sorted(frequencies, reverse = True)

    ranks = [idx + 1 for idx, freq in enumerate(sorted_frequencies) if freq == frequency_of_pos_edge]
    return sum(ranks) / len(ranks)

    #return np.random.choice(ranks) 
    
def run_hyperedgebank(dataset_name, dir, use_freq):

    with open(os.path.join(dir,f'{dataset_name}_val_ns.pkl'), 'rb') as file:
        val_ns = pickle.load(file)
        val_ts = sorted(list(set([ts for s,d,p,ts in val_ns.keys()])))
        val_min_ts, val_max_ts = min(val_ts), max(val_ts)
        
    with open(os.path.join(dir, f'{dataset_name}_test_ns.pkl'), 'rb') as file:
        test_ns = pickle.load(file)
        test_ts = sorted(list(set([ts for s,d,p,ts in test_ns.keys()])))
        test_min_ts, test_max_ts = min(test_ts), max(test_ts)
    
    with open(os.path.join(dir, f'{dataset_name}_meta.json'),"r") as file:
        metadata = json.load(file)
        product_min_id, id2entity, train_max_ts, val_max_ts, test_max_ts = (metadata[key] for key in 
                            ["product_threshold", "id2entity", "train_max_ts", "val_max_ts", "test_max_ts"])
        
    df_edgelist = pd.read_csv(os.path.join(dir,f'{dataset_name}_edgelist.csv'))
    df_edgelist_train = df_edgelist[df_edgelist["ts"] <= train_max_ts]
    edgelist_train = [list(df_edgelist_train[column]) for column in ["source","target","product","ts"]]
    
    #get frequency count of edges during training 
    train_edge_frequency = {}
    for s, d, p, t in zip(*edgelist_train):
        if (s,d,p) in train_edge_frequency: 
            train_edge_frequency[(s,d,p)] += 1
        else: 
            train_edge_frequency[(s,d,p)] = 1

    val_ranks, val_hist = [], {ts: {"hist": 0, "loose_hist": 0, "rand": 0} for ts in val_ts}
    for s, d, p, t in tqdm(val_ns): 
        negative_samples = [(s,d,p) for s,d,p in val_ns[(s,d,p,t)].astype(int)]
        rank = get_positive_rank(train_edge_frequency, (s,d,p), negative_samples, not use_freq)
        val_ranks.append(rank) 

    test_ranks, test_hist = [], {ts: {"hist": 0, "loose_hist": 0, "rand": 0} for ts in val_ts}
    for s, d, p, t in tqdm(test_ns): 
        negative_samples = [(s,d,p) for s,d,p in test_ns[(s,d,p,t)].astype(int)]
        rank = get_positive_rank(train_edge_frequency, (s,d,p), negative_samples, not use_freq)
        test_ranks.append(rank) 

    val_mean_MRR = calc_mean_MRR(val_ranks)
    test_mean_MRR = calc_mean_MRR(test_ranks)
    
    MRR_dict = {"val": val_mean_MRR, "test": test_mean_MRR}
    return MRR_dict

if __name__ == "__main__":
    args = get_args()
    MRR_dict = run_hyperedgebank(args.dataset_name, args.dir, args.use_freq)
    print(MRR_dict)
        
        
    """
    
    src, dest, ts in val_ns: 
        negative_dest = val_ns[(src, dest, ts)].astype(int)
        rank = get_positive_rank(train_freq, src, dest, negative_dest)
        val_hist[ts].append(get_prop_hist(train_freq, src, negative_dest))
        val_ranks.append(rank)
    """

    


    

    

    
    
    