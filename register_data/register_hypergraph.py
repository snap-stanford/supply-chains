"""
this file morphs the transaction-level data into the TGB hypergraph format, including the edge list, sampled negatives 
for the train/val/test splits, and supplementary metadata (e.g., mapping fom node IDs to firm & product names). 

SAMPLE USAGE (from root directory of repo):
python register_data/register_hypergraph.py --ARGS

"""

import pandas as pd
import argparse
import numpy as np
import pickle 
import json
from tqdm import tqdm
import os
import multiprocessing as mp

def get_args():
    #make these into argparse
    parser = argparse.ArgumentParser(description='Extracting graph data from the transactions in logistic_data')
    parser.add_argument('--csv_file', nargs='?', default = "../hitachi-supply-chains/temporal_graph/storage/daily_transactions_2021.csv", help = "path to CSV file with transactions")
    parser.add_argument('--dataset_name', nargs='?', default = "tgbl-supplychains", help = "name to be assigned to dataset")
    parser.add_argument('--metric', nargs='?', default = "total_amount", help = "either total amount (in USD), which is default, or weight")
    parser.add_argument('--dir', nargs='?', default = "./tgb_data", help = "directory to save data")
    parser.add_argument('--logscale', action='store_true', help = "if true, apply logarithm to edge weights")
    parser.add_argument('--workers', nargs='?', default = 10, type = int, help = "number of thread workers")
    parser.add_argument('--num_samples', nargs='?', default = 18, type = int, help = "number of negative samples per positive edge (default: 18)")
    args = parser.parse_args()
    return args

def process_csv(csv_file, metric, logscale):
    df = pd.read_csv(csv_file)
    df = df[(df["supplier_t"] != "") & (df["buyer_t"] != "") & (~df[metric].isna()) & (~df["supplier_t"].isna()) & (~df["buyer_t"].isna())]
    all_companies = list(set(df["supplier_t"]).union(set(df["buyer_t"])))
    all_products = list(set(df["hs6"]))
    
    #create map from node IDs to firm / product names, and inverse
    company2id = {value: key for key,value in enumerate(all_companies)}
    product2id = {value: key + len(all_companies) for key,value in enumerate(all_products)}
    id2entity = {key: value for key,value in enumerate(all_companies)}
    id2product = {key + len(all_companies): value for key,value in enumerate(all_products)}
    id2entity.update(id2product)
    #minimum ID for a product (to distinguish between firm & product nodes)
    product_threshold = min(list(id2product.keys()))

    #each hyperedge is comprised of a source firm, product, and target firm
    if (logscale == True):
        df[metric] = df[metric].apply(lambda value: np.log10(value + 1))
    graph = {"ts": list(df["time_stamp"]),
             "source": [company2id[firm] for firm in df["supplier_t"]],
             "target": [company2id[firm] for firm in df["buyer_t"]],
            "product": [product2id[firm] for firm in df["hs6"]],
             "weight": list(df[metric])}
    return pd.DataFrame.from_dict(graph), id2entity, product_threshold

def partition_edges(df, train_max_ts, val_max_ts, test_max_ts):
    #creates the chronological train / val / test split of the hyperedges 
    E_train = {ts: [] for ts in range(0, train_max_ts + 1)}
    E_val = {ts: [] for ts in range(train_max_ts + 1, val_max_ts + 1)}
    E_test = {ts: [] for ts in range(val_max_ts + 1, test_max_ts + 1)}
    
    df_rows = [df[row_name] for row_name in ["source","target","product","ts"]]
    for source, destination, product, ts in zip(*df_rows):
        ts, edge = int(ts), [int(source), int(destination), int(product)]
        if (ts <= train_max_ts): 
            E_train[ts].append(edge)
        elif (ts <= val_max_ts): 
            E_val[ts].append(edge)
        else:
            E_test[ts].append(edge)
    return E_train, E_val, E_test

def count_node_matches(edge1, edge2):
    return sum([edge1[j] == edge2[j] for j in range(len(edge1))])

""" simpler version 
def edge_sampler_wrapper(split):
    global edge_sampler 

    E_eval = None
    if (split == "train"):
        E_eval = E_train.copy()
    elif (split == "val"):
        E_eval = E_val.copy()
    else:
        E_eval = E_test.copy()

    def edge_sampler(key):
        pos_s, pos_d, pos_p, ts = key
        ts_eval_edges = [(source, destination, product) for source, destination, product in E_eval[ts]]
        ts_eval_edges_set = set(ts_eval_edges)

        hist, loose_hist = [], []
        for hyperedge in E_train_edges:
            if hyperedge in ts_eval_edges_set:
                continue 
            num_node_matches = count_node_matches((pos_s, pos_d, pos_p), hyperedge)
            if (num_node_matches == 2): #hist negative
                hist.append(hyperedge)
            elif (num_node_matches == 1): #loose hist negative 
                loose_hist.append(hyperedge)

        #sample historical negatives -- aim to get num_samples // 2 in total
        hist_sampled_idx = np.random.choice(range(len(hist)), size = min(num_samples // 2, len(hist)), replace = False)
        hist_sampled = [hist[j] for j in hist_sampled_idx]

        #use loose negatives if not enough historical negatives
        if (len(hist_sampled) < num_samples // 2):
            loose_hist_sampled_idx = np.random.choice(range(len(loose_hist)),
                                    size = min(num_samples // 2 - len(hist_sampled), len(loose_hist)), replace = False)
            loose_hist_sampled = [loose_hist[j] for j in loose_hist_sampled_idx]
            hist_sampled += loose_hist_sampled 
    
        # sample random negatives 
        hist_deficit = num_samples // 2 - len(hist_sampled)
        random_surplus = [0,0,0]
        for _ in range(hist_deficit):
            random_surplus[int(3 * np.random.rand())] += 1
    
        #ensure the random negatives aren't historical 
        rand_s = [s for s in L_firm if (s, pos_d, pos_p) not in ts_eval_edges_set and (s, pos_d, pos_p) not in E_train_edges]
        rand_d = [d for d in L_firm if (pos_s, d, pos_p) not in ts_eval_edges_set and (pos_s, d, pos_p) not in E_train_edges]
        rand_p = [p for p in L_products if (pos_s, pos_d, p) not in ts_eval_edges_set and (pos_s, pos_d, p) not in E_train_edges]
    
        #sample num_samples // 6 of each perturbation class, plus extras if there is a deficit of historical negatives
        rand_s_sampled = np.random.choice(rand_s, size = num_samples // 6 + random_surplus[0], replace = False)
        rand_d_sampled = np.random.choice(rand_d, size = num_samples // 6 + random_surplus[1], replace = False)
        rand_p_sampled = np.random.choice(rand_p, size = num_samples // 6 + random_surplus[2], replace = False)
        rand_s_sampled = [(s,pos_d,pos_p) for s in rand_s_sampled]
        rand_d_sampled = [(pos_s,d,pos_p) for d in rand_d_sampled]
        rand_p_sampled = [(pos_s, pos_d, p) for p in rand_p_sampled]
        rand_sampled = rand_s_sampled + rand_d_sampled + rand_p_sampled 
                              
        all_negative_samples = hist_sampled + rand_sampled
        return all_negative_samples
            
    return edge_sampler
"""

def edge_sampler_wrapper(split): #returns an edge sampler function for either the train, val, or test split 
    global edge_sampler

    E_eval = None
    if (split == "train"):
        E_eval = E_train.copy()
    elif (split == "val"):
        E_eval = E_val.copy()
    else:
        E_eval = E_test.copy()
    
    def edge_sampler(key):
        pos_s, pos_d, pos_p, ts = key 
        ts_eval_edges = [(source, destination, product) for source, destination, product in E_eval[ts]]
        ts_eval_edges_set = set(ts_eval_edges)
        
        #get historical negatives
        hist_s, hist_d, hist_p, loose_hist = [], [], [], []
        for hyperedge in E_train_edges: 
            if hyperedge in ts_eval_edges_set:
                continue 
            num_node_matches = count_node_matches((pos_s, pos_d, pos_p), hyperedge)
            if (num_node_matches == 2): #hist negative, check the perturbed node
                if (hyperedge[0] != pos_s): #source perturbed
                    hist_s.append(hyperedge)
                elif (hyperedge[1] != pos_d): #destination perturbed
                    hist_d.append(hyperedge)
                else: #product perturbed 
                    hist_p.append(hyperedge)
            elif (num_node_matches == 1): #loose hist negative 
                loose_hist.append(hyperedge)
        hist_candidates = hist_s + hist_d + hist_p
    
        #sample historical negatives -- first, try to get num_samples // 6 of each perturbation class
        hist_s_sampled_idx = np.random.choice(range(len(hist_s)), size = min(num_samples // 6, len(hist_s)), replace = False)
        hist_d_sampled_idx = np.random.choice(range(len(hist_d)), size = min(num_samples // 6, len(hist_d)), replace = False)
        hist_p_sampled_idx = np.random.choice(range(len(hist_p)), size = min(num_samples // 6, len(hist_p)), replace = False)
        hist_s_sampled = [hist_s[j] for j in hist_s_sampled_idx]
        hist_d_sampled = [hist_d[j] for j in hist_d_sampled_idx]
        hist_p_sampled = [hist_p[j] for j in hist_p_sampled_idx]
        hist_sampled = hist_s_sampled + hist_d_sampled + hist_p_sampled 
    
        #if the number of historical negatives hasn't reached the total allowed (num_samples // 2)
        if (len(hist_sampled) < num_samples // 2): 
            hist_remaining = [hist_edge for hist_edge in hist_candidates if hist_edge not in set(hist_sampled)]
            more_hist_sampled_idx = np.random.choice(range(len(hist_remaining)), 
                                     size = min(num_samples // 2 - len(hist_sampled), len(hist_remaining)), replace = False)
            more_hist_sampled = [hist_remaining[j] for j in more_hist_sampled_idx]
            hist_sampled += more_hist_sampled 
    
        #use loose negatives if not enough historical negatives
        if (len(hist_sampled) < num_samples // 2):
            loose_hist_sampled_idx = np.random.choice(range(len(loose_hist)),
                                    size = min(num_samples // 2 - len(hist_sampled), len(loose_hist)), replace = False)
            loose_hist_sampled = [loose_hist[j] for j in loose_hist_sampled_idx]
            hist_sampled += loose_hist_sampled 
    
        # sample random negatives 
        hist_deficit = num_samples // 2 - len(hist_sampled)
        random_surplus = [0,0,0]
        for _ in range(hist_deficit):
            random_surplus[int(3 * np.random.rand())] += 1
    
        #ensure the random negatives aren't historical 
        hist_edge_lexicon = ts_eval_edges_set.union(E_train_edges)
        rand_s = [s for s in L_firm if (s, pos_d, pos_p) not in ts_eval_edges_set and (s, pos_d, pos_p) not in E_train_edges]
        rand_d = [d for d in L_firm if (pos_s, d, pos_p) not in hist_edge_lexicon and (pos_s, d, pos_p) not in E_train_edges]
        rand_p = [p for p in L_products if (pos_s, pos_d, p) not in hist_edge_lexicon and (pos_s, pos_d, p) not in E_train_edges]
    
        #sample num_samples // 6 of each perturbation class, plus extras if there is a deficit of historical negatives
        rand_s_sampled = np.random.choice(rand_s, size = num_samples // 6 + random_surplus[0], replace = False)
        rand_d_sampled = np.random.choice(rand_d, size = num_samples // 6 + random_surplus[1], replace = False)
        rand_p_sampled = np.random.choice(rand_p, size = num_samples // 6 + random_surplus[2], replace = False)
        rand_s_sampled = [(s,pos_d,pos_p) for s in rand_s_sampled]
        rand_d_sampled = [(pos_s,d,pos_p) for d in rand_d_sampled]
        rand_p_sampled = [(pos_s, pos_d, p) for p in rand_p_sampled]
        rand_sampled = rand_s_sampled + rand_d_sampled + rand_p_sampled 
                              
        all_negative_samples = hist_sampled + rand_sampled
        return all_negative_samples
    
    return edge_sampler

def harness_negative_sampler(eval_ns_keys, split = "val", num_workers = 20):
    assert split in ["train", "val","test"], "split must be {'train','val','test'}"
    print("Sampling Edges in {} Split".format(split.capitalize()))
    edge_sample = edge_sampler_wrapper(split)
    with mp.Pool(num_workers) as p:
        eval_ns_values = list(tqdm(p.imap(edge_sample, eval_ns_keys), total = len(eval_ns_keys)))
    eval_ns = {key: np.array(value).astype(np.float64) for key, value in zip(eval_ns_keys, eval_ns_values)}
    return eval_ns
    
if __name__ == "__main__":
    args = get_args()
    
    df, id2entity, product_min_id = process_csv(args.csv_file, args.metric, args.logscale)
    df.to_csv(os.path.join(args.dir, f"{args.dataset_name}_edgelist.csv"), index = False) #save out edgelist
    num_nodes, num_firms, num_products = len(id2entity), product_min_id, len(id2entity) - product_min_id
    
    timestamps = sorted(list(df["ts"]))
    train_max_ts = np.percentile(timestamps, 70).astype(int)
    val_max_ts = np.percentile(timestamps, 85).astype(int)
    test_max_ts = max(timestamps)

    global E_train; global E_val; global E_test
    E_train, E_val, E_test = partition_edges(df, train_max_ts, val_max_ts, test_max_ts)

    global E_train_edges
    E_train_edges = set([(source, target, product) for ts in E_train for source, target, product in E_train[ts]])
    #create pool of node targets (firm & products) to be randomly selected during training 
    global num_samples; global L_firm; global L_products
    num_samples = args.num_samples
    L_firm = list(range(0, product_min_id))
    L_products = list(range(product_min_id, len(id2entity)))

   # E_train_edges = [(s,d,p,t) for t in E_train for s,d,p in E_train[t]]
    E_val_edges = [(s,d,p,t) for t in E_val for s,d,p in E_val[t]]
    E_test_edges = [(s,d,p,t) for t in E_test for s,d,p in E_test[t]]

    #train_ns = harness_negative_sampler(E_train_edges, split = "train", num_workers = args.workers)
    val_ns = harness_negative_sampler(E_val_edges, split = "val", num_workers = args.workers)
    test_ns = harness_negative_sampler(E_test_edges, split = "test", num_workers = args.workers)

    for pos_edge, negative_samples in val_ns.items():
        assert len(negative_samples) == num_samples
    for pos_edge, negative_samples in test_ns.items():
        assert len(negative_samples) == num_samples

    #save out sampled negatives and metadata
    with open(os.path.join(args.dir,f'{args.dataset_name}_val_ns.pkl'), 'wb') as handle:
        pickle.dump(val_ns, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(os.path.join(args.dir, f'{args.dataset_name}_test_ns.pkl'), 'wb') as handle:
        pickle.dump(test_ns, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(args.dir, f'{args.dataset_name}_meta.json'),"w") as file:
        meta = {"product_threshold": product_min_id, "id2entity": id2entity,
               "train_max_ts": int(train_max_ts), "val_max_ts": int(val_max_ts), "test_max_ts": int(test_max_ts)}
        json.dump(meta, file, indent = 4)
    



