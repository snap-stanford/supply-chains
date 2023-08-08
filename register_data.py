"""
this file morphs the transaction-level data into the TGB format, including the edge list, sampled negatives 
for the val & test splits, and supplementary metadata (e.g., mapping fom node IDs to firm & product names). 

SAMPLE USAGE:
python register_data.py --csv_file ./data/daily_transactions_2019.csv --dataset_name 
tgbl-supplychains-2021 --dir ./cache --logscale --workers 20 --bipartite
(use the --bipartite flag to create a firm-product graph, and remove it to create a smaller firm-firm graph divested of products)
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
    parser.add_argument('--ns_samples', nargs='?', default = 20, type = int, help = "number of negative samples / pertubations per positive edge")
    parser.add_argument('--bipartite', action='store_true', help = "whether to stratify nodes into firms & products")
    args = parser.parse_args()
    return args

def process_csv_firm_firm(csv_file, metric, logscale):
    df = pd.read_csv(csv_file)
    df = df.groupby(by = ["time_stamp","supplier_t","buyer_t"]).sum(numeric_only = True).reset_index()
    df = df[(df["supplier_t"] != "") & (df["buyer_t"] != "") & (~df[metric].isna()) & (~df["supplier_t"].isna()) & (~df["buyer_t"].isna())]
    
    all_companies = list(set(df["supplier_t"]).union(set(df["buyer_t"])))
    id2company = {key: value for key,value in enumerate(all_companies)}
    company2id = {value: key for key,value in enumerate(all_companies)}

    df["weight"] = df[metric].apply(lambda weight: np.log10(weight+1) if logscale == True else weight)
    df["source"] = df["supplier_t"].apply(lambda firm_t: company2id[firm_t])
    df["target"] = df["buyer_t"].apply(lambda firm_t: company2id[firm_t])
    df = df.rename(columns = {"time_stamp": "ts"})
    df = df.drop(columns = {"bill_count", "total_quantity","total_weight", 
                            "buyer_t","supplier_t", "hs6","total_amount"})
    df = df[["ts","source","target","weight"]]
    return df, id2company

def process_csv_firm_product(csv_file, metric, logscale):
    df = pd.read_csv(csv_file)
    df = df.groupby(by = ["time_stamp","supplier_t","buyer_t","hs6"]).sum(numeric_only = True).reset_index()
    df = df[(df["supplier_t"] != "") & (df["buyer_t"] != "") & (~df[metric].isna()) & (~df["supplier_t"].isna()) & (~df["buyer_t"].isna())]
    all_companies = set(df["supplier_t"]).union(set(df["buyer_t"]))
    all_products = set(df["hs6"])

    #create map from node IDs to firm / product names, and inverse
    company2id = {value: key for key,value in enumerate(all_companies)}
    product2id = {value: key + len(all_companies) for key,value in enumerate(all_products)}
    id2entity = {key: value for key,value in enumerate(all_companies)}
    id2product = {key + len(all_companies): value for key,value in enumerate(all_products)}
    id2entity.update(id2product)
    #minimum ID for a product (to distinguish between firm & product nodes)
    product_threshold = min(list(id2product.keys()))

    #create new dataframe for firm-product links (selling is firm -> product and purchase is product -> firm)
    rows = [list(df[row_t]) for row_t in ["time_stamp","hs6","supplier_t","buyer_t",metric]]
    df_bipartite = {"ts": [], "source": [], "target": [], "weight": []}
    for ts, product, supplier, buyer, edge_weight in zip(*rows):
        supplier_id, buyer_id = company2id[supplier], company2id[buyer]
        product_id = product2id[product]
        weight = np.log10(edge_weight + 1) if logscale == True else edge_weight
        
        df_bipartite["ts"].extend([ts,ts])
        df_bipartite["source"].extend([supplier_id, product_id])
        df_bipartite["target"].extend([product_id, buyer_id])
        df_bipartite["weight"].extend([weight, weight])

    df_bipartite = pd.DataFrame.from_dict(df_bipartite)
    return df_bipartite, id2entity, product_threshold

def partition_edges(df, train_max_ts, val_max_ts, test_max_ts):
    E_train = {ts: [] for ts in range(0, train_max_ts + 1)}
    E_val = {ts: [] for ts in range(train_max_ts + 1, val_max_ts + 1)}
    E_test = {ts: [] for ts in range(val_max_ts + 1, test_max_ts + 1)}
    
    df_rows = [df[row_name] for row_name in ["source","target","ts"]]
    for source, target, ts in zip(*df_rows):
        ts, edge = int(ts), [int(source), int(target)]
        if (ts <= train_max_ts): 
            E_train[ts].append(edge)
        elif (ts <= val_max_ts): 
            E_val[ts].append(edge)
        else:
            E_test[ts].append(edge)
    return E_train, E_val, E_test

def get_target_dict(temporal_edges):
    #creates dictionary where each key is a source node, and the corresponding value 
    #is the set of target nodes that appear in training 
    targets_map, list_of_edges = {}, []
    for ts, edges in temporal_edges.items():
        for source, target in edges:
            if source in targets_map:
                targets_map[source].add(target)
            else:
                targets_map[source] = {target}
            list_of_edges.append((source, target, ts))
    return targets_map, list_of_edges
        
def get_eval_negative_targets(E_train, E_eval): #E_eval among {E_val, E_test}
    train_targets, _ = get_target_dict(E_train)
    eval_ns_targets, eval_ns_keys = {}, []
    for ts in E_eval:
        eval_ns_targets[ts] = {}
        eval_targets_map, eval_edges = get_target_dict({ts: E_eval[ts]})
        eval_ns_keys.extend(eval_edges) 

        for source, positive_targets in eval_targets_map.items():
            hist_targets = []
            if (source in train_targets):
                hist_targets = train_targets[source]
                hist_targets = hist_targets.difference(positive_targets)

            negative_targets_inv = positive_targets.union(hist_targets)
            eval_ns_targets[ts][source] = {"hist": hist_targets, 
                 "negative_targets_inv": negative_targets_inv,
                 "source_node": "product" if source >= product_min_id else "firm"}
    
    return eval_ns_targets, eval_ns_keys 

def edge_sampler_wrapper(split): #returns a edge sampler function for either the val or test split 
    global edge_sampler
    eval_ns_targets = val_ns_targets.copy() if split == "val" else test_ns_targets.copy()
    def edge_sampler(key):
        source, target, ts = key 
        all_targets = eval_ns_targets[ts][source]
        #sample historical negatives
        if (len(all_targets["hist"]) >= num_samples // 2):
            sampled_hist = np.random.choice(list(all_targets["hist"]),
                                           size = num_samples // 2, replace = False)
        else:
            sampled_hist = all_targets["hist"]
        
        #sample "regular" negatives (i.e. not historical / having materialized in training)
        if (all_targets["source_node"] == "product" or bipartite == False): #target should be firm
            regular_negatives = [l for l in L_firm if l not in all_targets["negative_targets_inv"]]
        else: #target should be product 
            regular_negatives = [l for l in L_products if l not in all_targets["negative_targets_inv"]]
        sampled_negatives = np.random.choice(regular_negatives, size = num_samples - len(sampled_hist), replace = False)
        return np.array(list(sampled_hist) + list(sampled_negatives)).astype(np.float64)
    
    return edge_sampler 

def harness_negative_sampler(eval_ns_keys, split = "val", num_workers = 20):
    assert split in ["val","test"], "split must be {'val','test'}"
    edge_sample = edge_sampler_wrapper(split)
    with mp.Pool(num_workers) as p:
        eval_ns_values = list(tqdm(p.imap(edge_sample, eval_ns_keys), total = len(eval_ns_keys)))
    eval_ns = {key: value for key, value in zip(eval_ns_keys, eval_ns_values)}
    return eval_ns

if __name__ == "__main__":
    args = get_args()
    #extract temporal edges from the CSV file of supply-chain transactions
    if (args.bipartite == True):
        df, id2entity, product_min_id = process_csv_firm_product(args.csv_file, args.metric, args.logscale)
    else:
        df, id2entity = process_csv_firm_firm(args.csv_file, args.metric, args.logscale)
        product_min_id = len(id2entity)
    
    num_nodes = len(id2entity) #number of nodes
    df.to_csv(os.path.join(args.dir, f"{args.dataset_name}_edgelist.csv"), index = False) #save out edgelist

    #stratify the data into train, val, test split based on 70%-15%-15% of edges
    timestamps = sorted(list(df["ts"]))
    train_max_ts = np.percentile(timestamps, 70).astype(int)
    val_max_ts = np.percentile(timestamps, 85).astype(int)
    test_max_ts = max(timestamps)
    E_train, E_val, E_test = partition_edges(df, train_max_ts, val_max_ts, test_max_ts)

    #create pool of node targets (firm & products) to be randomly selected during training 
    global num_samples; global bipartite; global L_firm; global L_products
    num_samples, bipartite = args.ns_samples, args.bipartite
    L_firm = list(range(0, product_min_id))
    L_products = list(range(product_min_id, len(id2entity)))

    #retrieve positive edges and sample negative ones in the val & test splits 
    global val_ns_targets; global test_ns_targets
    val_ns_targets, val_ns_keys = get_eval_negative_targets(E_train, E_val)
    test_ns_targets, test_ns_keys = get_eval_negative_targets(E_train, E_test)
    val_ns = harness_negative_sampler(val_ns_keys, split = "val", num_workers = args.workers)
    test_ns = harness_negative_sampler(test_ns_keys, split = "test", num_workers = args.workers)

    #save out sampled negatives and metadata
    with open(os.path.join(args.dir,f'{args.dataset_name}_val_ns.pkl'), 'wb') as handle:
        pickle.dump(val_ns, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(os.path.join(args.dir, f'{args.dataset_name}_test_ns.pkl'), 'wb') as handle:
        pickle.dump(test_ns, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(args.dir, f'{args.dataset_name}_meta.json'),"w") as file:
        meta = {"product_threshold": product_min_id, "id2entity": id2entity}
        json.dump(meta, file, indent = 4)
    
    




