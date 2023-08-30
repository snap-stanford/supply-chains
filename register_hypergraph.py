"""
this file morphs the transaction-level data into the TGB format, including the edge list, sampled negatives 
for the val & test splits, and supplementary metadata (e.g., mapping fom node IDs to firm & product names). 

SAMPLE USAGE:
python register_data.py --csv_file ./data/daily_transactions_2019.csv --dataset_name 
tgbl-supplychains --dir ./cache --logscale --workers 20 --product_links both 
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
    parser.add_argument('--ns_samples_per_class', nargs='?', default = 20, type = int, help = "for each perturbation class, the number of negative samples per positive edge")
    args = parser.parse_args()
    return args

def process_csv(csv_file, metric, logscale):
    df = pd.read_csv(csv_file)
    df = df[df["time_stamp"] < 20] #for debugging
    
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
              "product": [product2id[product] for product in df["hs6"]],
             "target": [company2id[firm] for firm in df["buyer_t"]],
             "weight": list(df[metric])}
    return pd.DataFrame.from_dict(graph), id2entity, product_threshold

def partition_edges(df, train_max_ts, val_max_ts, test_max_ts):
    E_train = {ts: [] for ts in range(0, train_max_ts + 1)}
    E_val = {ts: [] for ts in range(train_max_ts + 1, val_max_ts + 1)}
    E_test = {ts: [] for ts in range(val_max_ts + 1, test_max_ts + 1)}
    
    df_rows = [df[row_name] for row_name in ["source","product","target","ts"]]
    for source, product, target, ts in zip(*df_rows):
        ts, edge = int(ts), [int(source), int(product), int(target)]
        if (ts <= train_max_ts): 
            E_train[ts].append(edge)
        elif (ts <= val_max_ts): 
            E_val[ts].append(edge)
        else:
            E_test[ts].append(edge)
    return E_train, E_val, E_test

def get_links_dict(temporal_edges, isTrain = False): #prepare to relax the condition a bit
    link_map, second_order_map, list_of_edges = {}, {}, []
    for ts, edges in temporal_edges.items():
        for source, product, target in edges: 
            source_key = (-1, product, target) #complete the source
            product_key = (source, -1, target)
            target_key = (source, product, -1)

            for primary_key, tail_node in zip([source_key, product_key, target_key],
                                        [source, product, target]):
                if primary_key in link_map:
                    link_map[primary_key].add(tail_node)
                else:
                    link_map[primary_key] = {tail_node}
                    
            if (isTrain == True):        
                source_key_same_target = (-1, -2, target)
                source_key_same_product = (-1, product, -2)
                product_key_same_source = (source, -1, -2)
                product_key_same_target = (-2, -1, target)
                target_key_same_source = (source, -2, -1)
                target_key_same_product = (-2, product, -1)

                for secondary_key, tail_node in zip([source_key_same_target, source_key_same_product,
                                                    product_key_same_source, product_key_same_target,
                                                    target_key_same_source, target_key_same_product],
                                                    [source, source, product, product, target, target]):
                    if secondary_key in second_order_map:
                        second_order_map[secondary_key].add(tail_node)
                    else:
                        second_order_map[secondary_key] = {tail_node}
                
            list_of_edges.append((source, product, target, ts))

    return link_map, second_order_map, list_of_edges

def search_map(map, key):
    if (key in map):
        return map[key]
    return set()

def get_eval_negative_links(E_train, E_eval, split = "val"): #E_eval among {E_val, E_test}
    train_links, second_links, _ = get_links_dict(E_train, True)
    eval_ns_links, eval_ns_keys = {}, []
    print("Processing Through {} Links ...".format(split.capitalize()))
    for ts in tqdm(E_eval):
        eval_ns_links[ts] = {}
        eval_links_map, _, eval_edges = get_links_dict({ts: E_eval[ts]}, False)
        eval_ns_keys.extend(eval_edges)

        for incomplete_link, positive_completions in eval_links_map.items():
            hist_completions = search_map(train_links, incomplete_link).difference(positive_completions)

            source, product, target = incomplete_link
            if (source == -1): #corrupted source
                shared_target, shared_product = (-1, -2, target), (-1, product, -2)
                second_completions = search_map(second_links, shared_target).union(
                    search_map(second_links, shared_product)).difference(positive_completions)
            elif (product == -1): #corrupted product
                shared_source, shared_target = (source, -1, -2), (-2, -1, target)
                second_completions = search_map(second_links, shared_source).union(
                    search_map(second_links, shared_target)).difference(positive_completions)
            else: #corrupted target
                shared_source, shared_product = (source, -2, -1), (-2, product, -1)
                second_completions = search_map(second_links, shared_source).union(
                    search_map(second_links, shared_product)).difference(positive_completions)
            
            second_completions = second_completions.difference(hist_completions)
            eval_ns_links[ts][incomplete_link] = {"hist": hist_completions,
                                                  "second_hist": second_completions,
                                                 "positive": positive_completions} 
    return eval_ns_links, eval_ns_keys
    
def edge_sampler_wrapper(split): #returns a edge sampler function for either the val or test split 
    global edge_sampler
    eval_ns_links = val_ns_links.copy() if split == "val" else test_ns_links.copy()
    def edge_sampler(key):
        source, product, target, ts = key 
    
        all_samples = [] #in order of sampled sources, products, targets 
        for perturbed_link in [(-1, product, target), (source, -1, target), (source, product, -1)]:
            perturbed_nodes = eval_ns_links[ts][perturbed_link]
            hist_nodes = np.random.choice(list(perturbed_nodes["hist"]),
                                        size = min(num_samples // 2, len(perturbed_nodes["hist"])), replace = False)
            second_nodes = np.random.choice(list(perturbed_nodes["second_hist"]),
                                        size = min(num_samples // 2 - len(hist_nodes), len(perturbed_nodes["second_hist"])), replace = False)
            negative_inv = perturbed_nodes["hist"].union(perturbed_nodes["second_hist"]).union(perturbed_nodes["positive"])

            if (perturbed_link[1] == -1): #sampling a product
                neg_nodes = np.random.choice([l for l in L_products if l not in negative_inv],
                                        size = num_samples - len(hist_nodes) - len(second_nodes), replace = False)
            else:
                neg_nodes = np.random.choice([l for l in L_firm if l not in negative_inv],
                                        size = num_samples - len(hist_nodes) - len(second_nodes), replace = False)
            sampled_nodes = list(hist_nodes) + list(second_nodes) + list(neg_nodes)
            
            all_samples.extend(sampled_nodes)

        return np.array(all_samples).astype(np.float64)

    return edge_sampler 

def harness_negative_sampler(eval_ns_keys, split = "val", num_workers = 20):
    assert split in ["val","test"], "split must be {'val','test'}"
    print("Sampling Edges in {} Split".format(split.capitalize()))
    edge_sample = edge_sampler_wrapper(split)
    with mp.Pool(num_workers) as p:
        eval_ns_values = list(tqdm(p.imap(edge_sample, eval_ns_keys), total = len(eval_ns_keys)))
    eval_ns = {key: value for key, value in zip(eval_ns_keys, eval_ns_values)}
    return eval_ns

if __name__ == "__main__":
    args = get_args()
    global product_min_id
    df, id2entity, product_min_id = process_csv(args.csv_file, args.metric, args.logscale)
    df.to_csv(os.path.join(args.dir, f"{args.dataset_name}_edgelist.csv"), index = False) #save out edgelist
    num_nodes, num_firms, num_products = len(id2entity), product_min_id, len(id2entity) - product_min_id
    print(f"Number of Nodes: {num_firms} (Firms), {num_products} (Products), {num_nodes} (Total)")
    print(f"Number of Hyperedges: {len(df)}")
    print(df.head(3), "\n", df.tail(3))
    
    #stratify the data into train, val, test split based on 70%-15%-15% of edges
    timestamps = sorted(list(df["ts"]))
    train_max_ts = np.percentile(timestamps, 70).astype(int)
    val_max_ts = np.percentile(timestamps, 85).astype(int)
    test_max_ts = max(timestamps)
    E_train, E_val, E_test = partition_edges(df, train_max_ts, val_max_ts, test_max_ts)

    #create pool of node targets (firm & products) to be randomly selected during training 
    global num_samples; global L_firm; global L_products
    num_samples = args.ns_samples_per_class
    L_firm = list(range(0, product_min_id))
    L_products = list(range(product_min_id, len(id2entity)))

    #retrieve positive edges and sample negative ones in the val & test splits 
    global val_ns_links; global test_ns_links
    val_ns_links, val_ns_keys = get_eval_negative_links(E_train, E_val, split = "val")
    test_ns_links, test_ns_keys = get_eval_negative_links(E_train, E_test, split = "test")
    val_ns = harness_negative_sampler(val_ns_keys, split = "val", num_workers = args.workers)
    test_ns = harness_negative_sampler(test_ns_keys, split = "test", num_workers = args.workers)

    #save out sampled negatives and metadata
    with open(os.path.join(args.dir,f'{args.dataset_name}_val_ns.pkl'), 'wb') as handle:
        pickle.dump(val_ns, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(os.path.join(args.dir, f'{args.dataset_name}_test_ns.pkl'), 'wb') as handle:
        pickle.dump(test_ns, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(args.dir, f'{args.dataset_name}_meta.json'),"w") as file:
        meta = {"product_threshold": product_min_id, "id2entity": id2entity,
               "train_max_ts": int(train_max_ts), "val_max_ts": int(val_max_ts), "test_max_ts": int(test_max_ts)}
        json.dump(meta, file, indent = 4)
