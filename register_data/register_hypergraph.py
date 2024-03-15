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
    parser.add_argument('dataset_name', help = "name to be assigned to dataset")
    parser.add_argument('dir', help = "directory to save data")
    parser.add_argument('--skip_process_csv', action='store_true', help = "if true, use already processed edgelist")
    parser.add_argument('--csv_file', nargs='?', default = "../hitachi-supply-chains/temporal_graph/storage/daily_transactions_2021.csv", help = "path to CSV file with transactions")
    parser.add_argument('--metric', nargs='?', default = "total_amount", help = "either total amount (in USD), which is default, or weight")
    parser.add_argument('--logscale', action='store_true', help = "if true, apply logarithm to edge weights")
    parser.add_argument('--use_prev_sampling', action='store_true', help = "if true, use the hyperedge sampling approach prior to fixing loose negatives on Oct 24")
    parser.add_argument('--workers', nargs='?', default = 10, type = int, help = "number of thread workers")
    parser.add_argument('--num_samples', nargs='?', default = 18, type = int, help = "number of negative samples per positive edge (default: 18)")
    parser.add_argument('--max_timestamp', nargs='?', default = None, type = int, help = "max timestamp to include (mainly for debugging)")
    args = parser.parse_args()
    return args

def process_csv(csv_file, metric, logscale, max_timestamp = None):
    df = pd.read_csv(csv_file)
    if (max_timestamp != None):
        df = df[df["time_stamp"] <= max_timestamp]
    
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

def partition_edges(df, train_max_ts, val_max_ts, test_max_ts, use_prev_sampling = True):
    #creates the chronological train / val / test split of the hyperedges 
    E_train = {ts: [] for ts in range(0, train_max_ts + 1)}
    E_val = {ts: [] for ts in range(train_max_ts + 1, val_max_ts + 1)}
    E_test = {ts: [] for ts in range(val_max_ts + 1, test_max_ts + 1)}
    
    df_rows = [df[row_name] for row_name in ["source","target","product","ts"]]
    for source, destination, product, ts in zip(*df_rows):
        if (use_prev_sampling == False):
            ts, edge = int(ts), [int(source), int(destination), int(product)]
        else:
            ts, edge = int(ts), [int(source), int(product), int(destination)]
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
    
def edge_sampler_deprecated_wrapper(split): #deprecated version (prior to Oct 24 -- loose negatives correction)
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
    
    

        #ensure the random negatives aren't historical 
        #hist_edge_lexicon = ts_eval_edges_set.union(E_train_edges)
        rand_s = [s for s in L_firm if (s, pos_d, pos_p) not in ts_eval_edges_set and (s, pos_d, pos_p) not in E_train_edges]
        rand_d = [d for d in L_firm if (pos_s, d, pos_p) not in ts_eval_edges_set and (pos_s, d, pos_p) not in E_train_edges]
        rand_p = [p for p in L_products if (pos_s, pos_d, p) not in ts_eval_edges_set and (pos_s, pos_d, p) not in E_train_edges]
        
        # sample random negatives 
        hist_deficit = num_samples // 2 - len(hist_sampled)
        random_surplus = [0,0,0]
        for _ in range(hist_deficit):
            random_surplus[int(3 * np.random.rand())] += 1
        
        #TODO: safety-check in case not enough random negatives (keep waterfalling)
        num_new_s = num_samples // 6  + random_surplus[0] #positive
        num_new_d = num_samples // 6  + random_surplus[1] #positive
        num_new_p = num_samples // 6  + random_surplus[2] # positive
        
        emergency_s, emergency_d, emergency_p = False, False, False
        while (len(rand_s) < num_new_s and num_new_s >= 0):
            num_new_s -= 1
            num_new_d += 1
        while (len(rand_d) < num_new_d and num_new_d >= 0):
            num_new_d -= 1
            num_new_p += 1
        if (len(rand_p) < num_new_p):
            #emergency activation 
            if (len(rand_p) > 0):
                emergency_p = True 
            elif (len(rand_s) > 0):
                num_new_s += num_new_p
                num_new_p = 0
                emergency_s = True
            elif (len(rand_d) > 0):
                num_new_d += num_new_p
                num_new_p = 0
                emergency_d = True 
            else: 
                raise ValueError("no random negatives at time stamp {}".format(ts))
    
        #sample num_samples // 6 of each perturbation class, plus extras if there is a deficit of historical negatives
        rand_s_sampled = np.random.choice(rand_s, size = num_new_s, replace = emergency_s)
        rand_d_sampled = np.random.choice(rand_d, size = num_new_d, replace = emergency_d)
        rand_p_sampled = np.random.choice(rand_p, size = num_new_p, replace = emergency_p)

        rand_s_sampled = [(s,pos_d,pos_p) for s in rand_s_sampled]
        rand_d_sampled = [(pos_s,d,pos_p) for d in rand_d_sampled]
        rand_p_sampled = [(pos_s, pos_d, p) for p in rand_p_sampled]
        rand_sampled = rand_s_sampled + rand_d_sampled + rand_p_sampled 
                              
        all_negative_samples = hist_sampled + rand_sampled
        return all_negative_samples
    
    return edge_sampler

def harness_negative_sampler(eval_ns_keys, split = "val", num_workers = 20, use_prev_sampling = False):
    assert split in ["train", "val","test"], "split must be {'train','val','test'}"
    print("Sampling Edges in {} Split".format(split.capitalize()))
    if (use_prev_sampling == False):
        edge_sample = edge_sampler_wrapper(split)
    else:
        edge_sample = edge_sampler_deprecated_wrapper(split)
    with mp.Pool(num_workers) as p:
        eval_ns_values = list(tqdm(p.imap(edge_sample, eval_ns_keys), total = len(eval_ns_keys)))
    eval_ns = {key: np.array(value).astype(np.float64) for key, value in zip(eval_ns_keys, eval_ns_values)}
    return eval_ns
    
if __name__ == "__main__":
    args = get_args()
    
    edgelist_fn = os.path.join(args.dir, f'{args.dataset_name}_edgelist.csv')
    meta_fn = os.path.join(args.dir, f'{args.dataset_name}_meta.json')
    if args.skip_process_csv:  # edgelist and ID mapping processed externally
        assert os.path.isfile(edgelist_fn)
        assert os.path.isfile(meta_fn)
        df = pd.read_csv(edgelist_fn)
        expected_cols = ['ts', 'source', 'target', 'product', 'weight']
        assert np.isin(expected_cols, df.columns).all()
        with open(meta_fn, 'r') as f:
            meta = json.load(f)
        assert 'id2entity' in meta and 'product_threshold' in meta
        id2entity = meta['id2entity']
        product_min_id = meta['product_threshold']
        assert (df['product'] >= product_min_id).all()
    else:
        df, id2entity, product_min_id = process_csv(args.csv_file, args.metric, args.logscale, args.max_timestamp)
        df.to_csv(edgelist_fn, index = False)  # save edgelist
    num_nodes, num_firms, num_products = len(id2entity), product_min_id, len(id2entity) - product_min_id
    
    timestamps = sorted(list(df["ts"]))
    train_max_ts = np.percentile(timestamps, 70).astype(int)
    val_max_ts = np.percentile(timestamps, 85).astype(int)
    test_max_ts = max(timestamps)

    global E_train; global E_val; global E_test
    E_train, E_val, E_test = partition_edges(df, train_max_ts, val_max_ts, test_max_ts, use_prev_sampling = args.use_prev_sampling)

    global E_train_edges
    E_train_edges = set([(source, target, product) for ts in E_train for source, target, product in E_train[ts]])
    #create pool of node targets (firm & products) to be randomly selected during training 
    global num_samples; global L_firm; global L_products
    num_samples = args.num_samples
    L_firm = list(range(0, product_min_id))
    L_products = list(range(product_min_id, len(id2entity)))
    print(len(L_firm), len(L_products))

   # E_train_edges = [(s,d,p,t) for t in E_train for s,d,p in E_train[t]]
    if (args.use_prev_sampling == False):
        E_val_edges = [(s,d,p,t) for t in E_val for s,d,p in E_val[t]]
        E_test_edges = [(s,d,p,t) for t in E_test for s,d,p in E_test[t]]
    
        #train_ns = harness_negative_sampler(E_train_edges, split = "train", num_workers = args.workers)
        val_ns = harness_negative_sampler(E_val_edges, split = "val", num_workers = args.workers)
        test_ns = harness_negative_sampler(E_test_edges, split = "test", num_workers = args.workers)
        
        for pos_edge, negative_samples in val_ns.items():
            assert len(negative_samples) == num_samples

    else: 
        global val_ns_links; global test_ns_links
        num_samples = num_samples // 3
        val_ns_links, val_ns_keys = get_eval_negative_links(E_train, E_val, split = "val")
        test_ns_links, test_ns_keys = get_eval_negative_links(E_train, E_test, split = "test")
        val_ns = harness_negative_sampler(val_ns_keys, split = "val", num_workers = args.workers,
                                          use_prev_sampling = args.use_prev_sampling)
        test_ns = harness_negative_sampler(test_ns_keys, split = "test", num_workers = args.workers,
                                           use_prev_sampling = args.use_prev_sampling)
        
        for pos_edge, negative_samples in test_ns.items():
            assert len(negative_samples) == num_samples * 3
        
    #save out sampled negatives and metadata
    with open(os.path.join(args.dir,f'{args.dataset_name}_val_ns.pkl'), 'wb') as handle:
        pickle.dump(val_ns, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(os.path.join(args.dir, f'{args.dataset_name}_test_ns.pkl'), 'wb') as handle:
        pickle.dump(test_ns, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(meta_fn, "w") as file:
        meta = {"product_threshold": product_min_id, "id2entity": id2entity,
               "train_max_ts": int(train_max_ts), "val_max_ts": int(val_max_ts), "test_max_ts": int(test_max_ts)}
        json.dump(meta, file, indent = 4)
    



