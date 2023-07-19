import pandas as pd
import argparse
import numpy as np
import pickle 
import json
from tqdm import tqdm
import os

#make these into argparse
parser = argparse.ArgumentParser(description='Extracting graph data from the transactions in logistic_data')
parser.add_argument('--csv_file', nargs='?', default = "../hitachi-supply-chains/temporal_graph/storage/daily_transactions_2021.csv")
parser.add_argument('--dataset_name', nargs='?', default = "tgbl-supplychains")
parser.add_argument('--metric', nargs='?', default = "total_amount")
parser.add_argument('--dir', nargs='?', default = "./tgb_data")
parser.add_argument('--logscale', action='store_true')
args = parser.parse_args()

csv_file = args.csv_file
dataset_name = args.dataset_name
metric = args.metric
logscale = args.logscale
DIR = args.dir

df_h = pd.read_csv(csv_file)
df_h = df_h.groupby(by = ["time_stamp","supplier_t","buyer_t"]).sum(numeric_only = True).reset_index()

all_companies = list(set(df_h["supplier_t"]).union(set(df_h["buyer_t"])))
id2company = {key: value for key,value in enumerate(all_companies)}
company2id = {value: key for key,value in enumerate(all_companies)}

df2 = df_h.copy()
df2 = df2[(df2["supplier_t"] != "") & (df2["buyer_t"] != "") & (~df2[metric].isna())]
df2["weight"] = [np.log10(a + 1) if logscale == True else a for a in df2[metric]]
df2["source"] = [company2id[name] for name in df2["supplier_t"]]
df2["target"] = [company2id[name] for name in df2["buyer_t"]]
df2 = df2.rename(columns = {"time_stamp": "ts"}).drop(columns = {"bill_count", "total_quantity",
                                                                "total_weight", "buyer_t","supplier_t",
                                                                "hs6","total_amount"})
timestamps = sorted(list(df2["ts"]))
train_val_cutoff = np.percentile(timestamps, 70).astype(int)
val_test_cutoff = np.percentile(timestamps, 85).astype(int)

#df_train = df2[df2["ts"] <= train_val_cutoff]
df2 = df2[["ts","source","target","weight"]]
df2.to_csv(os.path.join(DIR, f"{dataset_name}_edgelist.csv"), index = False)

n = len(all_companies)

#split data into 70-15-15 train/val/test
E_train = {ts: [] for ts in range(0, train_val_cutoff + 1)}
E_val = {ts: [] for ts in range(train_val_cutoff + 1, val_test_cutoff + 1)}
E_test = {ts: [] for ts in range(val_test_cutoff + 1, max(timestamps) + 1)}

df_rows = [df2[row_name] for row_name in ["source","target","ts"]]
for source, target, ts in zip(*df_rows):
    ts = int(ts)
    edge = [int(source), int(target)]
    if (ts <= train_val_cutoff): 
        E_train[ts].append(edge)
    elif (ts <= val_test_cutoff): 
        E_val[ts].append(edge)
    else:
        E_test[ts].append(edge)

#TODO: make faster (currently quite slow and obtusely written)
def sample_negatives(edges_train, edges_val, num_samples = 20):
    #preprocess the training edges to get a historical record, time-agnostic 
    train_record = {}
    for ts in edges_train:
        edges = edges_train[ts]
        for source, target in edges: 
            if source in train_record:
                train_record[source].add(target)
            else:
                train_record[source] = {target}

    val_ns = {}
    for ts in tqdm(edges_val):
        edges = edges_val[ts]
        #preprocessing to get a list of targets for each source node at the given timestamp 
        target_dict = {} 
        for source, target in edges: 
            if (source in target_dict):
                target_dict[source].add(target)
            else:
                target_dict[source] = {target}

        for source, target in edges:
            #perturb the target node 
            if (source in train_record):
                historical_targets = [dest for dest in train_record[source] if dest not in target_dict[source]] 
            else:
                historical_targets = []
            positives = target_dict[source].union(set(historical_targets))
            negative_targets = [dest for dest in range(n) if dest not in positives]

            #sample equally for each 
            if (len(historical_targets) >= num_samples // 2):
                sampled_negatives = np.random.choice(negative_targets, size = num_samples // 2, 
                                                     replace = False)
                sampled_historical = np.random.choice(historical_targets, size = num_samples // 2, 
                                                      replace = False)
            else:
                sampled_negatives = np.random.choice(negative_targets, 
                                                     size = num_samples - len(historical_targets),
                                                     replace = False)
                sampled_historical = historical_targets[:]
        
            val_ns[(source, target, ts)] = np.array(list(sampled_negatives) + list(sampled_historical)).astype(np.float64)
    return val_ns 
      
num_samples = 20
val_ns = sample_negatives(E_train, E_val, num_samples)
test_ns = sample_negatives(E_train, E_test, num_samples)

with open(os.path.join(DIR,f'{dataset_name}_val_ns.pkl'), 'wb') as handle:
    pickle.dump(val_ns, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(os.path.join(DIR, f'{dataset_name}_test_ns.pkl'), 'wb') as handle:
    pickle.dump(test_ns, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(DIR, f'{dataset_name}_id2company.json'),"w") as file:
    json.dump(id2company, file, indent = 4)