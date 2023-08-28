import os 
import glob
import pandas as pd
import pickle 
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse 
plt.style.use("fivethirtyeight")

def calc_mean_MRR(ranks):
    if (len(ranks) == 0): raise ValueError("len(ranks) >= 1")
    total = 0
    for rank in ranks:
        total += 1 / rank 
    return total / len(ranks)

def get_positive_rank(train_freq, src, positive_dest, negative_dest):
    #rank destinations by occurrence in the training data
    #(e.g. number of time-steps for which a transaction occurs)
    destinations = [positive_dest] + list(negative_dest)
    occurrences = [train_freq[(src, dest)] if (src, dest) in train_freq else 0 for dest in destinations]
    positive_freq = occurrences[0]
    occurrences = sorted(occurrences, reverse = True)

    ranks = [idx + 1 for idx, freq in enumerate(occurrences) if freq == positive_freq]
    return np.random.choice(ranks)

def get_prop_hist(train_freq, src, negative_dest):
    num_hist = len([dest for dest in negative_dest if (src, dest) in train_freq])
    return num_hist / len(negative_dest)

def run_baseline(dir, dataset_name):
    #load in data as sculpted for TGB training
    edgelist_filename = os.path.join(dir, f"{dataset_name}_edgelist.csv")
    df_edgelist = pd.read_csv(edgelist_filename)
    
    val_ns_filename = os.path.join(dir, f"{dataset_name}_val_ns.pkl")
    with open(val_ns_filename, "rb") as file:
        val_ns = pickle.load(file)
    val_ts = sorted(list(set([ts for source, target, ts in val_ns.keys()])))
    val_min_ts, val_max_ts = min(val_ts), max(val_ts)
    
    test_ns_filename = os.path.join(dir, f"{dataset_name}_test_ns.pkl")
    with open(test_ns_filename, "rb") as file:
        test_ns = pickle.load(file)
    test_ts = sorted(list(set([ts for source, target, ts in test_ns.keys()])))
    test_min_ts, test_max_ts = min(test_ts), max(test_ts)
    
    #calculate frequency of firm-firm links in training data
    df_train_edges = df_edgelist[df_edgelist["ts"] < val_min_ts]
    df_rows = [list(df_train_edges[r]) for r in ["source","target"]]
    train_freq = {}
    for source, target in zip(*df_rows):
        if (source, target) not in train_freq:
            train_freq[(source, target)] = 1
        else:
            train_freq[(source, target)] += 1

    #evaluate majority baseline on val negative samples
    val_ranks, val_hist = [], {ts: [] for ts in val_ts}
    for src, dest, ts in val_ns: 
        negative_dest = val_ns[(src, dest, ts)].astype(int)
        rank = get_positive_rank(train_freq, src, dest, negative_dest)
        val_hist[ts].append(get_prop_hist(train_freq, src, negative_dest))
        val_ranks.append(rank)
        
    #evaluate majority baseline on test negative samples
    test_ranks, test_hist = [], {ts: [] for ts in test_ts}
    for src, dest, ts in test_ns: 
        negative_dest = test_ns[(src, dest, ts)].astype(int)
        rank = get_positive_rank(train_freq, src, dest, negative_dest)
        test_hist[ts].append(get_prop_hist(train_freq, src, negative_dest))
        test_ranks.append(rank)
        
    val_mean_MRR = calc_mean_MRR(val_ranks)
    test_mean_MRR = calc_mean_MRR(test_ranks)
    
    MRR_dict = {"val": val_mean_MRR, "test": test_mean_MRR,
                     "val_hist": {ts: sum(L) / len(L) for ts, L in val_hist.items()},
                     "test_hist": {ts: sum(L) / len(L) for ts, L in test_hist.items()}}

    return MRR_dict

def plot_annotations(bars, fontsize):
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize = fontsize)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extracting graph data from the transactions in logistic_data')
    parser.add_argument('--dataset', nargs='?', default = "tgbl-supplychains2019", help = "name of dataset")
    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_dir = "./TGB/tgb/datasets/{}".format(dataset_name.replace("-","_"))
    results_dict = run_baseline(dataset_dir, dataset_name)
    
    #plot the proportions of historical nodes among the eval negatives  
    cmap = plt.cm.gist_earth
    plt.figure(figsize = (10,6), tight_layout = True)
    val_hist = np.array([[ts, hist] for ts, hist in results_dict["val_hist"].items()])
    x_val = val_hist[:,0][np.argsort(val_hist[:,0])]
    y_val = val_hist[:,1][np.argsort(val_hist[:,0])]

    test_hist = np.array([[ts, hist] for ts, hist in results_dict["test_hist"].items()])
    x_test = test_hist[:,0][np.argsort(test_hist[:,0])] 
    y_test = test_hist[:,1][np.argsort(test_hist[:,0])]

    plt.scatter(x_val,y_val, marker = "h", s = 30, label = "val", color = cmap(0.3), edgecolor = "black")
    plt.scatter(x_test,y_test, marker = "o", s = 30, label = "test", color = cmap(0.6), edgecolor = "black")

    plt.ylabel("Proportion of Hist Negatives", fontsize = 14)
    plt.xlabel(f"Time Stamp (Days)", fontsize = 14)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f"Negative-Link Sampling on {dataset_name}", fontsize = 20)
    plt.savefig(f"./cache/{dataset_name}_negativehist.jpg")
    plt.clf()

    #plot the performance of the majority baseline for link prediction 
    plt.figure(figsize = (6,4), tight_layout = True)
    bars = plt.bar(["Val","Test"], [results_dict["val"], results_dict["test"]], 0.8,
                  color = [cmap(0.3), cmap(0.6)])
    plot_annotations(bars, fontsize = 12)

    plt.xlabel("Split", fontsize = 13)
    plt.ylabel("Mean MRR", fontsize = 13)
    plt.title(f"Majority Baseline on {dataset_name}", fontsize = 14, y = 1.05)
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(f"./cache/{dataset_name}_results.jpg")
    plt.clf()
    
    
    
    


