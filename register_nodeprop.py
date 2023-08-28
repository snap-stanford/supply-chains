"""
this file morphs the transaction-level data into the TGB format for node property prediction, 
which encloses an edgelist CSV file, a node property CSV file, and a metadata JSON

<sample usage>
python register_nodeprop.py --csv_file ./transactions/daily_transactions_2021.csv --dir ./cache --dataset_name tgbn-supplychains --product_links supply
"""

import pandas as pd
import argparse
import numpy as np
import pickle 
import json
import os

def get_args():
    #make these into argparse
    parser = argparse.ArgumentParser(description='Extracting graph data from the transactions in logistic_data')
    parser.add_argument('--csv_file', nargs='?', default = "../hitachi-supply-chains/temporal_graph/storage/daily_transactions_2021.csv", help = "path to CSV file with transactions")
    parser.add_argument('--dataset_name', nargs='?', default = "tgbl-supplychains", help = "name to be assigned to dataset")
    parser.add_argument('--metric', nargs='?', default = "total_amount", help = "either total amount (in USD), which is default, or weight")
    parser.add_argument('--dir', nargs='?', default = "./tgb_data", help = "directory to save data")
    parser.add_argument('--logscale', action='store_true', help = "if true, apply logarithm to edge weights")
    parser.add_argument('--product_links', nargs='?', default = "none", help = "which product transactions to include, out of {'buy','supply'}")
    args = parser.parse_args()
    assert args.product_links in {'buy','supply'}, "product edges must be in {'buy','supply'}"
    return args

def format_csv(csv_file, metric = "total_amount", logscale = False, link_type = "buy"):
    df = pd.read_csv(csv_file)
    
    #excise firms with missing names and NA values 
    df = df.groupby(by = ["time_stamp","supplier_t","buyer_t","hs6"]).sum(numeric_only = True).reset_index()
    df = df[(df["supplier_t"] != "") & (df["buyer_t"] != "") & (~df["supplier_t"].isna()) & (~df["buyer_t"].isna())]
    df = df.drop(columns = {"bill_count","total_quantity","total_weight","total_amount"}.difference({metric}))
    df = df[(~df[metric].isna()) & (df[metric] > 0)]
    if (logscale == True):
        df[metric] = df[metric].apply(lambda value: np.log10(value + 1))
    
    #supply is supplier -> product link, while buy is buyer -> product link
    #normalize the trade values into proportions of a firm's activity at a given timestamp
    aggregation_node = "supplier_t" if link_type == "supply" else "buyer_t"
    df_node = df.groupby(by = ["time_stamp", aggregation_node,"hs6"]).sum(numeric_only = True).reset_index()
    df_node_total = df_node.groupby(by = ["time_stamp", aggregation_node]).sum(numeric_only = True).reset_index()
    df_node_total = df_node_total.rename(columns = {metric: "normalization"}).drop(columns = {"hs6"})
    df_node = pd.merge(df_node, df_node_total, on = ["time_stamp", aggregation_node], how = "inner")
    df_node["weight"] = [num / denom for num, denom in zip(df_node[metric], df_node["normalization"])]
    
    #format dataframe into the column order mandated by TGB
    df_node = df_node.drop(columns = {"normalization",metric})
    df_node["hs6"] = df_node["hs6"].apply(lambda product_code: f"HS6_PRODUCT_{product_code}")
    df_node = df_node[["time_stamp",aggregation_node, "hs6", "weight"]]
    num_classes = len(set(df_node["hs6"]))

    return df_node, num_classes
    
if __name__ == "__main__":
    args = get_args()
    df_edgelist, num_classes = format_csv(args.csv_file, args.metric, args.logscale, 
                                          link_type = args.product_links)
    
    #create a corresponding node property dataframe (one time_stamp shifted ahead)
    min_ts = min(df_edgelist["time_stamp"])
    df_nodelabel = df_edgelist[df_edgelist["time_stamp"] > min_ts].reset_index().drop(columns = {"index"})
    
    print(f"Number of Node Property Classes: {num_classes}")
    print(f"Size of Edge List: {len(df_edgelist)}")
    print(f"Size of Node Property List: {len(df_nodelabel)}")
    
    #save out dataframes to CSV files, and metadata to a JSON file
    df_edgelist.to_csv(os.path.join(args.dir, f"{args.dataset_name}_edgelist.csv"), index = False)
    df_nodelabel.to_csv(os.path.join(args.dir, f"{args.dataset_name}_node_labels.csv"), index = False)
    with open(os.path.join(args.dir, f"{args.dataset_name}_meta.json"),"w") as file:
        meta = {"num_classes": num_classes}
        json.dump(meta, file, indent = 4)

