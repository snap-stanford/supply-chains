"""
this file contains utility functions for reading the Hitachi logistic_data, 
and aggregating quantities across products, countries, etc. for global, bilateral trade flows 
** note: excludes transactions that constitute domestic trade
"""

import os 
import glob
import json
import argparse
import warnings
import sys
sys.path.append("/opt/libs")
from crystal_api.apiclass import APIClass, RedshiftClass
from crystal_api.apikeyclass import APIkeyClass
from dotenv import load_dotenv
import pandas as pd
import time
import datetime
import numpy as np

def get_days_between(date_1, date_2, date_format = "%Y-%m-%d"):
    """
    calculates the number of days between two dates (default format YY-MM-DD)
    """
    date_1_time = datetime.datetime.strptime(date_1, date_format)
    date_2_time = datetime.datetime.strptime(date_2, date_format)
    return (date_2_time - date_1_time).days

def retrieve_Hitachi_table(name = "company", dir = "."):
    """
    Retrieves the Hitachi tables saved as .json files by the script extract_tables.py (see code for details)
    
    Args:
        name (str): The specific reference table to retrieve, must be out of {company, country, product}
        dir (str): The path to the directory the table is stored in 
    
    Returns:
        dict: The retrieved reference table. Note that the company table comprises both the forward (id2company)
        and inverse (company2id) mappings. 
    """
    
    with open(os.path.join(dir, f"hitachi_{name}_mappers.json"),"r") as file:
        table = json.load(file) 
    return table

def get_aggregation_key(aggregation_type):
    #orig_country and dest_country represent the exporter and importer of a transaction, respectively
    entity2key = {"exporter": "orig_country", "importer": "dest_country", "product": "product"}
    valid_entities = set(entity2key.keys())
    entities = aggregation_type.split("_")
    for entity in entities: 
        if entity not in valid_entities: raise ValueError(f"{entity} not valid. must be among {valid_entities}")
        
    return ",".join(entity2key[entity] for entity in entities)
    
def aggregate_logistic(rs, aggregation_type = "product", hs_level = 6, start_date = "2019-01-01", 
                       end_date = "2019-12-31", maps_dir = "./", verbose = False):
    """
    Args:
        rs (RedshiftClass):  An instance of Redshift Class 
        aggregation_type (str): specifies the entity combinations for which we calculate trade flows.
                                Should take the form [entity1]_[entity2]_ ..., where each entity is among 
                               ["exporter","importer","product"] and ordered as such (e.g. no product_importer)
        hs_level (int), maps_dir (str): see get_Hitachi_date() below
        start_date (str): the start date from which transactions are aggregated, inclusive
        end_date (str): the end date from which transacted are aggregated, inclusive 
        verbose (bool): Whether to print out status updates (to console) from the retrieval 
        
    Returns:
        dict: See (2) in the Returns documentation for get_Hitachi_data()
    """
    num_days_between = get_days_between(start_date, end_date)
    country_map = retrieve_Hitachi_table(name = "country", dir = maps_dir)
    
    #deduplicate transactions
    PRIMARY_KEY = 'date, supplier_id, buyer_id, quantity, weight, price, amount, hs_code'
    AGGREGATION_KEY = get_aggregation_key(aggregation_type)
    product_filter = "product not like '% %' AND product not like '00%' AND len(product) = 6 \
                    AND product not like '%,%'" #selecting valid HS6 codes
    country_filter = "orig_country != dest_country AND orig_country != '' AND dest_country != ''"

    #restrict the transactions to the specified time period, and deduplicate 
    query = f"select {PRIMARY_KEY}, SUBSTRING(hs_code, 1, 6) as product, COUNT(*) as count, max(orig_country) as \
    orig_country, max(dest_country) as dest_country, COUNT(DISTINCT id) as num_ids from logistic_data \
    WHERE {product_filter} AND DATEDIFF(day, '{start_date}', date) BETWEEN 0 AND {num_days_between} \
    AND {country_filter} GROUP BY {PRIMARY_KEY}, product"
    
    #aggregate transactions under the desired entity combinations
    query = f"select {AGGREGATION_KEY}, COUNT(*) as bill_count, SUM(quantity) as total_quantity, SUM(amount) as total_amount, \
    SUM(weight) as total_weight from ({query}) GROUP BY {AGGREGATION_KEY}"
    
    #query the RedShift API
    if verbose == True: print("Querying logistic_data between {} and {}".format(start_date, end_date))
    start_t = time.time()
    df = rs.query_df(query).fillna(0)
    end_t = time.time()
    if verbose == True: print("Retrieved {} rows from logistic_data in {:.3f} seconds".format(len(df), end_t - start_t))
    
    #process the returned dataframe
    if ("orig_country" in df.columns):
        df["orig_country"] = df["orig_country"].apply(lambda name: country_map[name])
    if ("dest_country" in df.columns):
        df["dest_country"] = df["dest_country"].apply(lambda name: country_map[name])
    if ("product" in df.columns):
        df["product"] = df["product"].apply(lambda code: code[:hs_level])
        df = df.groupby(by = AGGREGATION_KEY.split(",")).sum(numeric_only = True).reset_index()
    
    #transform the dataframe into a dictionary from entities to corresponding trade values
    df["key"] = [",".join(entities) for entities in zip(*[list(df[key]) for key in AGGREGATION_KEY.split(",")])]
    df_rows = [list(df[row]) for row in ["key","bill_count","total_quantity","total_amount","total_weight"]]
    trade_flow_map = {}
    for key, bill_count, quantity, amount, weight in zip(*df_rows):
        entities = key.split(",")
        metrics = {"bill_count": bill_count, "currency": amount, "quantity": quantity, "weight": weight}
        trade_flow_map[tuple(entities) if len(entities) > 1 else entities[0]] = metrics
    
    return trade_flow_map
    
def get_Hitachi_data(rs, aggregation_type = "product", hs_level = 6, year = 2020, maps_dir = "./"):
    """
    reads the Hitachi logistic_data, aggregating transactions at the specified entity level
    
    Args:
        aggregation_type (str): species the entity for which we collate global trade flows (see aggregate_logistic)
        hs_level (int): The granularity of HS products (whether to use first 2, 4, or 6 digits)
        year (int): Year from which to collect data (from 2019 - 2023, inclusive)
        maps_dir (str): path to the directory storing the Hitachi tables retrieved by ../temporal_graph/extract_tables.py
                        
    Returns:
         tuple[dict]: Two dictionaries. (1) from product HS6 codes to
                    Hitachi descriptions, (2) from entity (e.g. HS6 product) to aggregated amount (in USD)
                    and weight (in tonnes) in global trade flows of that entity
    """
    assert year in list(range(2019,2023+1)), "year must be between 2019 and 2023, inclusive" 
    product_map = retrieve_Hitachi_table("product", dir = maps_dir)
    start_date, end_date = f"{year}-01-01", f"{year}-12-31"
    trade_flow_map = aggregate_logistic(rs, aggregation_type, hs_level, start_date, end_date, maps_dir,
                                       verbose = True)
    
    return product_map, trade_flow_map 

if __name__ == "__main__":
    """
    testing out this file's functionality in the command line
    """
    parser = argparse.ArgumentParser(description='Extracting graph data from the transactions in logistic_data')
    parser.add_argument('--rs_login', nargs=2, help='Username and password for RedShift, in that order', default = None)
    parser.add_argument('--hs_digits', nargs='?', help='Number of HS digits to group products by', default = 6,
                       type = int)
    parser.add_argument('--agg_type', nargs='?', help= 'entity level representations', default = "product")
    parser.add_argument('--year', nargs='?', help='Year of data comparison', default = 2020, type = int)
    args = parser.parse_args()
    
    rs = RedshiftClass(args.rs_login[0], args.rs_login[1])
    product_map, trade_flow_map = get_Hitachi_data(rs, args.agg_type, args.hs_digits, args.year, "./")
    keys = list(trade_flow_map.keys())
    sample_keys = np.random.choice(range(len(keys)), size = 10)
    for key_id in sample_keys:
        key = keys[key_id]
        print(key, trade_flow_map[key])