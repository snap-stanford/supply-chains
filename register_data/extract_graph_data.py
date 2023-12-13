"""
This file is for querying logistic_data and turning the result into an edge spreadsheet
(where each row represents a time-stamped, aggregated transaction between two firms or nodes).
Run python extract_graph_data.py -h to see details on argument passing

NOTE: FOR USE ON THE HITACHI JUPYTERHUB SERVERS
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

def retrieve_timestamped_data(redshift, start_date = "2019-01-01", length_timestamps = 1, num_timestamps = 1, 
                              hs6_restriction = [850760], verbose = True):
    """
    This will obtain a dataframe of time-stamped edges, which represent aggregated transactions between firms
    
    Args:
        redshift (RedshiftClass):  An instance of Redshift Class 
        start_date (str): The earliest date (YY-MM-DD) from which to retrieve transactions
        length_timestamps (int): The number of days to aggregate per time stamp
        num_timestamps (int): The number of time stamps to retrieve from logistic_data (i.e. the last day transactions
                              will be retrieved from is length_timestamps * num_timestamps - 1 days after start_date)
        hs6_restriction (List[int]): A list of HS6 products to narrow the transactions search. If None, no restrictions.
        verbose (bool): Whether to print out status updates (to console) from the retrieval 
        
    Returns:
        pd.Dataframe: A dataframe where each row is an aggregated, time-stamped transactions between a supplier and buyer,
                       with details such as total_amount, total_weight, etc.
    """
    
    PRIMARY_KEY = 'date, supplier_id, buyer_id, quantity, weight, price, amount, hs_code'
    AGGREGATION_KEY = 'time_stamp, hs6, supplier_id, buyer_id'
    K = float(length_timestamps) # floating point version to enable float (not integer) SQL division
    max_day = length_timestamps * num_timestamps
    
    #restrict transactions to a certain list of product codes if requested
    product_condition = ""
    if (hs6_restriction != None):
        for code in hs6_restriction: product_condition += f"hs_code like '{code}%' OR "
        product_condition = "({}) AND".format(product_condition[:-3]) #get rid of last OR at the tail

    #restrict the transactions to the specified time period, and deduplicate 
    query = f"select {PRIMARY_KEY}, DATEDIFF(day, '{start_date}', date) as time_interval, COUNT(*) as count, \
    COUNT(DISTINCT id) as num_ids from logistic_data WHERE {product_condition} time_interval \
    BETWEEN 0 AND {max_day-1} GROUP BY {PRIMARY_KEY}, time_interval"
    
    #aggregate based on the cadence specified by length_timestamps (number of days between consecutive time stamps) 
    query = f"select CEILING((time_interval + 1) / {K}) - 1 as time_stamp, SUBSTRING(hs_code, 1, 6) as hs6,\
    supplier_id, buyer_id, COUNT(*) as bill_count, SUM(quantity) as total_quantity, SUM(amount) as total_amount,\
    SUM(weight) as total_weight from ({query}) GROUP BY {AGGREGATION_KEY} ORDER BY time_stamp"
    
    date_format = '%Y-%m-%d'
    final_date = datetime.datetime.strptime(start_date, date_format) + datetime.timedelta(days = int(max_day) - 1)
    final_date = final_date.strftime(date_format)
    if verbose == True: print("Querying logistic_data between {} and {}".format(start_date, final_date))
    start_t = time.time()
    df = rs.query_df(query)
    end_t = time.time()
    if verbose == True: print("Retrieved {} rows from logistic_data in {:.3f} seconds".format(len(df), end_t - start_t))
    return df

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Extracting graph data from the transactions in logistic_data')
    parser.add_argument('--rs_login', nargs=2, help='Username and password for RedShift, in that order', default = None)
    parser.add_argument('--start_date', help='Starting day of form YY-MM-DD from which to collect data', default = "2019-01-01")
    parser.add_argument('--length_timestamps', help='Number of days to aggregate per time stamp', default = 1, type = int)
    parser.add_argument('--num_timestamps', help='Number of time stamps to retrieve', default = 1, type = int)
    parser.add_argument('--fname', help='Path to the .csv file for storing the resulting data', default = None)
    parser.add_argument('--use_titles', help = 'if provided, data uses company titles instead of IDs', action='store_true')
    args = parser.parse_args()
    
    rs = RedshiftClass(args.rs_login[0], args.rs_login[1])
    df = retrieve_timestamped_data(rs, args.start_date, args.length_timestamps, args.num_timestamps,
                                  hs6_restriction = None) 
    df = df[df["hs6"].str.match('^(?!00)[0-9]{6}')] #check for valid HS6 product codes via regex
    
    #create the company ID -> name mapper, and replace the IDs in the dataframe with company titles
    if (args.use_titles == True):
        id2company = retrieve_Hitachi_table(name = "company")["id2company"]
        company_ids = list(id2company.keys())
        df_companies = pd.DataFrame.from_dict({"company_id": company_ids,
                                       "company_t": [id2company[id] for id in company_ids]})
        
        #replace the company supplier IDs
        df = pd.merge(df, df_companies, left_on = "supplier_id", right_on = "company_id", how = "left")
        df = df.rename(columns = {"company_t": "supplier_t"}).drop(columns = {"company_id","supplier_id"})
        #replace the company buyer IDs
        df = pd.merge(df, df_companies, left_on = "buyer_id", right_on = "company_id", how = "left")
        df = df.rename(columns = {"company_t": "buyer_t"}).drop(columns = {"company_id","buyer_id"})
    
    print(df.head(5))
    print(df.tail(5))
    df.to_csv(args.fname, index = False)