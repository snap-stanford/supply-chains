"""
This file is for querying logistic_data and turning the result into an edge spreadsheet
(where each row represents a time-stamped, aggregated transaction between two firms or nodes).
Run python extract_graph_data.py -h to see details on argument passing

dev note: limited to a small subset of products for now (e.g. battery-related codes) due to scale
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
from constants import BATTERY_RELATED_CODES

def retrieve_timestamped_data(redshift, start_date = "2019-01-01", length_timestamps = 1, num_timestamps = 1, 
                              hs6_codes = [850760], verbose = True):
    
    PRIMARY_KEY = 'date, supplier_id, buyer_id, quantity, weight, price, amount, hs_code'
    SECONDARY_KEY = 'time_stamp, hs6, supplier_id, buyer_id'
    K = float(length_timestamps)
    max_day = length_timestamps * num_timestamps
    
    #restrict transactions to a certain list of product codes
    product_condition = ""
    for code in hs6_codes: product_condition += f"hs_code like '{code}%' OR "
    product_condition = product_condition[:-3] #get rid of last OR at the tail

    #restrict the transactions to the specified time period, and deduplicate 
    query = f"select {PRIMARY_KEY}, DATEDIFF(day, '{start_date}', date) as time_interval, COUNT(*) as count, \
    COUNT(DISTINCT id) as num_ids from logistic_data WHERE ({product_condition}) AND time_interval \
    BETWEEN 0 AND {max_day-1} GROUP BY {PRIMARY_KEY}, time_interval"
    
    #aggregate based on the cadence specified by length_timestamps (number of days between consecutive time stamps) 
    query = f"select CEILING((time_interval + 1) / {K}) * {K} as time_stamp, SUBSTRING(hs_code, 1, 6) as hs6,\
    supplier_id, buyer_id, COUNT(*) as bill_count, SUM(quantity) as total_quantity, SUM(amount) as total_amount,\
    SUM(weight) as total_weight from ({query}) GROUP BY {SECONDARY_KEY} ORDER BY time_stamp"
    
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
    args = parser.parse_args()
    
    rs = RedshiftClass(args.rs_login[0], args.rs_login[1])
    df = retrieve_timestamped_data(rs, args.start_date, args.length_timestamps, args.num_timestamps,
                                  hs6_codes = BATTERY_RELATED_CODES)
    print(df.head(10))
    df.to_csv(args.fname, index = False)