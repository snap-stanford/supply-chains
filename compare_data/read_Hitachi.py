""" utility functions for reading the Hitachi supply chain
index data, and aggregating quantities (e.g. weight, price)
"""

import csv
import json
import sys
from tqdm import tqdm
import pandas as pd
import os 
import glob
from datetime import date
import numpy as np

# parsing the year
def parse_date(date_str: str) -> tuple[int]:
    year, month, day = date_str.split("-")
    return int(year), int(month), int(day)

def is_leap_year(year: int) -> bool:
    if (year % 4 != 0 or (year % 100 == 0 and year % 400 != 0)):
        return False 
    return True

def get_hitachi_products(csv_file):
    pass

def get_hitachi_countries(csv_file):
    pass

def get_transaction_years(start_date: str, end_date: str) -> dict:
    start_year, start_month, start_day = parse_date(start_date)
    end_year, end_month, end_day = parse_date(end_date)
    if (start_year == end_year):
        return {start_year: 1}
    
    #otherwise, proportionalise the metrics among transaction years
    year_weights = {}
    
    #calculate number of days the transaction took place in the starting year
    year_weights[start_year] = (date(start_year, 12, 31) - date(start_year, start_month, start_day)).days + 1
    #calculate number of days the transaction took place in the ending year
    year_weights[end_year] = (date(end_year, end_month, end_day) - date(end_year, 1, 1)).days + 1
    #include all days for middle years of transaction 
    for year in range(start_year + 1, end_year):
        year_weights[year] = 365 if is_leap_year(year) == False else 366
    
    #normalise so that the weights add up to 1
    total_transaction_days = (date(end_year, end_month, end_day) - date(start_year, start_month, start_day)).days + 1
    for year in year_weights: 
        year_weights[year] = year_weights[year] / total_transaction_days
    
    return year_weights 
    
def aggregate_sc_products(csv_file = "./data/Hitachi/index_hs6.csv", hs_level = 6):
    """
    [TODO] documentation
            hs_level (int): the number of HS digits we use to represent products 
                        i.e. level of product granularity. Should be in [2,4,6].
    """
    df = pd.read_csv(csv_file)
    n = len(df)
    
    hs6_products = list(df["hs6"])
    start_dates = list(df["st"])
    end_dates = list(df["et"])
    weight_list = list(df["weight_sum"])
    currency_list = list(df["amount_sum"])
    
    all_years = set([parse_date(date_str)[0] for date_str in start_dates])
    all_years = all_years.union(set([parse_date(date_str)[0] for date_str in end_dates]))
    supply_chain_dict = {year: {} for year in all_years}
    
    print("#### Aggregating Hitachi Supply Chain for HS6 Products ####")
    for i in tqdm(range(n)):
        hs6_product, start_date, end_date, weight, currency = hs6_products[i], start_dates[i], end_dates[i], weight_list[i], currency_list[i]
        try:
            hs_product = int(hs6_product) // 10 ** (6 - hs_level)
            #hs_product = int(hs_product)
        except: 
            continue 
        
        #hs_product = int(str(hs_product)[:hs_level])
        year_weights = get_transaction_years(start_date, end_date)
        for year in year_weights: 
            if (hs_product in supply_chain_dict[year]):
                supply_chain_dict[year][hs_product]["weight"] += year_weights[year] * weight 
                supply_chain_dict[year][hs_product]["currency"] += year_weights[year] * currency
            else: 
                supply_chain_dict[year][hs_product] = {"weight": year_weights[year] * weight,
                                                        "currency": year_weights[year] * currency}
    
    return supply_chain_dict
    
def get_Hitachi_data(data_dir = "./data/BACI", aggregation_type = "product", hs_level = 6):
    pass
    
if __name__ == "__main__":
    supply_chain_dict = aggregate_sc_products(hs_level = 6)[2020]
    keys = list(supply_chain_dict.keys())
    print(f"Number of Unique Products in Hitachi SC Data: {len(keys)}")
    sampled_keys = np.random.choice(keys, size = 50, replace = False)
    for key in sampled_keys:
        print(f"{key}: {supply_chain_dict[key]}")
    
    """
    for i in np.random.choice(len(trading_map), size = 100, replace = False):
        key_name = tuple(product_map[entity] if entities[idx] == "product" else country_map[entity]["name"] for idx, entity in enumerate(keys[i]))
        print(key_name, trading_map[keys[i]])
    
    
    
    for hs6_product in sorted(list(supply_chain_dict[2020].keys()))[:20]:
        print(hs6_product, supply_chain_dict[2020][hs6_product])
    """
         
    
    