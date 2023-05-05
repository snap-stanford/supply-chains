""" utility functions for reading the BACI international
trade dataset and aggregating quantities (e.g. weight, price). 
"""

import csv
import json
import os 
import glob
import sys
from tqdm import tqdm
import pandas as pd
import argparse

def read_country_codes(csv_file = "../data/BACI/country_codes_V202301.csv"):
    """
    reads file on BACI country codes, mapping them to globally standardised ISO country codes
    
    Args:
        csv_file (str): path to BACI country codes file
    Returns:
         dict: a lookup dictionary mapping the country code (key) to a value dictionary
         composed of the country name and ISO 2-digit & 3-digit alpha representations. 
    """
    code2country = {}
    with open(csv_file, "r") as file: 
        reader = csv.reader(file)
        for line_ix, line in enumerate(reader):
            if (line_ix == 0): continue #header 
            code, country_name_abb, country_name_full, iso_alpha2, iso_alpha3 = line 
            code2country[int(code)] = {"name": country_name_full, "iso_alpha2": iso_alpha2, "iso_alpha3": iso_alpha3}
        
    return code2country

def read_product_codes(csv_file = "../data/BACI/product_codes_HS17_V202301.csv"):
    """
    reads BACI file on product codes, mapping each HS6 code to its specific description
    
    Args:
        csv_file (str): path to the product codes file
    Returns:
         dict: a lookup dictionary mapping the Harmonized System (HS) 6-digit codes found 
               in the BACI data to their corresponding descriptions
    """
    code2product = {}
    with open(csv_file,"r") as file:
        reader = csv.reader(file)
        for line_ix, line in enumerate(reader):
            if (line_ix == 0): continue #header 
            product_code, product_description = line 
            code2product[int(product_code)] = product_description 
            
    return code2product

def aggregate_global_products(csv_file = "../data/BACI/BACI_HS17_Y2018_V202301.csv", hs_level = 6):
    """
    reads a BACI yearly file on bilateral trade, aggregating across country pairs for each product
    
    Args:
        csv_file (str): path to a yearly BACI trade file
        hs_level (int): the number of HS digits we use to represent products 
                        i.e. level of product granularity. Should be in [2,4,6].
        
    Returns:
         dict: an econometrics dictionary from HS6 product code to the aggregated 
               currency flow (in current US dollars) and weight (in metric tonnes) in global trade
             
    """
    df = pd.read_csv(csv_file)
    n = len(df)
    product_list, cash_list, weight_list = list(df["k"]), list(df["v"]), list(df["q"])
    
    econometrics_dict = {}
    print("#### Aggregating the Global Trade Data #####")
    for i in tqdm(range(n)):
        product, currency, weight = product_list[i], cash_list[i], weight_list[i]
        
        #in current USD dollars
        currency = float(currency) * 1000 
        #in metric tons
        weight = float(weight) if "NA" not in weight else 0
        #whether we consider 2-digit, 4-digit, or full 6-digit HS codes
        product = product // 10**(6 - hs_level)
        
        if (product in econometrics_dict):
            econometrics_dict[product]["currency"] += currency
            econometrics_dict[product]["weight"] += weight
        else:
            econometrics_dict[product] = {"currency": currency, "weight": weight}
            
    return econometrics_dict
    
def aggregate_countries(csv_file = "../data/BACI/BACI_HS17_Y2018_V202301.csv"):
    """
    [TODO] implement this function to collect trading flow data between country pairs, aggregating over all exchanged products
    """
    df = pd.read_csv(csv_file)
    n = len(df)
    product_list, cash_list, weight_list = list(df["k"]), list(df["v"]), list(df["q"])
    country_i_list, country_j_list = list(df["i"]), list(df["j"]) 

    
def aggregate_countries_and_products():
    pass

def get_BACI_data(data_dir = "/home/jamin/supply-chains/data/BACI", year = 2020, hs_level = 6):
    """
    reads a BACI yearly file on bilateral trade, aggregating across country pairs for each product
    
    Args:
        data_dir (str): path--preferably absolute--to the directory storing the BACI data
        year (int): the year to extract bilateral trade data from (2017 - 2021, inclusive)
        
    Returns:
         tuple[dict]: Three dictionaries. (1) from internal county codes used in dataset to 
                     globally standardised ISO codes, (2) from product HS6 codes to
                     descriptions, (3) from hs6 product code to aggregated USD currency and 
                     weight in global trade flows of that product
    """
    years_covered = [2017, 2018, 2019, 2020, 2021]
    if (year not in years_covered): raise ValueError("year must be in 2017-2021")
    
    country_codes_file = os.path.join(data_dir, "country_codes_V202301.csv")
    product_codes_file = os.path.join(data_dir, "product_codes_HS17_V202301.csv")
    trade_data_file = os.path.join(data_dir, f"BACI_HS17_Y{year}_V202301.csv")
          
    country_map = read_country_codes(country_codes_file)
    product_map = read_product_codes(product_codes_file)
    trading_map = aggregate_global_products(trade_data_file, hs_level = hs_level)
    
    return country_map, product_map, trading_map
    
if __name__ == "__main__":
    
    country_map, product_map, trading_map = get_BACI_data("/home/jamin/supply-chains/data/BACI", year = 2020)
    for good in [860799, 853521, 382499, 853649, 853890]:
        print(good, trading_map[good])
    
    """
    basic testing 
    
    code2country = read_country_codes()
    code2product = read_product_codes()
    
    for code in list(code2country.keys())[:5]:
        print(code, code2country[code])
    print()
    for code in list(code2product.keys())[:5]:
        print(code, code2product[code])
    
    trade_data_dict = aggregate_global_products()
    products_hs6 = list(trade_data_dict.keys())
    for hs6_code in products_hs6[:5]:
        print(code2product[hs6_code], trade_data_dict[hs6_code])
    """
    
    
    