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
import argparse
from constants import geo_dict

# parsing the year
def parse_date(date_str: str) -> tuple[int]:
    """
    helper function: takes a string of the form year_month_day
    and returns an integer for each time value.
    """
    year, month, day = date_str.split("-")
    return int(year), int(month), int(day)

def is_leap_year(year: int) -> bool:
    """
    helper function: returns whether a year (int) is a leap year (True) or not (False)
    """
    if (year % 4 != 0 or (year % 100 == 0 and year % 400 != 0)):
        return False 
    return True

def get_hitachi_products(csv_file = "./data/Hitachi/hs_category_description.csv"):
    """
    mapping from Hitachi's hs products to descriptions,
    at the 2, 4, and 6-digit levels
    """
    df = pd.read_csv(csv_file)
    n = len(df)
    hs6_list, category_list = list(df["hs6"]), list(df["category"])
    subcategory_list, description_list = list(df["sub_category"]), list(df["description"])
    
    code2product = {}
    for i in range(n):
        hs6, category, subcategory, description = hs6_list[i], category_list[i], subcategory_list[i], description_list[i]
        hs4 = hs6 // 10**2
        hs2 = hs6 // 10**4
        code2product[int(hs6)] = description
        code2product[int(hs4)] = subcategory 
        code2product[int(hs2)] = category
        
    return code2product
   
def get_hitachi_countries_util(csv_file = "./data/Hitachi/country_region.csv"):
    """
    helper function: mapping from country ids in the Hitachi dataset to the corresponding name, and globalised ISO codes
    """
    df = pd.read_csv(csv_file)
    n = len(df)
    id_list, iso_2_list, iso_3_list = (list(df[col]) for col in ["country_id", "iso_2digit_alpha", "iso_3digit_alpha"])
    name_list, abb_list, region_list = (list(df[col]) for col in ["country_name_full", "country_name_abbreviation", "region"])

    code2country = {}
    for i in range(n):
        id, iso_2, iso_3, name, abb, region = int(id_list[i]), iso_2_list[i], iso_3_list[i], name_list[i], abb_list[i], region_list[i]
        code2country[id] = {"name": name, "iso_alpha2": iso_2, "iso_alpha3": iso_3, "region": region}
        
    return code2country 

def get_hitachi_companies_util(csv_file = "./data/Hitachi/group_subsidiary_site.csv"):
    """
    helper function: mapping from company site ids in the Hitachi dataset to the corresponding country ids they are located in
    """
    df = pd.read_csv(csv_file)
    n = len(df)
    code2company = {}
    
    #parcel into null and int 999.0 (ambiguous) cases
    irreg_df = df[(df["company_site_country_id"].isna()) | (df["company_site_country_id"] == 999.0)]
    for company_site_id, company_site_country_id, company_site_country in zip(irreg_df["company_site_id"], irreg_df["company_site_country_id"], irreg_df["company_site_country"]):
        try:
            code2company[company_site_id] = geo_dict[company_site_country]
        except:
            continue 
            
    #normal cases
    norm_df = df[(~df["company_site_country_id"].isna()) & (df["company_site_country_id"] != 999.0)]
    for company_site_id, company_site_country_id, company_site_country in zip(norm_df["company_site_id"], norm_df["company_site_country_id"], norm_df["company_site_country"]):
        code2company[company_site_id] = int(company_site_country_id)
        
    return code2company
    
def get_hitachi_countries(country_csv = "./data/Hitachi/country_region.csv", company_csv = "./data/Hitachi/group_subsidiary_site.csv"):
    """
    composing the two functions above to directly map from company site ids to the country name & ISO codes
    """
    countryid_2_country = get_hitachi_countries_util(country_csv)
    siteid_2_countryid = get_hitachi_companies_util(company_csv)
    
    #chain the two dictionaries together
    siteid_2_country = {}
    for siteid in siteid_2_countryid.keys():
        country_id = siteid_2_countryid[siteid]
        if (country_id in countryid_2_country):
            siteid_2_country[siteid] = countryid_2_country[country_id]
    return siteid_2_country
    
def get_transaction_years(start_date: str, end_date: str) -> dict:
    """
    [add documentation]
    """
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

def get_entity_constructor(aggregation_type, country_map, hs_level):
    """
    returns a tuple of the desired entity combination (a subset of {exporter, importer, product}) to be
    used for aggregating transactions
    
    note: wanted to implement this with prettier code, but this (the stratified
    use of lambda functions) optimises performance noticeably 
    """
    if (aggregation_type == "exporter"):
        return lambda exporter, importer, product: country_map[exporter]["iso_alpha3"]
    if (aggregation_type == "importer"):
        return lambda exporter, importer, product: country_map[importer]["iso_alpha3"]
    if (aggregation_type == "product"):
        return lambda exporter, importer, product: int(str(int(product))[:hs_level])
    if (aggregation_type == "exporter_importer"):
        return lambda exporter, importer, product: (country_map[exporter]["iso_alpha3"], country_map[importer]["iso_alpha3"])
    if (aggregation_type == "exporter_product"):
        return lambda exporter, importer, product: (country_map[exporter]["iso_alpha3"], int(str(int(product))[:hs_level]))
    if (aggregation_type == "importer_product"):
        return lambda exporter, importer, product: (country_map[importer]["iso_alpha3"], int(str(int(product))[:hs_level]))
    if (aggregation_type == "exporter_importer_product"):
        return lambda exporter, importer, product: (country_map[exporter]["iso_alpha3"], country_map[importer]["iso_alpha3"], int(str(int(product))[:hs_level]))
    raise ValueError("invalid aggregation type")
    
def aggregate_sc(csv_file = "./data/Hitachi/index_hs6.csv", aggregation_type = "product", hs_level = 6, country_map = {}, filter_unknown = False):
    """
    reads the Hitachi index file, aggregating under the specified entity combination (agagregation_type)
    
    Args:
        csv_file (str): path to the index_hs6 Hitachi file
        aggregation_type (str): specifies the entity combinations for which we calculate trade flows.
                                Should take the form [entity1]_[entity2]_ ..., where each entity is among 
                               ["exporter","importer","product"] and ordered as such (e.g. no product_importer)
        hs_level (int): the number of HS digits we use to represent products 
                        i.e. level of product granularity. Should be in [2,4,6].
        country_map (dict): maps from country id in the dataset to the standardised ISO codes. can be empty
                        if not aggregating for countries (i.e. aggregation_type is product)
        
    Returns:
         dict: an econometrics dictionary from entities (e.g. country, product; see aggregation_type) to
                 aggregated currency flow (in current US dollars) and weight (in metric tonnes) in global trade
             
    """
    df = pd.read_csv(csv_file)
    n = len(df)
    
    start_dates, end_dates = list(df["st"]), list(df["et"])
    hs6_products, weight_list, currency_list = list(df["hs6"]), list(df["weight_sum"]), list(df["amount_sum"])
    supplier_list, buyer_list = list(df["supplier_id"]), list(df["buyer_id"])
    
    all_years = set([parse_date(date_str)[0] for date_str in start_dates])
    all_years = all_years.union(set([parse_date(date_str)[0] for date_str in end_dates]))
    supply_chain_dict = {year: {} for year in all_years}
    
    entity_extractor = get_entity_constructor(aggregation_type, country_map, hs_level)
    print(f"#### Aggregating Hitachi Supply Chain for HS{hs_level} Entity Level {aggregation_type} ####")
    num_missing = 0
    
    #iterate through all transaction entries in the index table, and extract their weight, currency, and prodcut
    for i in tqdm(range(n)):
        hs6_product, start_date, end_date, weight, currency = hs6_products[i], start_dates[i], end_dates[i], weight_list[i], currency_list[i]
        supplier, buyer = supplier_list[i], buyer_list[i]
        
        try: #see whether the entity combination can be referenced in the database
            hybrid_entity = entity_extractor(supplier, buyer, hs6_product)
            
            #if filter_unknown, then harshly excise any entries where either supplier or buyer countries are unavailable 
            #if (filter_unknown == True and country_map[supplier]["iso_alpha3"] == country_map[buyer]["iso_alpha3"]):
             #   num_missing += 1; continue 
                
        except: 
            num_missing += 1; continue 
        
        #update the aggregation dict across all years the transaction took place
        year_weights = get_transaction_years(start_date, end_date)
        for year in year_weights: 
            if (hybrid_entity in supply_chain_dict[year]):
                supply_chain_dict[year][hybrid_entity]["weight"] += year_weights[year] * weight * 0.001
                supply_chain_dict[year][hybrid_entity]["currency"] += year_weights[year] * currency
            else: 
                supply_chain_dict[year][hybrid_entity] = {"weight": year_weights[year] * weight * 0.001,
                                                        "currency": year_weights[year] * currency}
    
    print(f"Percent of Data Excised: {num_missing / n * 100}%")
    return supply_chain_dict
    
def get_Hitachi_data(data_dir = "./data/Hitachi", aggregation_type = "product", hs_level = 6):
    """
    reads the data files from a specified Hitachi data directory, aggregating at the specified entity level
    
    Args:
        data_dir (str): path--preferably absolute--to the directory storing the BACI data
        aggregation_type (str): species the entity combinations for which we calculate trade flows (see aggregate)
        hs_level (int): The granularity of HS products (whether to use first 2, 4, or 6 digits)
                        
    Returns:
         tuple[dict]: Three dictionaries. (1) from internal county codes used in Hitachi dataset to 
                     globally standardised ISO codes, (2) from product HS6 codes to
                    Hitachi descriptions, (3) from hs6 product code to aggregated USD currency and 
                     weight in global trade flows of that product
    """
    product_codes_file = os.path.join(data_dir, "hs_category_description.csv")
    country_codes_file = os.path.join(data_dir, "country_region.csv")
    company_codes_file = os.path.join(data_dir, "group_subsidiary_site.csv")
    trade_data_file = os.path.join(data_dir, "index_hs6.csv")
          
    country_map = get_hitachi_countries(country_codes_file, company_codes_file)
    product_map = get_hitachi_products(product_codes_file)
    trading_map = aggregate_sc(trade_data_file, aggregation_type, hs_level, country_map)
    
    return country_map, product_map, trading_map
    
if __name__ == "__main__":
    """
    for testing out the code and printing out a results snippet
    """
    country_map, product_map, trading_map = get_Hitachi_data(aggregation_type = sys.argv[1], hs_level = 6)
    
    trading_map = trading_map[2020]
    keys = list(trading_map.keys())
    sampled_keys = np.random.choice(range(len(keys)), size = 20, replace = False)
    for key in sampled_keys:
        key = keys[key]
        if (key in product_map):
            print(f"{product_map[key]}: {trading_map[key]}")
        else:
            print(f"{key}: {trading_map[key]}")
        
    