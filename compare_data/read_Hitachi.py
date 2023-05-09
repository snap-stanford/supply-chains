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
    name_list, region_list = (list(df[col]) for col in ["country_name_full", "region"])

    code2country = {}
    for i in range(n):
        id, iso_2, iso_3, name, region = id_list[i], iso_2_list[i], iso_3_list[i], name_list[i], region_list[i]
        code2country[int(id)] = {"name": name, "iso_alpha2": iso_2, "iso_alpha3": iso_3, "region": region}
    return code2country 

def get_hitachi_companies_util(csv_file = "./data/Hitachi/group_subsidiary_site.csv"):
    """
    helper function: mapping from company site ids in the Hitachi dataset to the corresponding country ids they are located in
    """
    df = pd.read_csv(csv_file)
    n = len(df)
    code2company = {}
    for company_site_id, company_site_country_id in zip(df["company_site_id"], df["company_site_country_id"]):
        try:
            code2company[company_site_id] = int(company_site_country_id)
        except:
            continue
    return code2company
    
def get_hitachi_countries(country_csv = "./data/Hitachi/country_region.csv", company_csv = "./data/Hitachi/group_subsidiary_site.csv"):
    """
    composing the two functions above to directly map from company site ids to the country ISO 3 code
    """
    countryid_2_country = get_hitachi_countries_util(country_csv)
    siteid_2_countryid = get_hitachi_companies_util(company_csv)
    
    #chain the two dictionaries together
    siteid_2_country = {}
    for siteid in siteid_2_countryid.keys():
        country_id = siteid_2_countryid[siteid]
        if (country_id in countryid_2_country):
            siteid_2_country[siteid] = countryid_2_country[country_id]["iso_alpha3"]
    return siteid_2_country
    
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

#deprecated
def aggregate_sc_products(csv_file = "./data/Hitachi/index_hs6.csv", hs_level = 6):
    """
    [TODO] documentation
            hs_level (int): the number of HS digits we use to represent products 
                        i.e. level of product granularity. Should be in [2,4,6].
    """
    df = pd.read_csv(csv_file)
    n = len(df)
    start_dates, end_dates = list(df["st"]), list(df["et"])
    hs6_products, weight_list, currency_list = list(df["hs6"]), list(df["weight_sum"]), list(df["amount_sum"])
    
    all_years = set([parse_date(date_str)[0] for date_str in start_dates])
    all_years = all_years.union(set([parse_date(date_str)[0] for date_str in end_dates]))
    supply_chain_dict = {year: {} for year in all_years}
    
    print(f"#### Aggregating Hitachi Supply Chain for HS{hs_level} Products ####")
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
    
def get_entity_constructor(aggregation_type, country_map):
    """
    note: wanted to implement this with prettier code, but this optimises performance noticeably 
    """
    if (aggregation_type == "exporter"):
        return lambda exporter, importer, product: country_map[exporter]["iso_alpha3"]
    if (aggregation_type == "importer"):
        return lambda exporter, importer, product: country_map[importer]["iso_alpha3"]
    if (aggregation_type == "product"):
        return lambda exporter, importer, product: product
    if (aggregation_type == "exporter_importer"):
        return lambda exporter, importer, product: (country_map[exporter]["iso_alpha3"], country_map[importer]["iso_alpha3"])
    if (aggregation_type == "exporter_product"):
        return lambda exporter, importer, product: (country_map[exporter]["iso_alpha3"], product)
    if (aggregation_type == "importer_product"):
        return lambda exporter, importer, product: (country_map[importer]["iso_alpha3"], product)
    if (aggregation_type == "exporter_importer_product"):
        return lambda exporter, importer, product: (country_map[exporter]["iso_alpha3"], country_map[importer]["iso_alpha3"], product)
    raise ValueError("invalid aggregation type")
    
#new version (construction site status)
def aggregate_sc(csv_file = "", aggregation_type = "", hs_level = 6, country_map = {}):
    df = pd.read_csv(csv_file)
    n = len(df)
    
    start_dates, end_dates = list(df["st"]), list(df["et"])
    hs6_products, weight_list, currency_list = list(df["hs6"]), list(df["weight_sum"]), list(df["amount_sum"])
    suppler_list = list("suppler_id")
    
    all_years = set([parse_date(date_str)[0] for date_str in start_dates])
    all_years = all_years.union(set([parse_date(date_str)[0] for date_str in end_dates]))
    supply_chain_dict = {year: {} for year in all_years}
    
    print(f"#### Aggregating Hitachi Supply Chain for HS{hs_level} Products ####")
    for i in tqdm(range(n)):
        hs6_product, start_date, end_date, weight, currency = hs6_products[i], start_dates[i], end_dates[i], weight_list[i], currency_list[i]
        try: #how about try extractor 
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
    
def get_Hitachi_data(data_dir = "./data/Hitachi", aggregation_type = "product", hs_level = 6):
    product_codes_file = os.path.join(data_dir, "hs_category_description.csv")
    country_codes_file = os.path.join(data_dir, "country_region.csv")
    company_codes_file = os.path.join(data_dir, "group_subsidiary_site.csv")
    trade_data_file = os.path.join(data_dir, "index_hs6.csv")
          
    country_map = get_hitachi_countries(country_codes_file, company_codes_file)
    product_map = get_hitachi_products(product_codes_file)
    
    #trading_map = aggregate(trade_data_file, aggregation_type, hs_level, country_map)
    
    return country_map, product_map, trading_map
    
if __name__ == "__main__":
    supply_chain_dict = aggregate_sc_products(hs_level = 4)[2020]
    code2product = get_hitachi_products()
    
    
    keys = list(supply_chain_dict.keys())
    print(f"Number of Unique Products in Hitachi SC Data: {len(keys)}")
    sampled_keys = np.random.choice(keys, size = 20, replace = False)
    for key in sampled_keys:
        if (key in code2product):
            print(f"{code2product[key]}: {supply_chain_dict[key]}")
        else:
            print(f"{key}: {supply_chain_dict[key]}")
    
    country_map = get_hitachi_countries(country_csv = "./data/Hitachi/country_region.csv", company_csv = "./data/Hitachi/group_subsidiary_site.csv")
    sampled_keys = np.random.choice(list(country_map.keys()), size = 20, replace = False)
    print(f"Number of Company Site IDs with Country Tag: {len(list(country_map.keys()))}")
    for key in sampled_keys:
        print(key, country_map[key])
    
    