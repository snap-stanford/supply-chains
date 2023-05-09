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

geo_dict = {'ARGENTINA': 32,
 'AUSTRALIA': 36,
 'AUSTRIA': 40,
 'AZERBAIJAN': 31,
 'BANGLADESH': 50,
 'BELARUS': 112,
 'BELGIUM': 56,
 'BHUTAN': 64,
 'BOSNIA AND HERZEGOVINA': 70,
 'BOTSWANA': 72,
 'BRAZIL': 76,
 'CANADA': 124,
 'CHILE': 152,
  #this is an accordance with the dataset, and not 
  #an indication of my personal views, similar to any
  #other omissions and labellings throughout 
 'CHINA HONGKONG': 156,
 'CHINA MAINLAND': 156,
 'CHINA TAIWAN': 156,
 'COLOMBIA': 170,
 'COSTA RICA': 188,
 'CZECHIA': 203,
 "CÔTE D'IVOIRE": 384,
 'DENMARK': 208,
 'DJIBOUTI': 262,
 'ECUADOR': 218,
 'EGYPT': 818,
 'FRANCE': 251,
 'GERMANY': 276,
 'GREECE': 300,
 'INDIA': 699,
 'INDONESIA': 360,
 'IRELAND': 372,
 'ISRAEL': 376,
 'ITALY': 381,
 'JAPAN': 392,
 'KAZAKHSTAN': 398,
 'KENYA': 404,
  # 'KOSOVO',
 'KUWAIT': 414,
 'LIECHTENSTEIN': 757,
 'LITHUANIA': 440,
 'MALAYSIA': 458,
 'MALTA': 470,
 'MEXICO': 484,
 'MOLDOVA': 498,
 'NAMIBIA': 516,
 'NEPAL': 524,
 'NETHERLANDS': 528,
 'NEW ZEALAND': 554,
 'NIGERIA': 566,
 'NORTH MACEDONIA': 807,
 'OMAN': 512,
 'PAKISTAN': 586,
 'PANAMA': 591,
 'PERU': 604,
 'PHILIPPINES': 608,
 'POLAND': 616,
 'PORTUGAL': 620,
 'PUERTO RICO': 842,
 'QATAR': 634,
 'RUSSIA': 643,
 'SAINT BARTHELEMY': 652,
 'SAUDI ARABIA': 682,
 'SINGAPORE': 702,
 'SOUTH AFRICA': 710,
 'SOUTH KOREA': 410,
 'SPAIN': 724,
 'SRI LANKA': 144,
 'SWEDEN': 752,
 'SWITZERLAND': 757,
 'TANZANIA': 834,
 'THAILAND': 764,
 'TUNISIA': 788,
 'TURKEY': 792,
 'UGANDA': 800,
 'UKRAINE': 804,
 'UNITED ARAB EMIRATES': 784,
 'UNITED KINGDOM': 826,
 'UNITED STATES': 842,
 'URUGUAY': 858,
 'UZBEKISTAN': 860,
 'VIETNAM': 704,
 'VIRGIN ISLANDS (BRITISH)': 92,
 'ZIMBABWE': 716}

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
    note: wanted to implement this with prettier code, but this optimises performance noticeably 
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
    
#new version (construction site status)
def aggregate_sc(csv_file = "./data/Hitachi/index_hs6.csv", aggregation_type = "product", hs_level = 6, country_map = {}):
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
    t = 0
    for i in tqdm(range(n)):
        hs6_product, start_date, end_date, weight, currency = hs6_products[i], start_dates[i], end_dates[i], weight_list[i], currency_list[i]
        supplier, buyer = supplier_list[i], buyer_list[i]
        
        try:
            hybrid_entity = entity_extractor(supplier, buyer, hs6_product)
        except: 
            t += 1 
            continue 
    
        year_weights = get_transaction_years(start_date, end_date)
        
        for year in year_weights: 
            if (hybrid_entity in supply_chain_dict[year]):
                supply_chain_dict[year][hybrid_entity]["weight"] += year_weights[year] * weight 
                supply_chain_dict[year][hybrid_entity]["currency"] += year_weights[year] * currency
            else: 
                supply_chain_dict[year][hybrid_entity] = {"weight": year_weights[year] * weight,
                                                        "currency": year_weights[year] * currency}
    
    print(f"Percent of Data Excised: {t / n * 100}%")
    return supply_chain_dict
    
def get_Hitachi_data(data_dir = "./data/Hitachi", aggregation_type = "product", hs_level = 6):
    product_codes_file = os.path.join(data_dir, "hs_category_description.csv")
    country_codes_file = os.path.join(data_dir, "country_region.csv")
    company_codes_file = os.path.join(data_dir, "group_subsidiary_site.csv")
    trade_data_file = os.path.join(data_dir, "index_hs6.csv")
          
    country_map = get_hitachi_countries(country_codes_file, company_codes_file)
    product_map = get_hitachi_products(product_codes_file)
    trading_map = aggregate_sc(trade_data_file, aggregation_type, hs_level, country_map)
    
    return country_map, product_map, trading_map
    
if __name__ == "__main__":

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
        
    