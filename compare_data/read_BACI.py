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
import numpy as np

def read_country_codes(csv_file = "./data/BACI/country_codes_V202301.csv"):
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

def read_product_codes(csv_file = "./data/BACI/product_codes_HS17_V202301.csv"):
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

#def get_combined_entity(exporter, importer, product, aggregation_type):
#    entity_map = {"exporter": exporter, "importer": importer, "product": product}
#    entities = [entity.strip() for entity in aggregation_type.split("_")]
#    if len(entities) == 1: return entity_map[entities[0]]
#    return (entity_map[entity] for entity in entities)

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

def aggregate(csv_file =  "./data/BACI/BACI_HS17_Y2018_V202301.csv", aggregation_type = "product", hs_level = 6, country_map = {}):
    """
    reads a BACI yearly file on bilateral trade, aggregating across country pairs for each product
    
    Args:
        csv_file (str): path to a yearly BACI trade file
        aggregation_type (str): specifies the entity combinations for which we calculate trade flows.
                                Should take the form [entity1]_[entity2]_ ..., where each entity is among 
                               ["exporter","importer","product"] and ordered as such (e.g. no product_importer)
                                
        hs_level (int): the number of HS digits we use to represent products 
                        i.e. level of product granularity. Should be in [2,4,6].
        
    Returns:
         dict: an econometrics dictionary from entities (e.g. country, product; see aggregation_type) to
                 aggregated currency flow (in current US dollars) and weight (in metric tonnes) in global trade
             
    """
    df = pd.read_csv(csv_file)
    n = len(df)
    product_list, cash_list, weight_list = list(df["k"]), list(df["v"]), list(df["q"])
    exporter_list, importer_list = list(df["i"]), list(df["j"]) 
    entities_to_extract = aggregation_type.strip().split("_")
    
    econometrics_dict = {}

    entity_extractor = get_entity_constructor(aggregation_type, country_map)
    print(f"#### Aggregating Baci Data for HS{hs_level} at HS{hs_level} Entity Level {aggregation_type} ####")
    for i in tqdm(range(n)):
        product, currency, weight = product_list[i], cash_list[i], weight_list[i]
        exporter, importer = exporter_list[i], importer_list[i]
    
        #in current USD dollars
        currency = float(currency) * 1000 
        #in metric tons
        weight = float(weight) if "NA" not in weight else 0
        #whether we consider 2-digit, 4-digit, or full 6-digit HS codes
        product = product // 10**(6 - hs_level)
        
        hybrid_entity = entity_extractor(exporter, importer, product)
        if hybrid_entity in econometrics_dict:
            econometrics_dict[hybrid_entity]["currency"] += currency
            econometrics_dict[hybrid_entity]["weight"] += weight
        else:
            econometrics_dict[hybrid_entity] = {"currency": currency, "weight": weight}
             
    return econometrics_dict

def get_BACI_data(data_dir = "./data/BACI", aggregation_type = "product", year = 2020, hs_level = 6):
    """
    reads a BACI yearly file on bilateral trade, aggregating across country pairs for each product
    
    Args:
        data_dir (str): path--preferably absolute--to the directory storing the BACI data
        year (int): the year to extract bilateral trade data from (2017 - 2021, inclusive)
        aggregation_type (str): species the entity combinations for which we calculate trade flows (see aggregate)
                        
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
    trading_map = aggregate(trade_data_file, aggregation_type, hs_level, country_map)
    
    return country_map, product_map, trading_map
    
if __name__ == "__main__":
    
    aggregation_type = "exporter_importer_product" if (len(sys.argv) == 1) else sys.argv[1]
    entities = aggregation_type.strip().split("_")
    country_map, product_map, trading_map = get_BACI_data("/home/jamin/supply-chains/data/BACI", 
                                                          aggregation_type = aggregation_type, year = 2020, hs_level = 6)
    keys = list(trading_map.keys())
    print(f"Number of Unique Keys under Entity Combination {aggregation_type}: {len(keys)}")
    for i in np.random.choice(range(len(trading_map)), size = 20, replace = False):
        key_name = tuple(product_map[entity] if entities[idx] == "product" else entity for idx, entity in enumerate(keys[i]))
        print(key_name, trading_map[keys[i]])
    
    
    