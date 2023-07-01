"""
this file extracts mapping tables (for companies, products, countries) for the entities
that appear in the Hitachi logistic_data
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
from pycountry_convert import country_name_to_country_alpha3

def get_country_table(redshift):
    """
    Generates a map from country names as they appear in logistic_data to country
    ISO alpha-3 codes, which are the international convention. 
    
    Args:
        redshift (RedshiftClass): An instance of Redshift Class, 
    
    Returns:
        dict[str:str]: A dictionary from country names (key) to ISO alpha-3 codes (value).
    """
    
    #query for countries that appear as importers and exporters for transactions
    df_importer = redshift.query_df("select dest_country as country from logistic_data GROUP BY country")
    df_exporter = redshift.query_df("select orig_country as country from logistic_data GROUP BY country")

    #iterate through all countries and see whether they appear in the pycountry Python library
    all_countries = set(df_importer["country"]).union(set(df_exporter["country"]))
    country_to_iso3 = {}
    for name in all_countries:
        try:
            country_name_tokens = [word.lower().capitalize() if word != "AND" else word.lower() for word in name.split(" ")]
            country_name = " ".join(country_name_tokens)
            iso3 = country_name_to_country_alpha3(country_name, cn_name_format="default")
            country_to_iso3[name] = iso3
        except:
            continue
    
    #manual labelling of countries whose names cannot be readily mapped.
    country_to_iso3[""] = "" #for error avoidance as countries can appear blank for some transactions
    country_to_iso3["KOSOVO"] = "XKX"
    country_to_iso3["GUINEA-BISSAU"] = "GNB"
    country_to_iso3["CHINA MAINLAND"] = "CHN"
    country_to_iso3["COCOS (KEELING) ISLANDS"] = "CCK"
    country_to_iso3["SAINT HELENA, ASCENSION AND TR"] = "SHN"
    country_to_iso3["CONGO (KINSHASA)"] = "COD"
    country_to_iso3["CHINA TAIWAN"] = "TWN"
    country_to_iso3["CONGO (BRAZZAVILLE)"] = "COG"
    country_to_iso3["REUNION"] = "REU"
    country_to_iso3["NETHERLANDS ANTILLES"] = "NLD"
    country_to_iso3["CHINA MACAO"] = "MAC"
    country_to_iso3["CÔTE D'IVOIRE"] = "CIV"
    country_to_iso3["SAINT BARTHELEMY"] = "BLM"
    country_to_iso3["SAINT VINCENT AND THE GRENADIN"] = "VCT"
    country_to_iso3["CHINA HONGKONG"] = "HKG"
    country_to_iso3["VIRGIN ISLANDS (BRITISH)"] = "VGB"
    
    assert set(all_countries) == set(country_to_iso3.keys())
    return country_to_iso3
    
def get_company_table(redshift):
    """
    Generates a map from company IDs in logistic_data to their corresponding names / titles.

    Args:
        redshift (RedshiftClass): An instance of Redshift Class 
    
    Returns:
        company2id (dict[str:str]): A dictionary from company titles to IDs
        id2company (dict[str:str]): A dictionary from company IDs to titles
    """
    
    #query for all companies that appear as either buyers or sellers
    query = f"select COUNT(*) as count, supplier_id, supplier_t from logistic_data GROUP BY supplier_id, supplier_t"
    df_supplier = redshift.query_df(query)
    query = f"select COUNT(*) as count, buyer_id, buyer_t from logistic_data GROUP BY buyer_id, buyer_t"
    df_buyer = redshift.query_df(query)
    
    #concatenate into one table, remove duplicates
    df_combined = pd.concat([df_supplier.rename(columns = {"supplier_id":"company_id", "supplier_t":"company_t"}), 
                        df_buyer.rename(columns = {"buyer_id":"company_id", "buyer_t":"company_t"})])
    df_combined = df_combined.groupby(by=["company_id","company_t"]).sum().reset_index()
    
    #generate {ID -> company} map and inverse {company -> ID} map
    id2company_preliminary = {} #will need post-processing (see below code) 
    company2id = {}
    rows = [list(df_combined[row_name]) for row_name in ["company_id", "company_t", "count"]]
    
    for company_id, company_t, count in zip(*rows):
        company2id[company_t] = company_id #many titles can map to the same ID value
        if (company_id in id2company_preliminary):
            id2company_preliminary[company_id].append((company_t, count)) #keep track of all exitant titles
        else:
            id2company_preliminary[company_id] = [(company_t,count)]
        
    #for each ID, select company title (e.g. samsung vs SAMSUNG) appearing most often among the transactions
    id2company = {}
    for key in list(id2company_preliminary.keys()):
        id2company[key] = sorted(id2company_preliminary[key], key = lambda item: item[1], reverse = True)[0][0]

    return company2id, id2company

def get_product_table(redshift):
    """
    Generates a map from Harmonized System (HS) product codes (at the 2, 4, and 6 digit levels)
    to text descriptions. Here, 2, 4, and 6 digits correspond to chapters, headings, and subheadings,
    with an increasing level of granularity. 
    
    Args:
        redshift (RedshiftClass): An instance of Redshift Class
    
    Returns:
        dict[str:str]: A dictionary mapping between HS product codes (in string form) and
                        corresponding text descriptions. The codes should be zero-padded
                        at the front, with a total length of either 2,4, or 6.
                        
    """
    
    #query for products that appear in the Hitachi dataset and their descriptions
    query = f"select * from hs_category_description"
    df_products = redshift.query_df(query)
    row_names = ["hs6","category","sub_category","description"]
    rows = [list(df_products[row_name]) for row_name in row_names]
    
    #iterate through all listed products extracted from the query
    hscode_to_product = {}
    for hs6, category, sub_category, description in zip(*rows):
        hs6_code = hs6.zfill(6) #all six digits
        hs4_code = str(int(hs6) // 10**2).zfill(4) #first four digits
        hs2_code = str(int(hs6) // 10**4).zfill(2) #first two digits

        if (hs6_code not in hscode_to_product):
            hscode_to_product[hs6_code] = description
        if (hs4_code not in hscode_to_product):
            hscode_to_product[hs4_code] = sub_category
        if (hs2_code not in hscode_to_product):
            hscode_to_product[hs2_code] = category

    #manual labellings
    hscode_to_product["77"] = "Reserved for possible future use"
    hscode_to_product["98"] = "Special Classification Provisions"
    
    return hscode_to_product 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Extracting tables for companies, countries, and products in logistic_data')
    parser.add_argument('--dir', nargs='?', help='Where to store the company, country, and product tables', default = "./")
    parser.add_argument('--rs_login', nargs=2, help='Username and password for RedShift, in that order', default = None)
    args = parser.parse_args()
    
    #create the RedShift class instance based on the user-provided login, and retrieve tables
    rs = RedshiftClass(args.rs_login[0], args.rs_login[1])
    product_table = get_product_table(rs)
    company2id, id2company = get_company_table(rs)
    company_table = {"company2id": company2id, "id2company": id2company}
    country_table = get_country_table(rs)
    
    #save out the tables
    with open(os.path.join(args.dir, "hitachi_product_mappers.json"),"w") as file:
        json.dump(product_table, file, indent = 4)
    with open(os.path.join(args.dir, "hitachi_company_mappers.json"),"w") as file:
        json.dump(company_table, file, indent = 4)
    with open(os.path.join(args.dir, "hitachi_country_mappers.json"),"w") as file:
        json.dump(country_table, file, indent = 4)

    print("Saved out the tables to {}".format(args.dir))