""" comparing the Hitachi data with the global trade data from BACI
"""

import read_BACI
import read_Hitachi
import matplotlib.pyplot as plt 
import numpy as np
import scipy
import sys
import json 
from tqdm import tqdm

if __name__ == "__main__":
    
    #year = int(sys.argv[1])
    hs_level = int(sys.argv[1])
    agg_type = sys.argv[2]
    
    supply_chain_data_table = read_Hitachi.aggregate_sc(hs_level = hs_level, aggregation_type = agg_type)
    
    baci_products = set()
    hitachi_products = set()
    for year in [2019, 2020, 2021]:
        #country_map, product_map, globalised_data = read_BACI.get_BACI_data(year = year, hs_level = hs_level, aggregation_type = agg_type)
        supply_chain_data = supply_chain_data_table[year]
        #baci_products = baci_products.union(set(globalised_data.keys()))
        hitachi_products = hitachi_products.union(set(supply_chain_data.keys()))
        print(f"Number of Hitachi Products in Year {year}: {len(hitachi_products)}")
        
    #common_products = baci_products.intersection(hitachi_products)
    
    #print(f"Number of BACI Products: {len(baci_products)}")
    print(f"Number of Hitachi Products: {len(hitachi_products)}")
    #print(f"Number of Joint Products: {len(common_products)}")
    
    """
    with open("missing.json","w") as file: 
        missing_data = {}
        missing_data["HITACHI"] = [(prod, product_map[prod]) for prod in list(baci_products.difference(hitachi_products))] if hs_level == 6 else list(baci_products.difference(hitachi_products))
        missing_data["BACI"] = [prod for prod in list(hitachi_products.difference(baci_products))] if hs_level == 6 else list(hitachi_products.difference(baci_products))
        
        json.dump(missing_data, file, indent = 4)

    currency_dict = {}
    weight_dict = {}
    print(f"#### Comparing HS{hs_level} Product-Level Econometrics in Hitachi vs. BACI ({year}) ####")
    for product in tqdm(common_products):
        sc_info = supply_chain_data[product]
        globalised_info = globalised_data[product]
        
        if (sc_info["weight"] != 0 and globalised_info["weight"] != 0):
            weight_dict[product] = {"BACI": globalised_info["weight"], "HITACHI": sc_info["weight"]}
        if (sc_info["currency"] != 0 and globalised_info["currency"] != 0):
            currency_dict[product] = {"BACI": globalised_info["currency"], "HITACHI": sc_info["currency"]}
    
    for product, weight in sorted(weight_dict.items(),
                                 key = lambda item: item[1]["HITACHI"] / item[1]["BACI"], reverse = False)[:50]:
        coverage_rate = weight["HITACHI"] / weight["BACI"] * 100
        product_name = product_map[product] if hs_level == 6 else product
        print(f"Product HS{hs_level}: {product} | Product Name: {product_name} | Hitachi Coverage: {coverage_rate}")
    """
    