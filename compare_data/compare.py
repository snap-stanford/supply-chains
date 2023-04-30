""" comparing the Hitachi data with the global trade data from BACI
"""

import read_BACI
import read_Hitachi
import matplotlib.pyplot as plt 
import numpy as np

if __name__ == "__main__":
    
    supply_chain_data = read_Hitachi.aggregate_sc_products()
    supply_chain_data_2021 = supply_chain_data[2021]
    country_map, product_map, globalised_data_2021 = read_BACI.get_BACI_data(year = 2021)
    
    common_products = set(supply_chain_data_2021.keys()).intersection(set(globalised_data_2021.keys()))
    
    currency_points = []
    weight_points = []
    
    for product in common_products:
        sc_info = supply_chain_data_2021[product]
        globalised_info = globalised_data_2021[product]
        
        if (sc_info["weight"] != 0 and globalised_info["weight"] != 0):
            currency_points.append((sc_info["weight"], globalised_info["weight"]))
        if (sc_info["currency"] != 0 and globalised_info["currency"] != 0):
            weight_points.append((sc_info["currency"], globalised_info["currency"]))
        
    #plot them
    plt.scatter([c[0] for c in currency_points], [c[1] for c in currency_points], color = "green", s = 0.1)
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig("out1.jpg")
    plt.show()
    
    plt.clf()
    plt.scatter([c[0] for c in weight_points], [c[1] for c in weight_points], color = "blue", s = 0.1)
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig("out2.jpg")
    plt.show()
