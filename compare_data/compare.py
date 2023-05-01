""" comparing the Hitachi data with the global trade data from BACI
"""

import read_BACI
import read_Hitachi
import matplotlib.pyplot as plt 
import numpy as np
import scipy

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
        
    #plot currrency flow (x-axis being supply chain data, y-axis being the BACI global data)
    plt.scatter([c[0] for c in currency_points], [c[1] for c in currency_points], color = "green", s = 0.1)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("log(currency flow) in SC Data (in ?)", fontsize = 10)
    plt.ylabel("log(currency flow) in BACI Data (in current USD)", fontsize = 10)
    plt.title("Dataset Comparison for Global Product-Level Currency Flow", fontsize = 12)
    plt.savefig("trade_currency.jpg")
    plt.show()
    
    plt.clf()
    #plot shipping weight (x-axis being supply chain data, y-axis being the BACI global data)
    plt.scatter([c[0] for c in weight_points], [c[1] for c in weight_points], color = "blue", s = 0.1)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("log(total weight) in SC Data (in ?)", fontsize = 10)
    plt.ylabel("log(total weight) in BACI Data (in tonnes)", fontsize = 10)
    plt.title("Dataset Comparison for Global Product-Level Trade Weight", fontsize = 12)
    plt.savefig("trade_weight.jpg")
    plt.show()
    
    #statistical testing 
    currency_pearson_r = scipy.stats.pearsonr([c[0] for c in weight_points], [c[1] for c in weight_points]).statistic
    weight_pearson_r = scipy.stats.pearsonr([c[0] for c in weight_points], [c[1] for c in weight_points]).statistic
    
    currency_spearman_r = scipy.stats.spearmanr([c[0] for c in weight_points], [c[1] for c in weight_points]).statistic
    weight_spearman_r = scipy.stats.spearmanr([c[0] for c in weight_points], [c[1] for c in weight_points]).statistic
    
    print("Pearson Coefficient for Product Currency Flow", currency_pearson_r)
    print("Pearson Coefficient for Product Shipping Weight", weight_pearson_r)
    
    print("Spearman Coefficient for Product Currency Flow", currency_spearman_r)
    print("Spearman Coefficient for Product Shipping Weight", weight_spearman_r)
    
    #potentially look into log-scaled coefficients as well, and run scipy regression
