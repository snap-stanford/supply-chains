""" comparing the Hitachi data with the global trade data from BACI
"""

import read_BACI
import read_Hitachi
import matplotlib.pyplot as plt 
import numpy as np
import scipy
import sys
from tqdm import tqdm
import argparse

def plot_differences(currency_points, weight_points, year, fname = "trade_combined.jpg", hs_level = 6):
    marker_size = {2: 10, 4: 1, 6: 0.2}[hs_level]
    fig, ax = plt.subplots(figsize = (10,5), nrows = 1, ncols = 2, tight_layout = True)
    #plot currrency flow (x-axis being supply chain data, y-axis being the BACI global data)
    ax[0].scatter(x = [c[0] for c in currency_points], y = [c[1] for c in currency_points], 
                  color = "teal", s = marker_size, marker = "h")
    ax[0].set_xlabel("Currency flow in Hitachi Data (in ?)", fontsize = 10)
    ax[0].set_ylabel("Currency flow in BACI Data (in current USD)", fontsize = 10)
    ax[0].set_title(f"Global HS{hs_level} Product-Level Currency Flow ({year})", fontsize = 11)
    #plot the regression for currency flow 
    
    
    #plot shipping weight (x-axis being supply chain data, y-axis being the BACI global data)
    ax[1].scatter(x = [c[0] for c in weight_points], y = [c[1] for c in weight_points], 
                  color = "slateblue", s = marker_size, marker = "h")
    ax[1].set_xlabel("Total weight in Hitachi Data (in ?)", fontsize = 10)
    ax[1].set_ylabel("Total weight in BACI Data (in tonnes)", fontsize = 10)
    ax[1].set_title(f"Global HS{hs_level} Product-Level Trade Weight ({year})", fontsize = 11)
    #plot the regression for shipping weight
    
    for i in [0,1]:
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
    
    plt.savefig(fname)
    plt.show()
    
def plot_differences_with_regression(currency_points, weight_points, year, fname = "trade_combined.jpg", hs_level = 6, include_boxes = True):
    marker_size = {2: 10, 4: 1, 6: 0.2}[hs_level]
    fig, ax = plt.subplots(figsize = (10,5), nrows = 1, ncols = 2, tight_layout = True)
    text_weight_x = 0.35; text_weight_y = 0.1
    #plot currrency flow (x-axis being supply chain data, y-axis being the BACI global data)
    currency_x, currency_y = [c[0] for c in currency_points], [c[1] for c in currency_points]
    ax[0].scatter(x = currency_x, y = currency_y, color = "teal", s = marker_size, marker = "h")
    ax[0].set_xlabel("Currency flow in Hitachi Data (in ?)", fontsize = 10)
    ax[0].set_ylabel("Currency flow in BACI Data (in current USD)", fontsize = 10)
    ax[0].set_title(f"Global HS{hs_level} Product-Level Currency Flow ({year})", fontsize = 11)
    
    #plot the regression for currency flow 
    slope, intercept, r_val, p_value, standard_error = scipy.stats.linregress(np.log10(currency_x), np.log10(currency_y))
    line_x = np.linspace(np.log10(min(currency_x)), np.log10(max(currency_x)), 100)
    line_y = slope * line_x + intercept
    ax[0].plot(10**line_x, 10**line_y, linewidth = 1.5, color = "black")
    t = ax[0].text(x = 10**(text_weight_x * np.log10(min(currency_x)) + (1 - text_weight_x) * np.log10(max(currency_x))), 
               y = 10**(text_weight_y * np.log10(max(currency_x)) + (1 - text_weight_y) * np.log10(min(currency_y))), 
               s = f"Log(y)={slope:.2f}Log(x)+{intercept:.2f}\nR\u00b2={r_val**2:.3f}", fontsize = 8)
    if include_boxes == True: t.set_bbox(dict(facecolor='orange', alpha=0.15))
    
    #plot shipping weight (x-axis being supply chain data, y-axis being the BACI global data)
    weight_x, weight_y = [c[0] for c in weight_points], [c[1] for c in weight_points]
    ax[1].scatter(x = weight_x, y = weight_y, color = "slateblue", s = marker_size, marker = "h")
    ax[1].set_xlabel("Total weight in Hitachi Data (in ?)", fontsize = 10)
    ax[1].set_ylabel("Total weight in BACI Data (in tonnes)", fontsize = 10)
    ax[1].set_title(f"Global HS{hs_level} Product-Level Trade Weight ({year})", fontsize = 11)
    
    #plot the regression for shipping weight
    slope, intercept, r_val, p_value, standard_error = scipy.stats.linregress(np.log10(weight_x), np.log10(weight_y))
    line_x = np.linspace(np.log10(min(weight_x)), np.log10(max(weight_x)), 100)
    line_y = slope * line_x + intercept
    ax[1].plot(10**line_x, 10**line_y, linewidth = 1.5, color = "black")
    t = ax[1].text(x = 10**(text_weight_x * np.log10(min(weight_x)) + (1 - text_weight_x) * np.log10(max(weight_x))), 
               y = 10**(text_weight_y * np.log10(max(weight_y)) + (1 - text_weight_y) * np.log10(min(weight_y))), 
               s = f"Log(y)={slope:.2f}Log(x)+{intercept:.2f}\nR\u00b2={r_val**2:.3f}", fontsize = 8)
    if include_boxes == True: t.set_bbox(dict(facecolor='orange', alpha=0.15))
    
    for i in [0,1]:
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
    
    plt.savefig(fname)
    plt.show()
        
def run_stats_testing(currency_points, weight_points, year):
    currency_pearson = scipy.stats.pearsonr(np.log([c[0] for c in currency_points]), np.log([c[1] for c in currency_points]))
    weight_pearson = scipy.stats.pearsonr(np.log([c[0] for c in weight_points]), np.log([c[1] for c in weight_points]))
    
    currency_spearman = scipy.stats.spearmanr([c[0] for c in currency_points], [c[1] for c in currency_points])
    weight_spearman = scipy.stats.spearmanr([c[0] for c in weight_points], [c[1] for c in weight_points])
    
    print(f"\n#### Correlation Metrics Between Product-Level Hitachi and BACI Data in {year} ####")
    print(f"Pearson Coefficient for log(Currency Flow): {currency_pearson.statistic:.3f}")
    print(f"Pearson Coefficient for log(Shipping Weight): {weight_pearson.statistic:.3f}")
    print(f"Spearman Coefficient for Currency Flow: {currency_spearman.statistic:.3f}")
    print(f"Spearman Coefficient for Shipping Weight: {weight_spearman.statistic:.3f}")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Parse directory paths for data downloading.')
    parser.add_argument('--year', nargs='?', help='Year of data comparison', default = 2020)
    parser.add_argument('--hs_digits', nargs='?', help='Number of HS digits to group products by', default = 6)
    args = parser.parse_args()
    
    year = args.year
    hs_level = args.hs_digits
    
    supply_chain_data = read_Hitachi.aggregate_sc_products(hs_level = hs_level, use_redshift = True)[year]
    country_map, product_map, globalised_data = read_BACI.get_BACI_data(year = year, hs_level = hs_level)
    
    common_products = set(supply_chain_data.keys()).intersection(set(globalised_data.keys()))
    
    currency_points = []
    weight_points = []
    #iterate through all products, and excising ones with non-reported values
    print(f"#### Comparing HS{hs_level} Product-Level Econometrics in Hitachi vs. BACI ({year}) ####")
    for product in tqdm(common_products):
        sc_info = supply_chain_data[product]
        globalised_info = globalised_data[product]
        
        if (sc_info["weight"] != 0 and globalised_info["weight"] != 0):
            currency_points.append((sc_info["weight"], globalised_info["weight"]))
        if (sc_info["currency"] != 0 and globalised_info["currency"] != 0):
            weight_points.append((sc_info["currency"], globalised_info["currency"]))
        
    #save out scatterplot for the currency flow and global trade weight
    plot_differences_with_regression(currency_points, weight_points, year, hs_level = hs_level, fname = f"./images/trade_{year}_hs{hs_level}.jpg")
    #statistical testing 
    run_stats_testing(currency_points, weight_points, year)