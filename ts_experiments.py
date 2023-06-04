import pandas as pd
import warnings
import time
import datetime
import sys
import os
import glob
import re
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec 
import numpy as np
import math
from scipy.stats.stats import pearsonr
sys.path.append("/opt/libs")
from apiclass import APIClass,RedshiftClass
from sc_experiments import *
from constants_and_utils import *

warnings.filterwarnings('ignore')
rs = RedshiftClass('zhiyin','Zhiyin123')

def get_transaction_df(PROD='battery', tx_type='supplier', companies=[]):
    '''
    Args:
        PROD (str or Tuple(str)): a string of bom key e.g. 'bms' or a tuple of strings that are general hs_codes e.g. ('850670', '482110',)
        tx_type (str): 'supplier' or 'buyer'
        companies (list(str)): company keywords for SQL query. If empty list, assume 'all' companies. 

    Returns:
        str: identifier name of the csv, such as 'samsung', 'lg', 'all', etc.
    '''
    assert (tx_type=='supplier' or tx_type=='buyer')
    
    # preprocess constants
    hs_codes = BOM[PROD] if PROD in BOM else PROD
    name = PROD if PROD in BOM else "baseline"
    tx_col = 'supplier_t' if tx_type=='supplier' else 'buyer_t'
    if len(companies) >= 1:
        csv_name = " ".join(re.findall("[a-zA-Z]+", companies[0]))
    else:
        csv_name = "all"
    
    # get transactions where company is supplying/buying PROD
    for hs in hs_codes:
        if len(companies) >= 1:
            cmp_string = f"{tx_col} like '{companies[0]}'" + "".join([f" or {tx_col} like '{company}'" for company in companies[1:]])
            query = f"select {PRIMARY_KEY}, COUNT(*) as count, COUNT(DISTINCT id) as num_ids from logistic_data where (hs_code like '{hs}%') and ({cmp_string}) GROUP BY {PRIMARY_KEY};"
        else:
            query = f"select {PRIMARY_KEY}, COUNT(*) as count, COUNT(DISTINCT id) as num_ids from logistic_data where (hs_code like '{hs}%') GROUP BY{PRIMARY_KEY};"
        df = rs.query_df(query)
        if df is not None:
            df = df.drop_duplicates()
            print(hs, '->', len(df), len(df.drop_duplicates()))
            df['hs_code'] = df['hs_code'].str[:6] # Standardize all hs_codes to first 6 digits
        else:
            # no results found for this query
            print(hs, '-> None')
            continue

        # save transactions
        outdir = f'./data/{csv_name}/'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        df.to_csv(f'./data/{csv_name}/{name}_{tx_type}_{csv_name}_{hs}.csv', index=False) 
        del df
    
    return csv_name

def load_df(name='battery', csv_name='samsung'):
    cur_dir = os.getcwd()
    path = f"{cur_dir}/data/{csv_name}/{name}*.csv" # use your path
    all_files = glob.glob(path)

    df = []
    for filename in all_files:
        df.append(pd.read_csv(filename, index_col=None, header=0))
    if len(df) == 0:
        return 
    df = pd.concat(df, axis=0, ignore_index=True)
    return df

def get_supply_buy_df(SUPPLY_PROD='battery', BUY_PROD='bms', csv_name='samsung'):
    '''
    Load in company csv_name's supply_df, buy_df for SUPPLY_PROD, BUY_PROD respectively.
    Returns:
        supply_df (pd.DataFrame)
        buy_df (pd.DataFrame)
        early_termination (bool): True if at least one of the supply, buy lengths <= THRESHOLD 
    '''
    supply_name = SUPPLY_PROD if SUPPLY_PROD in BOM else "baseline" 
    buy_name = BUY_PROD if BUY_PROD in BOM else "baseline"
    print(f"{csv_name}: supply name is {supply_name}; buy name is {buy_name}")
    
    supply_df = load_df(name=supply_name, csv_name=csv_name)
    buy_df = load_df(name=buy_name, csv_name=csv_name)
    
    # Check both dataframes are not None 
    if supply_df is None or buy_df is None:
        return supply_df, buy_df, True
    
    # Check both dataframes has at least one hs code with >= THRESHOLD transactions
    supply_max = max(supply_df.groupby(['hs_code'])['hs_code'].count())
    buy_max = max(buy_df.groupby(['hs_code'])['hs_code'].count())
    if supply_max <= THRESHOLD or buy_max <= THRESHOLD:
        return supply_df, buy_df, True
    
    # If all checking passed, print length of supply df and buy df 
    print(f"Supply len is {len(supply_df)}, buy len is {len(buy_df)}")
    return supply_df, buy_df, False

def add_transaction_time(supply_df, buy_df):
    '''
    Add daily, weekly, and monthly datetime columns.
    '''
    # convert date to datetime
    supply_df['datetime'] = pd.to_datetime(supply_df.date)
    buy_df['datetime'] = pd.to_datetime(buy_df.date)

    # extract month from date string
    supply_df['month'] = supply_df.date.apply(lambda x: x.rsplit('-', 1)[0])
    buy_df['month'] = buy_df.date.apply(lambda x: x.rsplit('-', 1)[0])

    # convert month to datetime
    supply_df['month_datetime'] = pd.to_datetime(supply_df.month)
    buy_df['month_datetime'] = pd.to_datetime(buy_df.month)

    # extract week from datetime.dt
    supply_df['week'] = supply_df.datetime.dt.strftime("%G-%V-1")
    buy_df['week'] = buy_df.datetime.dt.strftime("%G-%V-1")

    # convert week to datetime
    supply_df['week_datetime'] = supply_df.week.apply(lambda x: datetime.date.fromisocalendar(int(x.rsplit('-', 2)[0]), int(x.rsplit('-', 2)[1]), 1))
    buy_df['week_datetime'] = buy_df.week.apply(lambda x: datetime.date.fromisocalendar(int(x.rsplit('-', 2)[0]), int(x.rsplit('-', 2)[1]), 1))
    
    return supply_df, buy_df

def get_time_col(time):
    '''
    Mapping from time keyword to dataframe column name.
    '''
    if time.lower()=='daily':
        return 'datetime'
    elif time.lower()=='weekly':
        return 'week_datetime'
    elif time.lower()=='monthly':
        return 'month_datetime'
    raise Exception("Time window is not supported")

def save_fig(fig, outdir, filename):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    fig.savefig(outdir + filename, bbox_inches="tight")
    
def plot_time_versus_sale_purchase(supply_df, buy_df, SUPPLY_PROD='battery', BUY_PROD='bms', time='monthly', csv_name='samsung'):
    '''Plot sale info (quantity and amount) over time
    Args:
        time (str): supports 'daily', 'weekly', or 'monthly'
    '''
    time_col = get_time_col(time)
    supply_name = SUPPLY_PROD if SUPPLY_PROD in BOM else "baseline" 
    buy_name = BUY_PROD if BUY_PROD in BOM else "baseline" 
    
    # plot time-ly sales vs time-ly purchases
    supply_summary = supply_df.groupby(time_col)[['quantity', 'amount']].sum()
    buy_summary = buy_df.groupby(time_col)[['quantity', 'amount']].sum()

    # plot quantity
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax = axes[0]           
    ax.set_title(f'{time} sales of {supply_name}', fontsize=12)
    ax.plot(supply_summary.index.values, supply_summary.quantity.values)
    ax.set_ylabel('Total quantity', fontsize=12)
    ax = axes[1]
    ax.set_title(f'{time} purchases of {buy_name}', fontsize=12)
    ax.plot(buy_summary.index.values, buy_summary.quantity.values)
    ax.set_ylabel('Total quantity', fontsize=12)
    plt.show()
    
    outdir = f'./fig/{csv_name}/'
    filename = f'time_versus_sale_purchase_quantity_{supply_name}_{buy_name}_{time}_{csv_name}.jpg'
    save_fig(fig, outdir, filename)

    # plot amount
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax = axes[0]           
    ax.set_title(f'{time} sales of {supply_name}', fontsize=12)
    ax.plot(supply_summary.index.values, np.log(supply_summary.amount.values))
    ax.set_ylabel('Total amount, log ($)', fontsize=12)
    ax = axes[1]
    ax.set_title(f'{time} purchases of {buy_name}', fontsize=12)
    ax.plot(buy_summary.index.values, np.log(buy_summary.amount.values))
    ax.set_ylabel('Total amount, log ($)', fontsize=12)
    plt.show()
    
    outdir = f'./fig/{csv_name}/'
    filename = f'time_versus_sale_purchase_amount_{supply_name}_{buy_name}_{time}_{csv_name}.jpg'
    save_fig(fig, outdir, filename)

def compare_sale_purchase_quantity_per_hscode(supply_df, buy_df, SUPPLY_PROD='battery', BUY_PROD='bms', time='monthly', num_before=0, num_after=0, lag=0, do_plot=True, csv_name='samsung'):
    '''Calculate correlation for sale info (quantity) over time per supply hs_code, buy hs_code
    Args:
        time (str): supports 'daily', 'weekly', or 'monthly'
        num_before (int): nonnegative integer, see utils 'apply_smoothing'
        num_after (int): nonnegative integer, see utils 'apply_smoothing'
        lag (int): shift buy product backward for {lag} days, only supports 'daily' time level
    Returns:
        summary_df (pd.DataFrame): a single row that records summary statistics fo time series correlations
    '''
    # preprocess constants 
    supply_hs_codes = BOM[SUPPLY_PROD] if SUPPLY_PROD in BOM else SUPPLY_PROD
    buy_hs_codes = BOM[BUY_PROD] if BUY_PROD in BOM else BUY_PROD
    supply_name = SUPPLY_PROD if SUPPLY_PROD in BOM else "baseline" 
    buy_name = BUY_PROD if BUY_PROD in BOM else "baseline" 
    time_col = get_time_col(time)
    supply_df['hs_code_str'] = supply_df.hs_code.astype(str)  # convert HS code to str
    buy_df['hs_code_str'] = buy_df.hs_code.astype(str)  # convert HS code to str
    
    # add lags
    if time=='daily' and lag!=0:
        buy_df[time_col] += datetime.timedelta(days=lag)
        
    # prepare summary table
    summary_df = {}
    summary_df['group'] = csv_name
    
    for supply_hs in supply_hs_codes:
        sub_supply_df = supply_df[supply_df.hs_code_str.str.contains(supply_hs)]
        summary_df[f"# of supply txn {supply_name}_{supply_hs}"] = len(sub_supply_df)
        if len(sub_supply_df) <= THRESHOLD:
            for buy_hs in buy_hs_codes:
                summary_df[f"# of supply txn {buy_name}_{buy_hs}"] = None
            print(supply_hs, len(sub_supply_df))
        else:
            supply_summary = sub_supply_df.groupby(time_col).quantity.sum()
            
            # Apply smoothing: supply
            smooth_values = apply_smoothing(supply_summary.values, num_before=num_before, num_after=num_after)
            assert(len(supply_summary)==len(smooth_values))
            supply_summary.replace(supply_summary.values, smooth_values, inplace=True)
            supply_summary.fillna(0, inplace=True)
            
            if do_plot:
                # Plot: supply
                fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
                ax = axes[0]           
                ax.set_title(f'{time} sales of {supply_hs}', fontsize=12)
                ax.plot(supply_summary.index.values, supply_summary.values)
                ax.set_ylabel('Total quantity', fontsize=12)
                ax = axes[1]
                ax.set_title(f'{time} purchases of {buy_name}', fontsize=12)
                
            # Apply smoothing, calculate correlation: all buy
            summary_df[f"# of buy txn {buy_name}_all"] = len(buy_df)
            if len(buy_df) <= THRESHOLD:
                summary_df[f"corr {supply_name}_{supply_hs}, {buy_name}_all smooth_{num_before}_{num_after}"] = None
            else:
                buy_summary = buy_df.groupby(time_col).quantity.sum()
                # Apply smoothing: buy
                smooth_values = apply_smoothing(buy_summary.values, num_before=num_before, num_after=num_after)
                assert(len(buy_summary)==len(smooth_values))
                buy_summary.replace(buy_summary.values, smooth_values, inplace=True)
                buy_summary.fillna(0, inplace=True)
                
                # Calculate correlation: buy
                merged = pd.merge(supply_summary.rename('x'), buy_summary.rename('y'), 
                                          left_index=True, right_index=True, how='inner')
                if len(merged.x) < 2 or len(merged.y) < 2:
                    r, p = float("nan"), float("nan")
                else:
                    r, p = pearsonr(merged.x, merged.y)
                summary_df[f"corr {supply_name}_{supply_hs}, {buy_name}_all smooth_{num_before}_{num_after}"] = r                
                
            # Apply smoothing, calculate correlation: per buy hscode
            for buy_hs in buy_hs_codes:
                sub_buy_df = buy_df[buy_df.hs_code_str.str.contains(buy_hs)]
                summary_df[f"# of supply txn {buy_name}_{buy_hs}"] = len(sub_buy_df)
                if len(sub_buy_df) <= THRESHOLD:
                    print(buy_hs, len(sub_buy_df))
                    summary_df[f"corr {supply_name}_{supply_hs}, {buy_name}_{buy_hs} smooth_{num_before}_{num_after}"] = None
                else:
                    buy_summary = sub_buy_df.groupby(time_col).quantity.sum()
                    
                    # Apply smoothing: buy hscode
                    smooth_values = apply_smoothing(buy_summary.values, num_before=num_before, num_after=num_after)
                    assert(len(buy_summary)==len(smooth_values))
                    buy_summary.replace(buy_summary.values, smooth_values, inplace=True)
                    buy_summary.fillna(0, inplace=True)

                    # Calculate correlation: buy hscode
                    merged = pd.merge(supply_summary.rename('x'), buy_summary.rename('y'), 
                                      left_index=True, right_index=True, how='inner')
                    if len(merged.x) < 2 or len(merged.y) < 2:
                        r, p = float("nan"), float("nan")
                    else:
                        r, p = pearsonr(merged.x, merged.y)
                    print(buy_hs, len(sub_buy_df), 'r=%.3f (n=%d, p=%.3f)' % (r, len(merged), p)) # number of transaction, n is number of dates
                    summary_df[f"corr {supply_name}_{supply_hs}, {buy_name}_{buy_hs} smooth_{num_before}_{num_after}"] = r
                    
                    if do_plot:
                        # Normalize by mean to make plot easier
                        ax.plot(buy_summary.index.values, buy_summary.values / np.mean(buy_summary.values), label=buy_hs)
            if do_plot:
                ax.legend(bbox_to_anchor=(1,1))
                ax.set_ylabel('Total quantity (normalized)', fontsize=12)
                plt.show()
            
        if do_plot:
            # save figure 
            outdir = f'./fig/{csv_name}/'
            time_surfix = f"{time_col}" + f"_smooth_{num_before}_{num_after}" if num_before + num_after > 0 else ""
            filename = f'compare_sale_purchase_quantity_{supply_name}_{supply_hs}_{buy_name}_{time_surfix}_{csv_name}.jpg'
            save_fig(fig, outdir, filename)
    
    # remove lags
    if time=='daily' and lag!=0:
        buy_df[time_col] -= datetime.timedelta(days=lag)
        
    return summary_df

def get_hs_r_cnt_values(df):
    hs_codes = []
    r_values = []
    cnt_values = []
    for col in df.columns:
        if 'all' in col:
            continue
        if "# of supply txn battery_850760" in col:
            num_battery_txn = df[col].values[0]
        elif '#' in col:
            hs_codes.append(col[-6:])
            cnt_values.append(df[col].values[0])
        elif "corr" in col:
            r = df[col].values[0]
            if math.isnan(r):
                r = 0
            r_values.append(r)
        
    zipped = list(zip(hs_codes, r_values, cnt_values))
    sorted_zipped = sorted(zipped, key=lambda x: x[2])
    hs_codes, r_values, cnt_values = zip(*sorted_zipped)

    return (hs_codes, r_values, cnt_values), num_battery_txn

def plot_summary(company):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw = {'height_ratios':[16, 5]}, sharex=True)
    
    summary_smooth = pd.read_csv(f'./summary/{company}_smooth_15-15_summary.csv').set_index('group')
    (hs_codes, r_values, cnt_values), num_battery_txn = get_hs_r_cnt_values(summary_smooth)
    ax = axes[0]
    g = ax.barh(hs_codes, r_values, height=0.8)
    ax.set_ylabel("HS Codes (BMS)")
    ax.bar_label(g, cnt_values, label_type="edge") 
    ax.set_title(f"{company} with {num_battery_txn} battery transactions")
    num_bms_txn = sum(cnt_values)
    
    baseline_summary_smooth = pd.read_csv(f'./summary/{company}_smooth_15-15_summary_baseline.csv').set_index('group')
    (hs_codes, r_values, cnt_values), num_battery_txn = get_hs_r_cnt_values(baseline_summary_smooth)
    ax = axes[1]
    g = ax.barh(hs_codes, r_values, color='orange', height=0.8)
    ax.set_ylabel("HS Codes (Baseline)")
    ax.bar_label(g, cnt_values, label_type="edge") 
    num_baseline_txn = sum(cnt_values)

    ax.set_xlabel("Correlation (31-days Smooth)")
    ax.set_xlim([-1, 1])
    
    plt.show()
    
    fig.savefig(f"./summary/{company}_viz.jpg", bbox_inches="tight")
    return num_battery_txn, num_bms_txn, num_baseline_txn

def plot_lag(company, num_before, num_after):
    import matplotlib
    
    lag_df = pd.read_csv(f"./summary/{company}_smooth_{num_before}-{num_after}_lag.csv")
    baseline_df = pd.read_csv(f"./summary/{company}_smooth_{num_before}-{num_after}_lag_baseline.csv")

    plt.figure(figsize=(10,8))
    
    cmap = matplotlib.cm.Blues(np.linspace(0,1,len(lag_df.columns)))
    for idx, col in enumerate(lag_df.columns):
        if "#" in col:
            print(f"{col}: {lag_df[col][0]}")
            label = col.split('_')[-1]
        elif "corr" in col and "all" not in col: 
            plt.plot(lag_df.index.values, lag_df[col].values, label=label, color=cmap[idx])
            
    print("\n")
    
    cmap = matplotlib.cm.Oranges(np.linspace(0,1,len(baseline_df.columns)))
    for idx, col in enumerate(baseline_df.columns):
        if "#" in col:
            print(f"{col}: {baseline_df[col][0]}")
            label = col.split('_')[-1]
        elif "corr" in col and "all" not in col: 
            plt.plot(baseline_df.index.values, baseline_df[col].values, label=label, color=cmap[idx])

    plt.xlabel("Lag for buy product (days)")
    plt.ylabel("R-value (31 day smooth)")
    plt.ylim(-1, 1)
    plt.legend(loc ="lower right", bbox_to_anchor=(1.2,0))
    plt.title(f"{company}: Battery-BMS vs -Baseline Quantity Correlation over Lags (Days)")
    plt.show()
    plt.savefig(f"./summary/{company}_lag.jpg", bbox_inches="tight")