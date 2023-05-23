import http.cookiejar as cookielib
import pandas as pd
import requests
import warnings
import logging
import sqlite3
import boto3
import glob
import json
import time
import datetime
import math
import tqdm
import sys
import os
import glob
import re
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats.stats import pearsonr

sys.path.append("/opt/libs")
from apiclass import APIClass,RedshiftClass
from apikeyclass import APIkeyClass
from dotenv import load_dotenv

from sc_experiments import *
from constants_and_utils import *

warnings.filterwarnings('ignore')
rs = RedshiftClass('zhiyin','Zhiyin123')

THRESHOLD = 100

dict_regex = {
    "samsung": ["%samsung%", "sevt", "sehc", "sdiv"],
    "wistron": ["%wistron%"],
    "hp": ["hp %"],
    "elentec": ["%elentec%"],
    "itm": ["%itm %"],
    "verdant": ["verdant %"],
    "luxshare": ["%luxshare%", "lxvt", "lxvn"],
    "apple": ["%apple %"],
    "wingtech": ["wingtech %"],
    "lg": ['%lg %', 'lgdvh', 'lgevh', 'lgitvh'], 
    "techtronic": ["%techtronic%", "%tti %"],
    "arnold": ["arnold %"],
    "hefei_gotion": ["hefei gotion %"],
    "shenzhen_chenshi": ["%shenzhen chenshi %"],
    "tcl": ["tcl %"],
    "hansol": ["%hansol %"],
    "buckeye": ["%buckeye %"],
    "compal": ["%compal %"],
    "truper": ["%truper %"]
}
# dict_regex = {
#     "samsung": ["%samsung%", "sevt", "sehc", "sdiv"],
#     "dell": ["dell%"],
#     "wistron": ["%wistron%"],
#     "byd": ["byd %"],
#     "hp": ["hp %"],
#     "elentec": ["%elentec%"],
#     "itm": ["%itm %"],
#     "yiwu": ["yiwu %"],
#     "verdant": ["verdant %"],
#     "luxshare": ["%luxshare%", "lxvt", "lxvn"],
#     "railhead": ["railhead %"],
#     "navitasys": ["%navitasys %"],
#     "motorola": ["motorola %"],
#     "apple": ["%apple %"],
#     "wingtech": ["wingtech %"],
#     "techtronic": ["%techtronic%", "%tti %"],
#     "lg": ['%lg %', 'lgdvh', 'lgevh', 'lgitvh'], 
#     "shenzhen_coman": ["shenzhen coman %"],
#     "arnold": ["arnold %"],
#     "greenway": ["%greenway %"],
#     "hui_shun": ["%hui shun %"],
#     "hefei_gotion": ["hefei gotion %"],
#     "shenzhen_chenshi": ["%shenzhen chenshi %"],
#     "tcl": ["tcl %"],
#     "lenovo": ["%lenovo %"],
#     "hansol": ["%hansol %"],
#     "star_in": ["%star-in %"],
#     "zamax": ["zamax %"],
#     "sunwoda": ["sunwoda %"],
#     "cheape": ["cheape %"],
#     "sr_tech": ["%sr tech %"],
#     "khvatec": ["%khvatec %"],
#     "cistech": ["cistech %"],   
#     "bosch": ["%bosch %"],
#     "leclanche": ["%leclanche %"],
#     "sky_royal": ["%sky royal %"],
#     "tecno": ["%tecno %"],
#     "lishen": ["%lishen %"],
#     "huidafa": ["%huidafa %"],
#     "lanway": ["%lanway %"],
#     "asus": ["%asus %", "%asustek %"],
#     "xiaomi": ["%xiaomi %"],
#     "ge_healthcare": ["%ge healthcare %"],
#     "buckeye": ["%buckeye %"],
#     "compal": ["%compal %"],
#     "liuzhou_gotion": ["%liuzhou gotion %"],
#     "k_tech": ["%k - tech %"],
#     "twister": ["%twister %"],
#     "asian_star": ["%asian star %"], 
#     "hilti": ["%hilti %"], 
#     "ag_tech": ["%ag tech%"],
#     "ctechi": ["%ctechi %"], 
#     "neway": ["%neway %"], 
#     "truper": ["%truper %"], 
#     "linkworld": ["%linkworld %"], 
#     "audi": ["%audi %"], 
#     "zebra": ["%zebra %"], 
#     "transcend": ["%transcend %"], 
#     "ty": ["%tyev %"], 
#     "zhongshan_tianze": ["%zhongshan tianze %"], 
#     "loram": ["%loram %"], 
#     "power_tools": ["%power tools %"], 
#     "dragerwerk": ["%dragerwerk %"],
#     "covidien": ["%covidien %"], 
#     "ecovice": ["%ecovice %"], 
#     "ingram": ["%ingram %"], 
#     "commscope": ["%commscope %"], 
#     "huaqin": ["%huaqin %"], 
#     "longcheer": ["%longcheer %"], 
#     "hq_telecom_singapore": ["%hq telecom singapore %"], 
#     "black_decker": ["black%&%decker%"], 
#     "haihang": ["%haihang %"], 
#     "fushan": ["%fushan %"], 
#     "number_king": ["%number king %"], 
#     "lishengyuan": ["%lishengyuan %"], 
#     "j_run": ["%j-run %"], 
#     "fullymax": ["%fullymax %"], 
#     "mr_global": ["%mr global %"], 
#     "lotus": ["%lotus %"], 
#     "ecoflow": ["%ecoflow %"], 
#     "grepow": ["%grepow %"], 
#     "ivory": ["%ivory %"], 
#     "tesan": ["tesan %"], 
#     "mission": ["mission %"], 
#     "philips": ["philips %"], 
#     "export_distribution_center": ["export distribution center"], 
#     "triathlon": ["%triathlon %"], 
#     "toyota": ["%toyota %"], 
#     "evergreen": ["%evergreen %"], 
#     "hmd": ["%hmd %"], 
#     "direction": ["%direction %"], 
#     "yokogawa": ["%yokogawa %"], 
#     "tisky": ["%tisky %"], 
#     "jiangxi_beston": ["%jiangxi beston %"], 
#     "shenzhen_just_link": ["%shenzhen just link %"], 
#     "bmw": ["bmw %"], 
#     "panasonic": ["panasonic %"], 
#     "kyboton": ["kyboton %"], 
#     "let_hk": ["let hk %"], 
#     "sekatai": ["sekatai %"], 
#     "henglikai": ["%henglikai %"], 
#     "tomstar": ["tomstar %"], 
#     "futuretech": ["futuretech %"], 
#     "global_connections": ["global connections %"], 
#     "sony": ["sony %"], 
#     "cwv": ["cwv"], 
#     "shone": ["shone %"], 
#     "regional empire trading": ["regional empire trading"], 
#     "imarket": ["imarket%"], 
#     "v_scope": ["v-scope %"], 
#     "jolly": ["jolly %"], 
#     "asa_story": ["asa story %"], 
#     "yue_hao_hong_hong": ["yue hao hong hong %"], 
#     "chuang_xin": ["chuang xin %"], 
#     "jiangxi_beston": ["jiangxi beston %"], 
#     "guangzhou_zhangdi": ["guangzhou zhangdi %"], 
#     "smartilike": ["smartilike %"], 
#     "maitrox": ["%maitrox %"], 
#     "tws": ["tws %"]
# }

bom = parse_battery_bom()
bom =  {k.lower(): v for k, v in bom.items()}

def get_transaction_df(PROD='battery', tx_type='supplier', companies=[]):
    '''
    Args:
        PROD (str or Tuple(str)): a string type key of bom or a tuple of strings that are general hs_codes such as 'bms' or ('850670', '482110',)
        tx_type (str): 'supplier' or 'buyer'
        companies (list(str)): company keywords for SQL query

    Returns:
        pandas.dataframe: transaction dataframe
        str: identifier name of the csv, such as 'samsung', 'lg', 'all', etc. If both 'None', error state
    '''
    # save transactions
    if len(companies) >= 1:
        csv_name = " ".join(re.findall("[a-zA-Z]+", companies[0]))
    else:
        csv_name = "all"
    
    # get transactions where company is supplying/buying PROD
    hs_codes = bom[PROD] if PROD in bom else PROD
    name = PROD if PROD in bom else "baseline"
    tx_col = 'supplier_t' if tx_type=='supplier' else 'buyer_t' if tx_type=='buyer' else None
    for hs in hs_codes:
        if len(companies) >= 1:
            tx_string = f"{tx_col} like '{companies[0]}'" + "".join([f" or {tx_col} like '{company}'" for company in companies[1:]])
            query = f"select date, supplier_id, buyer_id, hs_code, quantity, weight, price, amount, COUNT(*) as count, COUNT(DISTINCT id) as num_ids from logistic_data where (hs_code like '{hs}%') and ({tx_string}) GROUP BY date, supplier_id, buyer_id, hs_code, quantity, weight, price, amount;"
        else:
            query = f"select date, supplier_id, buyer_id, hs_code, quantity, weight, price, amount, COUNT(*) as count, COUNT(DISTINCT id) as num_ids from logistic_data where (hs_code like '{hs}%') GROUP BY date, supplier_id, buyer_id, hs_code, quantity, weight, price, amount;"
        df = rs.query_df(query)
        if df is not None:
            df = df.drop_duplicates()
            print(hs, '->', len(df), len(df.drop_duplicates()))
            df['hs_code'] = df['hs_code'].str[:6]
        else:
            # no results found for this query
            print(hs, '-> None')
            continue

    # if len(all_dfs) == 0:
    #     print("No df to concatnate")
    #     return None, None

    # all_dfs = pd.concat(all_dfs).drop_duplicates()
    # print("all df shape is", all_dfs.shape)
    # assert(len(all_dfs)==len(all_dfs.drop_duplicates()))  # should be the same length

        outdir = f'./data/{csv_name}/'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        df.to_csv(f'./data/{csv_name}/{name}_{tx_type}_{csv_name}_{hs}.csv', index=False) 
        del df
    
    return csv_name

def preprocess_transaction_data(SUPPLY_PROD='battery', BUY_PROD='bms', companies=[]):
    '''
    Args:
        SUPPLY_PROD (str or Tuple(str)): a string type key of bom or a tuple of strings that are general hs_codes
                                    such as 'bms' or ('850670', '482110',) for supplying
        BUY_PROD (str or Tuple(str)): same as SUPPLY_PROD, but for buying
        companies (list(str)): company keywords for SQL query

    Returns:
        pandas.dataframe: supply dataframe
        pandas.dataframe: buy dataframe
        str: identifier name of the csv, such as 'samsung', 'lg', 'all', etc.
    '''
    print("Getting supply df...")
    csv_name = get_transaction_df(PROD=SUPPLY_PROD, tx_type='supplier', companies=companies)
    print("Getting buy df...")
    csv_name = get_transaction_df(PROD=BUY_PROD, tx_type='buyer', companies=companies)
    
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

def get_transaction_time_df(SUPPLY_PROD='battery', BUY_PROD='bms', csv_name='samsung'):
    '''
    Add daily, weekly, and monthly datetime columns.
    '''
    supply_name = SUPPLY_PROD if SUPPLY_PROD in bom else "baseline" 
    buy_name = BUY_PROD if BUY_PROD in bom else "baseline"
    print(f"Supply name is {supply_name}; buy name is {buy_name}")
    
    print(supply_name, csv_name)
    supply_df = load_df(name=supply_name, csv_name=csv_name)
    buy_df = load_df(name=buy_name, csv_name=csv_name)
    print(len(supply_df), len(buy_df))

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
    if time.lower()=='daily':
        return 'datetime'
    elif time.lower()=='weekly':
        return 'week_datetime'
    elif time.lower()=='monthly':
        return 'month_datetime'
    raise Exception("Time window is not supported")

def plot_time_versus_sale_purchase(supply_df, buy_df, SUPPLY_PROD='battery', BUY_PROD='bms', time='monthly', csv_name='samsung'):
    '''Plot sale info (quantity and amount) over time
    Args:
        time (str): supports 'daily', 'weekly', or 'monthly'
    '''
    time_col = get_time_col(time)
    supply_name = SUPPLY_PROD if SUPPLY_PROD in bom else "baseline" 
    buy_name = BUY_PROD if BUY_PROD in bom else "baseline" 
    
    # plot time-ly sales vs daily purchases
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
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    fig.savefig(f"./fig/{csv_name}/time_versus_sale_purchase_quantity_{supply_name}_{buy_name}_{time}_{csv_name}.jpg", bbox_inches="tight")

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
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    fig.savefig(f"./fig/{csv_name}/time_versus_sale_purchase_amount_{supply_name}_{buy_name}_{time}_{csv_name}.jpg", bbox_inches="tight")

def compare_sale_purchase_quantity_per_hscode(supply_df, buy_df, SUPPLY_PROD='battery', BUY_PROD='bms', time='monthly', csv_name='samsung'):
    '''Calculate correlation for sale info (quantity) over time per supply hs_code, buy hs_code
    Args:
        time (str): supports 'daily', 'weekly', or 'monthly'
    '''
    supply_hs_codes = bom[SUPPLY_PROD] if SUPPLY_PROD in bom else SUPPLY_PROD
    buy_hs_codes = bom[BUY_PROD] if BUY_PROD in bom else BUY_PROD
    supply_name = SUPPLY_PROD if SUPPLY_PROD in bom else "baseline" 
    buy_name = BUY_PROD if BUY_PROD in bom else "baseline" 
    time_col = get_time_col(time)
    
    supply_df['hs_code_str'] = supply_df.hs_code.astype(str)  # convert HS code to str
    buy_df['hs_code_str'] = buy_df.hs_code.astype(str)  # convert HS code to str

    summary_df = {}
    summary_df['group'] = csv_name
    
    for supply_hs in supply_hs_codes:
        sub_supply_df = supply_df[supply_df.hs_code_str.str.contains(supply_hs)]
        summary_df[f"# of supply txn {supply_name}_{supply_hs}"] = len(sub_supply_df)
        if len(sub_supply_df) > THRESHOLD:
            supply_summary = sub_supply_df.groupby(time_col).quantity.sum()
            fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
            ax = axes[0]           
            ax.set_title(f'{time} sales of {supply_hs}', fontsize=12)
            ax.plot(supply_summary.index.values, supply_summary.values)
            ax.set_ylabel('Total quantity', fontsize=12)
            ax = axes[1]
            ax.set_title(f'{time} purchases of {buy_name}', fontsize=12)
            
            summary_df[f"# of buy txn {buy_name}_all"] = len(buy_df)
            if len(buy_df) > THRESHOLD:
                buy_summary = buy_df.groupby(time_col).quantity.sum()
                merged = pd.merge(supply_summary.rename('x'), buy_summary.rename('y'), 
                                          left_index=True, right_index=True, how='inner')
                r, p = pearsonr(merged.x, merged.y)
                summary_df[f"corr {supply_name}_{supply_hs}, {buy_name}_all monthly"] = r
            else:
                summary_df[f"corr {supply_name}_{supply_hs}, {buy_name}_all monthly"] = None
                
            
            for buy_hs in buy_hs_codes:
                sub_buy_df = buy_df[buy_df.hs_code_str.str.contains(buy_hs)]
                summary_df[f"# of supply txn {buy_name}_{buy_hs}"] = len(sub_buy_df)
                if len(sub_buy_df) > THRESHOLD:
                    buy_summary = sub_buy_df.groupby(time_col).quantity.sum()
                    # merge to find common months
                    merged = pd.merge(supply_summary.rename('x'), buy_summary.rename('y'), 
                                      left_index=True, right_index=True, how='inner')
                    r, p = pearsonr(merged.x, merged.y)
                    print(buy_hs, len(sub_buy_df), 'r=%.3f (n=%d, p=%.3f)' % (r, len(merged), p)) # number of transaction, n is number of dates
                    # normalize by mean to make comparison easier
                    ax.plot(buy_summary.index.values, buy_summary.values / np.mean(buy_summary.values), label=buy_hs)
                    summary_df[f"corr {supply_name}_{supply_hs}, {buy_name}_{buy_hs} monthly"] = r
                else:
                    print(buy_hs, len(sub_buy_df))
                    summary_df[f"corr {supply_name}_{supply_hs}, {buy_name}_{buy_hs} monthly"] = None
                
            ax.legend(bbox_to_anchor=(1,1))
            ax.set_ylabel('Total quantity (normalized)', fontsize=12)
            plt.show()
        else:
            for buy_hs in buy_hs_codes:
                summary_df[f"# of supply txn {buy_name}_{buy_hs}"] = None
            print(supply_hs, len(sub_supply_df))
            
        fig.tight_layout()
        outdir = f'./fig/{csv_name}/'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        fig.savefig(f"./fig/{csv_name}/compare_sale_purchase_quantity_{supply_name}_{supply_hs}_{buy_name}_{time}_{csv_name}.jpg", bbox_inches="tight")

    return summary_df

def compare_sale_purchase_quantity_per_hscode_smooth(supply_df, buy_df, SUPPLY_PROD='battery', BUY_PROD='bms', time='monthly', num_before=3, num_after=3, csv_name='samsung'):
    '''Calculate correlation for sale info (quantity) over time per supply hs_code, buy hs_code
    Args:
        time (str): supports 'daily', 'weekly', or 'monthly'
    '''
    supply_hs_codes = bom[SUPPLY_PROD] if SUPPLY_PROD in bom else SUPPLY_PROD
    buy_hs_codes = bom[BUY_PROD] if BUY_PROD in bom else BUY_PROD
    supply_name = SUPPLY_PROD if SUPPLY_PROD in bom else "baseline" 
    buy_name = BUY_PROD if BUY_PROD in bom else "baseline" 
    time_col = get_time_col(time)
    
    supply_df['hs_code_str'] = supply_df.hs_code.astype(str)  # convert HS code to str
    buy_df['hs_code_str'] = buy_df.hs_code.astype(str)  # convert HS code to str

    summary_df = {}
    summary_df['group'] = csv_name
    
    for supply_hs in supply_hs_codes:
        sub_supply_df = supply_df[supply_df.hs_code_str.str.contains(supply_hs)]
        summary_df[f"# of supply txn {supply_name}_{supply_hs}"] = len(sub_supply_df)
        if len(sub_supply_df) > THRESHOLD:
            supply_summary = sub_supply_df.groupby(time_col).quantity.sum()
            # Begin smoothing
            smooth_values = apply_smoothing(supply_summary.values, num_before=num_before, num_after=num_after)
            assert(len(supply_summary)==len(smooth_values))
            supply_summary.replace(supply_summary.values, smooth_values, inplace=True)
            supply_summary.fillna(0, inplace=True)
            # End smoothing
            fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
            ax = axes[0]           
            ax.set_title(f'{time} sales of {supply_hs}', fontsize=12)
            ax.plot(supply_summary.index.values, supply_summary.values)
            ax.set_ylabel('Total quantity', fontsize=12)
            ax = axes[1]
            ax.set_title(f'{time} purchases of {buy_name}', fontsize=12)
            
            summary_df[f"# of buy txn {buy_name}_all"] = len(buy_df)
            if len(buy_df) > THRESHOLD:
                buy_summary = buy_df.groupby(time_col).quantity.sum()
                # Begin smoothing
                smooth_values = apply_smoothing(buy_summary.values, num_before=num_before, num_after=num_after)
                assert(len(buy_summary)==len(smooth_values))
                buy_summary.replace(buy_summary.values, smooth_values, inplace=True)
                buy_summary.fillna(0, inplace=True)
                # End smoothing
                merged = pd.merge(supply_summary.rename('x'), buy_summary.rename('y'), 
                                          left_index=True, right_index=True, how='inner')
                r, p = pearsonr(merged.x, merged.y)
                summary_df[f"corr {supply_name}_{supply_hs}, {buy_name}_all smooth_{num_before}_{num_after}"] = r
            else:
                summary_df[f"corr {supply_name}_{supply_hs}, {buy_name}_all smooth_{num_before}_{num_after}"] = None
                
            
            for buy_hs in buy_hs_codes:
                sub_buy_df = buy_df[buy_df.hs_code_str.str.contains(buy_hs)]
                summary_df[f"# of supply txn {buy_name}_{buy_hs}"] = len(sub_buy_df)
                if len(sub_buy_df) > THRESHOLD:
                    buy_summary = sub_buy_df.groupby(time_col).quantity.sum()
                    # Begin smoothing
                    smooth_values = apply_smoothing(buy_summary.values, num_before=num_before, num_after=num_after)
                    assert(len(buy_summary)==len(smooth_values))
                    buy_summary.replace(buy_summary.values, smooth_values, inplace=True)
                    buy_summary.fillna(0, inplace=True)
                    # End smoothing
                    # merge to find common months
                    merged = pd.merge(supply_summary.rename('x'), buy_summary.rename('y'), 
                                      left_index=True, right_index=True, how='inner')
                    r, p = pearsonr(merged.x, merged.y)
                    print(buy_hs, len(sub_buy_df), 'r=%.3f (n=%d, p=%.3f)' % (r, len(merged), p)) # number of transaction, n is number of dates
                    # normalize by mean to make comparison easier
                    ax.plot(buy_summary.index.values, buy_summary.values / np.mean(buy_summary.values), label=buy_hs)
                    summary_df[f"corr {supply_name}_{supply_hs}, {buy_name}_{buy_hs} smooth_{num_before}_{num_after}"] = r
                else:
                    print(buy_hs, len(sub_buy_df))
                    summary_df[f"corr {supply_name}_{supply_hs}, {buy_name}_{buy_hs} smooth_{num_before}_{num_after}"] = None
                
            ax.legend(bbox_to_anchor=(1,1))
            ax.set_ylabel('Total quantity (normalized)', fontsize=12)
            plt.show()
        else:
            for buy_hs in buy_hs_codes:
                summary_df[f"# of supply txn {buy_name}_{buy_hs}"] = None
            print(supply_hs, len(sub_supply_df))
            
        fig.tight_layout()
        outdir = f'./fig/{csv_name}/'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        fig.savefig(f"./fig/{csv_name}/compare_sale_purchase_quantity_{supply_name}_{supply_hs}_{buy_name}_smooth_{num_before}_{num_after}_{csv_name}.jpg", bbox_inches="tight")

    return summary_df

def mega_sale_purchase_pipeline(SUPPLY_PROD='battery', BUY_PROD='bms', csv_name='samsung'):
    print("Mega Analysis of Sale - Purchase Pipeline on Daily, Weekly, and Monthly Scale\n")
    print("Assume Transaction Data Has Been Queried and Preprocessed...")
    print("Constructing Transaction Time Dataframe...Loading from csv...")
    supply_df, buy_df = get_transaction_time_df(SUPPLY_PROD=SUPPLY_PROD, BUY_PROD=BUY_PROD, csv_name=csv_name)
    print("Plotting sale, purchase over time...")
    plot_time_versus_sale_purchase(supply_df, buy_df, SUPPLY_PROD=SUPPLY_PROD, BUY_PROD=BUY_PROD, time='daily', csv_name=csv_name)
    plot_time_versus_sale_purchase(supply_df, buy_df, SUPPLY_PROD=SUPPLY_PROD, BUY_PROD=BUY_PROD, time='weekly', csv_name=csv_name)
    plot_time_versus_sale_purchase(supply_df, buy_df, SUPPLY_PROD=SUPPLY_PROD, BUY_PROD=BUY_PROD, time='monthly', csv_name=csv_name)
    print("Compare sale, purchase quantity per hscode...")
    compare_sale_purchase_quantity_per_hscode(supply_df, buy_df, SUPPLY_PROD=SUPPLY_PROD, BUY_PROD=BUY_PROD, time="daily", csv_name=csv_name)
    compare_sale_purchase_quantity_per_hscode(supply_df, buy_df, SUPPLY_PROD=SUPPLY_PROD, BUY_PROD=BUY_PROD, time="weekly", csv_name=csv_name)
    compare_sale_purchase_quantity_per_hscode(supply_df, buy_df, SUPPLY_PROD=SUPPLY_PROD, BUY_PROD=BUY_PROD, time="monthly", csv_name=csv_name)
    return supply_df, buy_df

# Baseline Methods
def get_baseline_hs6(tx_type=None, hs_codes=None, companies=[], TOPCNT=10):
    '''If provided hs_codes, we return that. Otherwise, select top TOPCNT==10 non-battery-hs6-codes 
    from all hscodes as baseline.
    Args:
        hs_codes (None or list(str)): list of potential hs codes of interest
        companies (list(str)): company keywords for SQL query

    Returns:
        list(str): baseline hs6 codes
    '''
    battery_hs6_set = set(list(itertools.chain.from_iterable(bom.values()))) # All hs6 codes battery
    BASELINE_HS6 = []
    
    # SQL query to all hs_code, sum(quantity) for LG
    tx_col = 'supplier_t' if tx_type=='supplier' else 'buyer_t' if tx_type=='buyer' else None
    if len(companies) >= 1:
        tx_string = f"{tx_col} like '{companies[0]}'" + "".join([f" or {tx_col} like '{company}'" for company in companies[1:]])
        query = f"select hs_code, sum(quantity) from logistic_data where ({tx_string}) group by hs_code;"
    else:
        query = f"select hs_code, sum(quantity) from logistic_data group by hs_code;"
    print(query)
    df = rs.query_df(query)
    all_hs_codes = df.sort_values('sum', ascending=False).hs_code.values 
        
    if hs_codes==None:    
        for idx in range(min(len(all_hs_codes), TOPCNT)):
            hs_code = all_hs_codes[idx]
            if hs_code[:6] not in battery_hs6_set: # When checking, take first six digits. When saving, use original
                BASELINE_HS6.append(hs_code) 
    else:
        for hs_code in all_hs_codes:
            if hs_code[:6] in set(hs_codes):
                BASELINE_HS6.append(hs_code)
            
    return BASELINE_HS6

def baseline(SUPPLY_PROD = 'battery', 
             BUY_PROD = 'bms', 
             companies = ['%samsung%', 'sehc', 'sevt'],
             tx_type = 'buyer', 
             hs_codes = ['482110', '480591']):
    BASELINE_HS6 = get_baseline_hs6(tx_type=tx_type, 
                                hs_codes=hs_codes, 
                                companies=companies)
    
    BASELINE_HS6 = [hs6[:6] for hs6 in BASELINE_HS6] # Added to focus on only hs6 level 
    
    if tx_type=='buyer':
        BUY_PROD = tuple(BASELINE_HS6)
    elif tx_type=='supplier':
        SUPPLY_PROD = tuple(BASELINE_HS6)
    else:
        raise Exception("Invalid tx_type value")
        
    supply_df, buy_df, csv_name = preprocess_transaction_data(SUPPLY_PROD=SUPPLY_PROD, 
                                                              BUY_PROD=BUY_PROD, 
                                                              companies=companies)
    print(f"CSV name is {csv_name}...")
    supply_df, buy_df = mega_sale_purchase_pipeline(SUPPLY_PROD=SUPPLY_PROD,
                                                    BUY_PROD=BUY_PROD, 
                                                    csv_name=csv_name)
    return