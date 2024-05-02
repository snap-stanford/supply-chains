"""
for pre-processing the sem.csv file, which contains transactions related to 
the SEM supply-chain; will output a separate, timestamp-aggregated CSV file 
afterward, run the register_hypergraph.py program to crystallize the SEM
data into files compatible with TGN / TGN-PL model training

<SAMPLE USAGE>
python register_data/preprocess_sem.py --sem_filepath ./sem.csv --out_filepath    
     ./sem_transactions.csv --use_titles
"""
import pandas as pd
import argparse
import datetime
import numpy as np

#global variables 
PRIMARY_KEY = ["date","supplier_id", "buyer_id","hs_code","quantity_sum","weight_sum","amount_sum", "bill_count"]
AGGREGATION_KEY = ["time_stamp","hs6","supplier_id","buyer_id"]

def get_args():
    parser = argparse.ArgumentParser(description='Extracting hypergraph from the SEM dataset')
    parser.add_argument('--start_date', help='Starting day of form YY-MM-DD from which to collect data', default = "2019-01-01")
    parser.add_argument('--end_date', help='Ending day of form YY-MM-DD from which to collect data', default = "2024-02-19")
    parser.add_argument('--length_timestamps', help='Number of days to aggregate per time stamp', default = 1, type = int)
    parser.add_argument('--sem_filepath', help='Path to the .csv file that contains the SEM supply-chain transactions', default = None)
    parser.add_argument('--out_filepath', help='Path to the .csv file for storing the resulting data', default = None)
    parser.add_argument('--use_titles', help = 'if provided, data uses company titles instead of IDs', action='store_true')
    args = parser.parse_args()
    return args

def add_to_dictionary(company_dict, company_pairs):
    #helper function that is used for get_company_idname_mappers() to map firm IDs to names
    for firm_t, firm_id in zip(*company_pairs):
        if (firm_id in company_dict and firm_t in company_dict[firm_id]):
            company_dict[firm_id][firm_t] += 1
        elif (firm_id in company_dict):
            company_dict[firm_id][firm_t] = 1
        else:
            company_dict[firm_id] = {firm_t: 1}
    return company_dict 

def get_ts(start_date, end_date, length_timestamps = 1, date_format = '%Y-%m-%d'):
    #helper function used for calculating discrete timestamps 
    delta = datetime.datetime.strptime(end_date, date_format) - datetime.datetime.strptime(
        start_date, date_format)
    delta_days = delta.days
    return int(np.ceil((delta_days + 1) / length_timestamps)) - 1

def get_company_idname_map(df):
    #obtain a dictionary mapping from the firm hashed IDs to the firm names 
    supplier_info = [list(df[column_t]) for column_t in ["supplier_t","supplier_id"]]
    buyer_info = [list(df[column_t]) for column_t in ["buyer_t","buyer_id"]]

    company_id2name = {}
    company_id2name = add_to_dictionary(company_id2name, supplier_info)
    company_id2name = add_to_dictionary(company_id2name, buyer_info)
    
    #pre-process to get the firm name that appears the most frequently for each form ID
    company_id2name_final = {}
    for i, firm_id in enumerate(company_id2name):
        title_counts = company_id2name[firm_id]
        firm_t = max(title_counts, key = title_counts.get)
        company_id2name_final[firm_id] = firm_t

    return company_id2name_final
  
def preprocess_sem(df, start_date = "2019-01-01", end_date = "2024-02-19", use_titles = True, id2company = None,
                   length_timestamps = 1, date_format = '%Y-%m-%d'):
    #select dates of interest
    start_date_dt = datetime.datetime.strptime(start_date, date_format)
    end_date_dt = datetime.datetime.strptime(end_date, date_format)
    df_date_dt = pd.to_datetime(df.date)
    df = df[(df_date_dt >= start_date_dt) & (df_date_dt <= end_date_dt)]
    print("df has shape {} after selecting rows within {} and {} inclusive".format(df.shape, start_date, end_date))

    #deduplicate the SEM dataset according to the primary key
    df_sem = df[PRIMARY_KEY]
    df_sem = df_sem.drop_duplicates(subset = PRIMARY_KEY)
    
    #calculate the time stamps, as well as siphon out incorrectly categorized products (HS6)
    data_to_ts = {date: get_ts(start_date, date, length_timestamps) for date in set(df_sem["date"])}
    # df_sem["bill"] = 1
    df_sem["time_stamp"] = df_sem["date"].apply(lambda date: data_to_ts[date])
    df_sem["hs6"] = df_sem["hs_code"].apply(lambda hs_code: str(hs_code)[:6])
    df_sem = df_sem[df_sem["hs6"].str.match('^(?!00)[0-9]{6}')] 
    
    #aggregate the transactions by time stamps, and sort by ascending time stamp 
    df_sem_agg = df_sem.groupby(by = AGGREGATION_KEY).sum(numeric_only = True).reset_index()
    df_sem_agg = df_sem_agg.rename(columns = {"quantity_sum": "total_quantity",
                                                  "weight_sum": "total_weight",
                                                  "amount_sum": "total_amount"})
    # df_sem_agg = df_sem_agg.drop(columns = ["price"])
    df_sem_agg = df_sem_agg.sort_values(by = ["time_stamp"], ascending = True)
    
    #replace the company hashed IDs to the company names if requested 
    if (use_titles == True):
        company_ids = list(id2company.keys())
        df_companies = pd.DataFrame.from_dict({"company_id": company_ids,
                                       "company_t": [id2company[id] for id in company_ids]})

        #replace the company supplier IDs
        df_sem_agg = pd.merge(df_sem_agg, df_companies, 
                                left_on = "supplier_id", right_on = "company_id", how = "left")
        df_sem_agg = df_sem_agg.rename(columns = {"company_t": "supplier_t"}).drop(
                                columns = {"company_id","supplier_id"})
        #replace the company buyer IDs
        df_sem_agg = pd.merge(df_sem_agg, df_companies, left_on = "buyer_id", 
                                right_on = "company_id", how = "left")
        df_sem_agg = df_sem_agg.rename(columns = {"company_t": "buyer_t"}).drop(
                                columns = {"company_id","buyer_id"})
        
        #reorder the columns in the dataframe
        df_sem_agg = df_sem_agg[["time_stamp","supplier_t","buyer_t","hs6","bill_count",
                                    "total_quantity","total_amount","total_weight"]]
    else:
        df_sem_agg = df_sem_agg[["time_stamp","supplier_id","buyer_id","hs6","bill_count",
                                    "total_quantity","total_amount","total_weight"]]
    
    return df_sem_agg


if __name__ == "__main__":
 
    args = get_args()
    df = pd.read_csv(args.sem_filepath)
    print("Loaded in {} raw entries from the SEM dataset at {}!\n...".format(len(df), args.sem_filepath))

    # Extra preprocess to deal with 7 negative amount_sum rows: they exist in data provider source
    df = df[df.amount_sum >= 0]
    print("After removing 7 negative amount_sum rows, there are {} raw entries\n".format(len(df)))
    
    map_id2company = get_company_idname_map(df)
    df_sem = preprocess_sem(df, args.start_date, args.end_date, args.use_titles, map_id2company,
                              length_timestamps = args.length_timestamps)
    df_sem.to_csv(args.out_filepath, index = False)
    
    print("Saved out {} transactions to {}!".format(len(df_sem), args.out_filepath))
    print(df_sem.head(5))
    print(df_sem.tail(5))
