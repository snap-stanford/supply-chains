"""
-for pre-processing the synthetic transactions file; will output a separate CSV file 
-afterward, run the register_hypergraph.py program to crystallize the synthetic
data into files compatible with TGN / TGN-PL model training

<SAMPLE USAGE>
python register_data/preprocess_synthetic.py --transactions_filepath ./observed_transactions.psv 
        --out_filepath ./synthetic_transactions.csv
"""

import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Extracting hypergraph from the Tesla dataset')
    parser.add_argument('--transactions_filepath', help='Path to the .psv file that contains the observed transactions of the synthetic data', default = None)
    parser.add_argument('--out_filepath', help='Path to the .csv file for storing the resulting data', default = None)
    parser.add_argument('--timestamps_start_at_zero', help = 'uniformly shift the time stamps so that the first transactions occurs at time 0', action='store_true')
    args = parser.parse_args()
    return args

def preprocess_synthetic(df, timestamps_start_at_zero = True):
    #modify the column names
    column_metamorphosis = {"origin_company": "supplier_t", 
                  "dest_company": "buyer_t",
                  "amt": "total_amount",
                  "time": "time_stamp",
                  "product": "hs6"}
    df = df.rename(columns = column_metamorphosis)
    
    if (timestamps_start_at_zero == True):
        #shift the time stamps so that the beginning is at 0 
        df = df.sort_values(by = ["time_stamp"], ascending = True)
        min_ts = int(min(df["time_stamp"]))
        df["time_stamp"] = df["time_stamp"].apply(lambda ts: int(ts - min_ts))
        
    #deduplicate if necessary and re-order the columns
    cols = ["time_stamp", "supplier_t","buyer_t","hs6","total_amount"]
    df = df.drop_duplicates(subset = cols)
    df = df[cols]
    return df
                    
if __name__ == "__main__":
              
    args = get_args()
    df_transactions = pd.read_csv(args.transactions_filepath, delimiter = "|")
    print("Loaded in {} raw entries from the synthetic transactions at {}!\n...".format(
        len(df_transactions), args.transactions_filepath))
    
    df_transactions = preprocess_synthetic(df_transactions, 
                          timestamps_start_at_zero = args.timestamps_start_at_zero)
    df_transactions.to_csv(args.out_filepath, index = False)
    
    print("Saved out {} synthetic transactions to {}!".format(len(df_transactions), args.out_filepath))
    print(df_transactions.head(5))
    print(df_transactions.tail(5))
    