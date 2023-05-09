import os 
import argparse
import warnings
import sys
sys.path.append("/opt/libs")
from apiclass import APIClass,RedshiftClass
from apikeyclass import APIkeyClass
from dotenv import load_dotenv
import pandas as pd

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Parse directory paths for data downloading.')
    parser.add_argument('--baci_dir', nargs='?', help='Where to store BACI data', default = None)
    parser.add_argument('--hitachi_dir', nargs='?', help='Where to store Hitachi data', default = None)
    parser.add_argument('--rs_login', nargs=2, help='Username and password for RedShift', default = None)
    args = parser.parse_args()
    
    #retrieving the BACI data if requested
    if (args.baci_dir != None):
        os.system("wget http://www.cepii.fr/DATA_DOWNLOAD/baci/data/BACI_HS17_V202301.zip -P .")
        os.system(f"unzip -j BACI_HS17_V202301.zip -d {args.baci_dir}")
        os.system(f"rm BACI_HS17_V202301.zip")
        print(f"Unloaded BACI files to {args.baci_dir}")
    
    #retrieving the latest Hitachi data if requested from Hitachi
    if (args.hitachi_dir != None):
        rs = RedshiftClass(args.rs_login[0], args.rs_login[1])
        relevant_files = ["index_hs6", "group_subsidiary_site", "hs_category_description","country_region"]
        for filename in relevant_files:
            #query the redshift database for the entire table
            query = f"select * from {filename}"
            df = rs.query_df(query)
            
            #save to the specified Hitachi data folder 
            df.to_csv(os.path.join(args.hitachi_dir, f"{filename}.csv"))
            print(f"Unloaded {filename}.csv to {args.hitachi_dir}")
            
        
            
            
    