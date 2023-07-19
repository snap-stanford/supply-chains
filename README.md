# Supply chains & GNN propagation
Files
- **sc_experiments.py**: construct multi-tier supply chain, convert to networkx graph, compute PageRank and PMI.
- **battery_parts_prediction.ipynb**: preliminary experiments to predict battery parts based on supply chain data.
- **ts_experiments.py**: get data, construt time series analysis, build baseline
- **battery_time_series_analysis.ipynb**: preliminary experiments to analyze battery-bms time series correlation, with baseline. 
- **constants_and_utils.py**: constants and utilities for experiments
- **battery_time_series_analysis.ipynb**: preliminary experiments to analyze battery-bms time series correlation, with baseline. 
- **test_RAM.ipynb**: track real-time RAM usage

See the `compare_data` module for code and documentation on aggregating the Hitachi supply chains dataset, and its comparison with the referent United Nations BACI dataset. The latter is widely used in econometrics and international trade research.

## Running TGN Model
First, create a directory named `data` in the root directory, and place the following CSV files inside (the link will be in our Slack messages): `daily_transactions_all_years.csv` and `daily_transactions_{year}.csv` where year ranges from 2019 to 2022, inclusive. Make sure to use the Conda environment as described in `tgb/README.md`, with the requisite packages. Then, run the following:
```
mkdir TGB/tgb/datasets/tgbl_supplychains
bash run_models.sh
```
That script trains the model on all transactions from 2019 - 2022, with a 70/15/15% train/val/test temporal split. To stratify by year, go into `run_model.sh`, and replace any mention of `all_years` or `allyears` with the desired year (e.g. 2019). 