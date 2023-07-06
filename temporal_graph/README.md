# Temporal Graph Data

This module is for transfiguring the `logistic_data` into a form usable for GNNs—that is, graphs with firms as <em>nodes</em> and time-stamped <em>edges</em> as (aggregated) transactions between firms.

## Environment & Setup
This should be run on the Hitachi `JupyterHub` server, interlaced with their remote `AWS ec2` data container and RedShift API. Install the following libraries.
```zsh
pip install pycountry-convert
```
To get vital tables for the products, countries, and companies that appear in the Hitachi data, run this:
```zsh
python extract_tables.py --rs_login <Redshift username> <Redshift password> --dir ./
```
It will save three dictionaries (`hitachi_{company, country, product}_mappers.json`) to this directory.

## Acquire Graph Data
For instance, to get time-stamped transactions starting from `2019-01-01` with `2` days aggregated per time stamp, and `10` time stamps worth of data (e.g. end date is `2019-01-20` with `20` total days), run the following script:
```zsh
python extract_graph_data.py --rs_login <Redshift username> <Redshift password> --start_date 2019-01-01 \
--length_timestamps 2 --num_timestamps 10 --fname out.csv
```
This will save out the time-stamped edges as a spreadsheet to `out.csv`, where each row represents `{length_timestamps}` days worth of transactions of a particular HS6 product between two firms. An example is shown below.

time_stamp  |  hs6  |  supplier_id           |  buyer_id   | total_amount | ...
------------|-------------|---------------------|----------------- |---- | ---
2.0       |  850760     |  company A  |  company B | 30 | ...
4.0    |  850760     |  company A |  company C | 40 | ...
6.0     |  850450  |  company B |  company A | 50 | ...

The `time_stamp` column indicates the row includes transactions between `{time_stamp} - {length_timestamps}` and `{time_stamp}-1` days <b>after</b> the `{start_date}`, inclusive. To alchemize the company IDs (e.g. supplier_id, buyer_id) into their company names, add the `--use_titles` flag to the above command. 

## Transform into PyG Temporal Graph
From the .csv file saved using `extract_graph_data.py` (<b>make sure</b> to have included the `--use_titles` flag), you'll want to use our `dataloading` module to transform it into a PyG graph. 
```python
from dataloading import SupplyChainDataset

data = SupplyChainDataset("out.csv", start_date = "2022-01-01", length_timestamps = 2, metric = "total_amount")
priorGraph, nextGraph = data.loadData(current_date = "2022-01-10", prior_days = 6, next_days = 4)

for timestep, snapshot in enumerate(priorGraph): #iterate through temporal graph
    print(type(snapshot)) #<class 'torch_geometric.data.hetero_data.HeteroData'>
```

This will load data from the last 6 days of `2022-01-10` (Jan 5th to Jan 10th) and the next 4 days (Jan 11th to 14th) into `priorGraph` and `nextGraph`, respectively. These are `DynamicHeteroGraphTemporalSignal` iterator objects from PyG Temporal, with firms as nodes and product-heterogeneous, dynamic edges that are time-stamped. Each time iteration corresponds to a PyG `HeteroData` graph. See the source code at `dataloading.py` for details.