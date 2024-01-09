# Pulling Graph Data From Hitachi Servers

This module is for transfiguring the `logistic_data` into a form usable for GNNs—that is, graphs with firms as <em>nodes</em> and time-stamped <em>edges</em> as (aggregated) transactions between firms. 

*Note:* If you're using the Tesla dataset instead, run `register_data/preprocess_tesla.py` instead—see the file itself for usage documentation—then proceed directly to the **Crystallization into TGB Data Format** step. The procedure for using the synthetic data is analagous, except run `register_data/preprocess_synthetic.py`.

## Environment & Setup
This should be run on the Hitachi `JupyterHub` server, interlaced with their remote `AWS ec2` data container and RedShift API. Install the following libraries.
```zsh
pip install pycountry-convert
```
To get vital tables for the products, countries, and companies that appear in the Hitachi data, run this:
```zsh
python register_data/extract_tables.py --rs_login <Redshift username> <Redshift password> --dir ./
```
It will save three dictionaries (`hitachi_{company, country, product}_mappers.json`) to this directory.

## Acquire Graph Data
For instance, to get time-stamped transactions starting from `2019-01-01` with `2` days aggregated per time stamp, and `10` time stamps worth of data (e.g. end date is `2019-01-20` with `20` total days), run the following script:
```zsh
python register_data/extract_graph_data.py --rs_login <Redshift username> <Redshift password> --start_date 2019-01-01 \
--length_timestamps 2 --num_timestamps 10 --fname out.csv
```
This will save out the time-stamped edges as a spreadsheet to `out.csv`, where each row represents `{length_timestamps}` days worth of transactions of a particular HS6 product between two firms. An example is shown below.

time_stamp  |  hs6  |  supplier_id           |  buyer_id   | total_amount | ...
------------|-------------|---------------------|----------------- |---- | ---
0      |  850760     |  company A  |  company B | 30 | ...
1   |  850760     |  company A |  company C | 40 | ...
2    |  850450  |  company B |  company A | 50 | ...

The `time_stamp` column indicates the row includes transactions between `{time_stamp} * {length_timestamp}` and `({time_stamp}+1)* {length_timestamps} - 1` days <b>after</b> the `{start_date}`, inclusive. To alchemize the company IDs (e.g. supplier_id, buyer_id) into their company names, add the `--use_titles` flag to the above command. 

# Crystallization into TGB Data Format

After obtaining the transactions CSV file, you'll want to run the `register_hypergraph.py` program—which will create the files necessary to run the TGN / TGN-PL models on the data. This can occur outside the JupyterHub servers. An example run may appear as

```zsh
python register_data/register_hypergraph.py --csv_file daily_transactions_2020.csv --dataset_name tgbl-hypergraph --metric total_amount --dir cache --workers 20 --num_samples 18
```

where `daily_transactions_2020.csv` was the CSV file outputted at the previous step, `--workers 20` pertains to the Python multithreading used for the sampling process, and `--num_samples 18` means 18 negative hyper-edges will be sampled for each positive hyper-edge. 

Here, the program will generate:

* `tgbl-hypergraph_edgelist.csv`: A CSV file where each row is a hyper-edge serialized as <time stamp, buyer node ID, supplier node ID, transaction metric>, with the latter among total amount, weight, etc.
* `tgbl-hypergraph_meta.json`: Contains metadata such as number of firms, products, mapping from firm IDs back to the original names, etc.
* `tgbl-hypergraph_val_ns.pkl`: Contains the negative hyper-edge samples for the validation split, as a dictionary of the form {positive edge: corresponding negative edges}
* `tgbl-hypergraph_test_ns.pkl`: Contains the negative hyper-edge samples for the test splits; identical format to the analagous val file. 

You'll want to transport these files into the relevant TGB dataset subfolder. 

# Run EdgeBank Baseline on Hypergraph Link Prediction 
We also include an HyperEdgeBank baseline—which ranks hyper-edge candidates at testing time based on whether they appeared in the training set or not (as positive samples). To run it for the dataset above:

```
python register_data/edgebank.py --dir ./ --dataset_name tgbl-hypergraph2020
```

This will print out the mean reciprocal rank (MRR) score on the `val` and `test` splits; you can read more about MRR <a href = "https://en.wikipedia.org/wiki/Mean_reciprocal_rank">here</a>.