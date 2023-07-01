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

## Acquiring Graph Data
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

The `time_stamp` column indicates the row includes transactions between `{time_stamp} - {length_timestamps}` and `{time_stamp}-1` days <b>after</b> the `{start_date}`, inclusive. As a <b>note</b>, for now, the script only gathers data for battery-related products due to scale. To alchemize the company IDs (e.g. supplier_id, buyer_id) into their company names, add the `--use_titles` flag to the above command. 