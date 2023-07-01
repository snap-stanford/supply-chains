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
It will save three dictionaries (`hitachi_{company, country, product}_mappers.json`) to this directory (`./temporal_graph`).

## Acquiring Graph Data
TODO




