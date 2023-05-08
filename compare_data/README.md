# Comparing Supply Chain Data with Global Records 
## Dependencies
A base Conda environment should be sufficient for running all the code here, with the principal Python libraries being `matplotlib`, `numpy`, `scipy`, and `pandas`. 

## Data 
The Hitachi supply chain data resides on the JupyterLab server. The BACI data is available at [dataset link](`http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37`), with the most recent version HS17. You can either download and unzip all the BACI `.csv` files to the same directory, or run
```zsh
mkdir data
mkdir data/BACI
mkdir data/Hitachi

python retrieve_data.py --baci_dir data/BACI --hitachi_dir data/Hitachi --rs_login <redshift username> <redshift password>
```
to obtain both the most recent Hitachi and BACI data, saved to folders `data/Hitachi` and `data/BACI`, respectively. 

## Command Line Usage
To compare the data in the year 2019 at the HS 6-digit product level, run
```zsh
python compare.py --year 2019 --hs_digits 6
```
Dataset comparisons are available for years 2019, 2020, 2021, and the recommended HS digit levels are 2,4, and 6 (for ease of product interpretability).
