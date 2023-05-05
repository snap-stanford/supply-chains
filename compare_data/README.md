# Comparing Supply Chain Data with Global Records 
## Dependencies
A base Conda environment should be sufficient for running all the code here, with the principal Python libraries being `matplotlib`, `numpy`, `scipy`, and `pandas`. 

## Data 
The Hitachi supply chain data resides on the JupyterLab server. Download the BACI data at <a href = `http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37`>dataset link</a>, selecting the most recent version, HS17. Unzip all the `csv` files to the same directory. 

## Command Line Usage
To compare the data in the year 2019 at the HS 6-digit product level, run
```zsh
python compare.py 2019 6
```
