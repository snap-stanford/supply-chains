# Learning production functions from temporal graphs
This repo contains code for the paper "Learning production functions from temporal graphs" (under review). Our code extends code from the [TGB](https://github.com/shenyangHuang/TGB) and [TGB_Baselines](https://github.com/fpour/TGB_Baselines) repositories. 

## Installation
```
pip install -r requirements.txt
pip install -e ./TGB/
```

## File Description
- **register_data**: convert raw data into standardized transaction-level data, construct hypergraphs 
- **TGB/examples/linkproppred/general**: run experiments
- **TGB/modules**: model architecture and training modules 
- **TGB/tgb/datasets**: pre- and post-processed datasets 
- **TGB/tgb/linkproppred**: dataset, logging, evaluation, and negative sampling frameworks
- **TGB/tgb/utils**: constants and utils for experiments

## Data
We run experiments on the two real-world supply chain datasets and three synthetic datasets from our simulator, SupplySim: a standard setting with high exogenous supply (“SS-std”), a setting with shocks to exogenous supply (“SS-shocks”), and a setting with missing transactions (“SS-missing”). We release synthetic datasets in this repository, but we are not able to release the real-world datasets due to their proprietary nature. 

### Generate synthetic datasets
SupplySim is implemented in `./TGB/modules/synthetic_data.py`. This file contains code to generate static graphs, generate exogenous supply and demand schedules, and generate the time-varying transactions. See Section 3.2 and Appendix B.2 for details. Once the transactions are generated, create a new directory called `{DATASET_NAME}` in `./TGB/tgb/datasets/`, and save the transactions as `{DATASET_NAME}_transactions.csv`.

### Preprocess real-world data
We also built pipelines to work with real-world datasets (Tesla and SEM). See Section 3.1 and Appendix B.1 for details on data sources. 
First, create a directory named `{DATASET_NAME}` in the `./TGB/tgb/datasets/` directory and place the raw CSV files inside. Make sure the raw CSV file contains transaction-level information including but not limited to time, supplier, buyer, product hs code, and amount (measured in quntity, weight, price, or etc.).  

Next, transform the raw data into the standardized format `{DATASET_NAME}_transactions.csv` by running the following command in the root directory:
```
python ./register_data/preprocess_{DATASET_NAME}.py --ARGS
```
This preprocessing file will differ slightly across datasets based on dataset properties. 

### Prepare data for model experiments
Finally, transform the transactions data, saved as `{DATASET_NAME}_transactions.csv` for both synthetic and real-world data, into the format expected by model experiments (e.g., represent as hypergraph, do negative sampling). To do this, run the following command in the root directory: 
```
python ./register_data/register_hypergraph.py --ARGS
```

This command will generate files including the edgelist, sampled negatives for the train/val/test splits, and supplementary metadata (e.g., mapping fom node IDs to firm & product names).

## Model Experiments
The repository supports a large variety of training setups, so here we list the commands for running model experiments described in the submission. 

### Learning production functions
`./TGB/modules/prod_learning.py` contains code to test the inventory module and different baseline methods at learning production functions. In the paper, we reported three baselines including temporal correlations, PMI, and node2vec, and reported results when learning attention directly and when using product embeddings (see Table 1). For example, if we use 
```
compare_methods_on_data('sem', ['corr', 'pmi', 'node2vec', 'inventory'], save_results=True)
```
we will obtain results on the SEM dataset for those four methods, and results are saved.

### Predicting future edges
For these experiments, `cd` into the `./TGB/examples/linkproppred/general/` directory. In the commands below, remember to specify your dataset of interest, and if applicable utilize hyperparameters reported in Appendix C.2 Table 4.

To run Edgebank, use the following command:
```
python test_hyper_edgebank.py
```

To run SC-TGN, use the following command:
```
python model_experiments.py --train_with_fixed_samples --model tgnpl --memory_name tgnpl --emb_name attn --ARGS
``` 
Other variations of above including static, graph transformer, SC-TGN (id) replace `--memory_name` and `--emb_name` arguments with `static` and `id`, `static` and `attn`, and `tgnpl` and `id`, respectively. The model is comparable with the original TGN (Huang et al., 2023), if one appends two additional arguments `--init_memory_not_learnable` and `--update_penalty 0` to the above command (see more justification of SC-TGN vs TGN in Appendix A.2).

To run SC-GraphMixer, use the following command:
```
python model_experiments.py --train_with_fixed_samples --model graphmixer --ARGS
```

To include the inventory module, simply add the flag `--use_inventory`. There are also a number of other optional parameters related to the inventory module. For example, we find that providing initial attention weights via `--att_weights` helps with model training. Optionally, one can skip amount prediction by using `--skip_amount`. When the ground-truth production functions are known, they can also be provided via `--prod_graph {PROD_GRAPH_FILE}.pkl`.
