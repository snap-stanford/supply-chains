# Learning production functions from temporal graphs
This repo contains implementations and documentations of "Learning product functions from temporal graphs" submission. We extend our codes upon [TGB](https://github.com/shenyangHuang/TGB) and [TGB_Baselines](https://github.com/fpour/TGB_Baselines) repositories. 

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
- **TGB/tgb/utils**: constants and utilities for experiments

## Data
We run experiments on the two real-world supply chain datasets and three synthetic datasets from SupplySim: a standard setting with high exogenous supply (“SS-std”), a setting with shocks to
272 exogenous supply (“SS-shocks”), and a setting with missing transactions (“SS-missing”). We release synthetic datasets in this repository, but we are not able to release the real-world datasets due to their proprietary nature. 

### Generate synthetic datasets
Section 3.2 of the submission describes the supply chains simulator, SupplySim, in greater details. All codes are included in `./TGB/modules/synthetic_data.py` and are used by production learning experiments in `./TGB/modules/prod_learning.py`

### Generate Tesla, SEM, or customized datasets
We also built pipelines to work with real-world datasets and customized datasets. 
To start with, create a directory named `{DATASET_NAME}` in the `./TGB/tgb/datasets/` directory and place the raw CSV files inside. Make sure the raw CSV file contain transaction-level information including but not limited to time, supplier, buyer, product hs code, and amount (measured in quntity, weight, price, or etc.).  

Second, we morph the raw data into standardized format `{DATASET_NAME}_transactions.csv` by running the following command in the root directory:
```
python register_data/preprocess_{DATASET_NAME}.py --ARGS
```
This preprocess python file will differ slightly across datasets based on dataset properties. 

Third, we morph the standard, transaction-level data into the hypergraph format by running the following command in the root directory: 
```
python ./register_data/register_hypergraph.py --ARGS
```

After generation, we obtain files including the edge list, sampled negatives for the train/val/test splits, and supplementary metadata (e.g., mapping fom node IDs to firm & product names).

## Model Experiments
The repository supports a large variety of training setups, so here we list the commands for running model experiments described in the submission. 
### Learning production functions
To get results, run `./TGB/modules/prod_learning.py`. Specify dataset name with synthetic_type in ['std', 'shocks', 'missing'] (if applicable) and a subset of list of methods ['random', 'corr', 'pmi', 'node2vec', 'inventory', 'inventory-corr', 'inventory-node2vec', 'inventory-emb', 'inventory-tgnpl']for comparison. In the submission, we reported three baselines including temporal correltaions ('corr'), PMI ('pmi'), and node2vec ('node2vec'); and moreover, we reported results when learning attention directly ('inventory') and when using attention embedding ('inventory-emb'). If `--save_results` is on, one saves the learned inventory weights for later use. For example, if we use 
```
compare_methods_on_data('sem', ['inventory-emb'], save_results=True)
```
we will obtain `inventory-emb_sem.pkl` attention weights that can be loaded in later as initialization. We found this helpful for model trainings. 

### Predicting future edges
Remember to run under the `./TGB/examples/linkproppred/general/` directory, specify your dataset of interest, and if applicable utilize hyperparameters reported in Appendix C.2 Table 4.

To run Edgebank, use the following command:
```
python test_hyper_edgebank.py
```

To run SC-TGN, use the following command:
```
python model_experiments.py --train_with_fixed_samples --model tgnpl --memory_name tgnpl --emb_name attn --ARGS
``` 
Other variations of above including static, graph transformer, SC-TGN (id) replace `--memory_name` and `--emb_name` arguments with `static, id`, `static, attn`, and `tgnpl, id` respectively. The repository is compatible with original TGN (Huang et al., 2023), if one appends two additional arguments `--init_memory_not_learnable` and `--update_penalty 0` to the above command (see more justification of SC-TGN vs TGN in Appendix A.2).

To run SC-GraphMixer, use the following command:
```
python model_experiments.py --train_with_fixed_samples --model graphmixer --ARGS
```

To include inventory module, use the following command:
```
python model_experiments.py --train_with_fixed_samples --model graphmixer --use_inventory --learn_att_direct --att_weights inventory-{DATASET_NAME}.pkl --skip_inventory_penalties --ARGS
``` 
The inventory module stands independently from model choice, so the +inv experiments simply use the additional `--use_inventory` flag. We also include some additional training setups that help model learn slightly better. In particular, we find that providing initial attention weights via `--att_weights` helps with model training because inventory and model converge at different rate, so joint-learning tend to stuck the model in local optima. Synthetic datasets experiments have ground-truth production graphs, so one could use `--prod_graph {PROD_GRAPH_FILE}.pkl` to input the production graph file for additional evaluations. 

Optionally, one could skip amount predictions by using `--skip_amount`.