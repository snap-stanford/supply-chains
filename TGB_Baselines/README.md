# TGB Baselines
A repository for benchmarking continuous-time dynamic graph models for link property prediction.
<h4>
	<a href="https://arxiv.org/abs/2307.01026"><img src="https://img.shields.io/badge/arXiv-pdf-yellowgreen"></a>
	<a href="https://pypi.org/project/py-tgb/"><img src="https://img.shields.io/pypi/v/py-tgb.svg?color=brightgreen"></a>
	<a href="https://tgb.complexdatalab.com/"><img src="https://img.shields.io/badge/website-blue"></a>
	<a href="https://docs.tgb.complexdatalab.com/"><img src="https://img.shields.io/badge/docs-orange"></a>
</h4>

## Overview
With the code provided in this repository, we benchmark the performance of several state-of-the-art continuous-time dynamic graph models on transductive link prediction tasks.

This repo utilizes the datasets and evaluation framework of [TGB](https://github.com/shenyangHuang/TGB/tree/main).
For further information about TGB, please consult TGB [website](https://tgb.complexdatalab.com) or its [repo](https://github.com/shenyangHuang/TGB/tree/main).


## Datasets
We benchmark the transductive dynamic link prediction task on the dataset provided by TGB for the dynamic link property prediction. 
These includes `tgbl-wiki`, `tgbl-review`, `tgbl-coin`, `tgbl-comment`, and `tgbl-flight`.
A summary of datasets cab be found on [TGB Learderboard](https://tgb.complexdatalab.com/docs/linkprop/).


## Temporal Graph Learning Models
The following continuous-time dynamic graph models can be utilized as TGB baselines for dynamic link property prediction task:

[JODIE](https://dl.acm.org/doi/10.1145/3292500.3330895), 
[DyRep](https://openreview.net/forum?id=HyePrhR5KX), 
[TGAT](https://openreview.net/forum?id=rJeW1yHYwH), 
[TGN](https://arxiv.org/abs/2006.10637), 
[CAWN](https://openreview.net/forum?id=KYPz4YsCPj), 
[EdgeBank](https://openreview.net/forum?id=1GVpwr2Tfdg), 
[TCL](https://arxiv.org/abs/2105.07944), 
[GraphMixer](https://openreview.net/forum?id=ayPPc0SyLv1), 
[DyGFormer](http://arxiv.org/abs/2303.13047).


## Transductive Dynamic Link Prediction
For training a model for transductive dynamic link property prediction on a dataset, you can use the following command:

```
dataset="tgbl-wiki"
model="GraphMixer"

python train_tgb_lpp.py --dataset_name "$dataset" --model_name "$model"
```
The above command trains and evaluates a `GraphMixer` model on the `tgbl-wiki` dataset.

The exact configuration arguments can be found in `utils/load_configs.py` file.


## Environments
The required dependencies are specified in the `requirements.txt` file.



## Acknowledgments
We are grateful to the authors of 
[DyGLib](https://github.com/yule-BUAA/DyGLib) and [TGB](https://github.com/shenyangHuang/TGB/tree/main)
for publicly sharing their project codes.


## Citation
If this repository is helpful for your research, please consider citing TGB.


```{bibtex}
@article{huang2023temporal,
  title={Temporal Graph Benchmark for Machine Learning on Temporal Graphs},
  author={Huang, Shenyang and Poursafaei, Farimah and Danovitch, Jacob and Fey, Matthias and Hu, Weihua and Rossi, Emanuele and Leskovec, Jure and Bronstein, Michael and Rabusseau, Guillaume and Rabbany, Reihaneh},
  journal={arXiv preprint arXiv:2307.01026},
  year={2023}
}
```