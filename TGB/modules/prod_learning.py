from collections import Counter
import json
import matplotlib.pyplot as plt
import networkx as nx
from node2vec import Node2Vec
import numpy as np
import os
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import pairwise_distances
import torch

from inventory_module import TGNPLInventory, mean_average_precision
from synthetic_data import *


def predict_product_relations_with_corr(transactions, prod2idx, min_nonzero=5, products_to_test=None,
                                        verbose=False):
    """
    Create matrix of predicted relationships between products based on temporal correlation.
    """
    assert np.isin(['ts', 'source', 'target', 'product', 'weight'], transactions.columns).all()
    if products_to_test is None:
        products_to_test = prod2idx.keys()
    min_t = np.min(transactions.ts.values)
    max_t = np.max(transactions.ts.values)
    m = np.zeros((len(prod2idx), len(prod2idx)))
    for prod in products_to_test:
        p_idx = prod2idx[prod]
        prod_txns = transactions[transactions['product'] == p_idx]  
        if verbose:
            print(prod, f'-> found {len(prod_txns)} transactions')
        if len(prod_txns) > 0:
            candidates = {}
            for s_idx, supp_df in prod_txns.groupby('source'):  # iterate through suppliers
                supp_ts = convert_txns_to_timeseries(supp_df, min_t, max_t, time_col='ts', amount_col='weight')
                if (supp_ts > 0).sum() >= min_nonzero:  # has at least min_nonzero nonzero weights
                    buy_df = transactions[transactions['target'] == s_idx]  # all txns where s is buying
                    for b_idx, buy_df_s in buy_df.groupby('product'):  # group by products bought
                        buy_ts = convert_txns_to_timeseries(buy_df_s, min_t, max_t, time_col='ts', amount_col='weight')
                        if (buy_ts > 0).sum() >= min_nonzero:
                            best_corr, best_lag = get_best_corr_with_lag(buy_ts, supp_ts)
                            candidates[b_idx] = candidates.get(b_idx, []) + [best_corr]
            for b_idx, corrs in candidates.items():
                m[p_idx, b_idx] = np.mean(corrs)
    return m


def predict_product_relations_with_inventory_module(transactions, firms, products, prod2idx, prod_graph, 
                                                    module_args, num_epochs=100, show_weights=True,
                                                    prod_emb=None, gpu=0):
    """
    Train inventory module on transactions, return final attention weights.
    """
    assert np.isin(['ts', 'source', 'target', 'product', 'weight'], transactions.columns).all()
    # initialize inventory module
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    module_args['device'] = device
    module = TGNPLInventory(**module_args)                
    opt = torch.optim.Adam(module.parameters())
    if prod_emb is not None:
        assert not module.learn_att_direct
        prod_emb = torch.Tensor(prod_emb).to(device)
        assert module.emb_dim == prod_emb.shape[1]
    
    # train inventory module
    losses = []
    maps = []
    min_t = np.min(transactions.ts.values)
    max_t = np.max(transactions.ts.values)
    for ep in range(num_epochs):
        for t in range(min_t, max_t+1):
            to_keep = transactions.ts.values == t
            src = torch.Tensor(transactions['source'].values[to_keep]).long().to(device)
            dst = torch.Tensor(transactions['target'].values[to_keep]).long().to(device)
            prod = torch.Tensor(transactions['product'].values[to_keep]).long().to(device) + len(firms)
            time = torch.Tensor(transactions['ts'].values[to_keep]).long().to(device)
            msg = torch.Tensor(transactions['weight'].values[to_keep].reshape(-1, 1)).to(device)
            
            opt.zero_grad()
            loss, debt_loss, consump_rwd = module(src, dst, prod, time, msg, prod_emb=prod_emb)
            loss.backward(retain_graph=False)
            opt.step()
            module.detach()
            losses.append((float(loss), float(debt_loss), float(consump_rwd)))
            
        weights = module._get_prod_attention(prod_emb=prod_emb).cpu().detach().numpy()
        mean_avg_pr = mean_average_precision(prod_graph, prod2idx, weights)
        maps.append(mean_avg_pr)
        
        if show_weights and (ep % 5) == 0:
            pos = plt.imshow(weights)
            plt.colorbar(pos)
            plt.title(f'Ep {ep}')
            plt.show()
        module.reset()  # reset inventory
    return losses, maps, weights, module


def test_inventory_module_on_sem_data():
    """
    Test inventory module on SEM.
    """
    data_dir = '/lfs/local/0/serinac/supply-chains/TGB/tgb/datasets/tgbl_hypergraph_sem_22_23'
    data_name = 'tgbl-hypergraph_sem_22_23'
    
    with open(os.path.join(data_dir, f'{data_name}_meta.json'), "r") as f:
        metadata = json.load(f)
    num_nodes = len(metadata["id2entity"])  
    num_firms = metadata["product_threshold"]
    num_products = num_nodes - num_firms
    # create necessary mappings
    firms = []
    firm2idx = {}
    products = []
    prod2idx = {}
    for idx in range(num_nodes):
        entity = metadata['id2entity'][str(idx)]
        if idx < num_firms:
            firms.append(entity)
            firm2idx[entity] = idx
        else:
            idx -= num_firms
            products.append(entity)
            prod2idx[entity] = idx
    
    # load transactions
    transactions = pd.read_csv(os.path.join(data_dir, f'{data_name}_edgelist.csv'))
    assert (transactions.ts.values == sorted(transactions.ts.values)).all()
    transactions['product'] = transactions['product']-num_firms
    train_max_ts = transactions.ts.quantile(0.7).astype(int)
    train_transactions = transactions[transactions.ts <= train_max_ts]
    print(f'Num txns: {len(transactions)}, num train: {len(train_transactions)}')  # should match tgnpl experiments
    
    # make prod_graph
    sem_codes = [901210, 902780, 903141, 903180]
    all_parts = []
    for code in sem_codes:
        parts = pd.read_csv(os.path.join(data_dir, f'{code}_parts.csv'))
        print(f'{code} -> num parts={len(parts)}')
        parts = parts.rename(columns={'prod_hs6':'dest', 'part_hs6':'source'})
        all_parts.append(parts[['source', 'dest', 'description']])
    prod_graph = pd.concat(all_parts)

    module_args = {'num_firms': num_firms, 'num_prods': num_products, 'learn_att_direct': True}
    inv_m = predict_product_relations_with_inventory_module(
        train_transactions, firms, products, prod2idx, prod_graph, module_args, show_weights=True)
    return inv_m
    

def train_node2vec_on_product_firm_graph(transactions, firms, products, out_file, emb_dim=64):
    """
    Train inventory module with direct attention on synthetic data, return final attention weights.
    """
    assert np.isin(['ts', 'source', 'target', 'product', 'weight'], transactions.columns).all()
    # make firm-product graph
    G = nx.Graph()
    supp_prod = transactions.groupby(['source', 'product'])['weight'].sum().reset_index()
    supp_prod['supplier_name'] = supp_prod['source'].apply(lambda x: firms[x])
    supp_prod['product_name'] = supp_prod['product'].apply(lambda x: products[x])
    G.add_weighted_edges_from(zip(supp_prod['supplier_name'].values, supp_prod['product_name'].values, 
                                  supp_prod['weight'].values))
    buy_prod = transactions.groupby(['target', 'product'])['weight'].sum().reset_index()
    buy_prod['buyer_name'] = buy_prod['target'].apply(lambda x: firms[x])
    buy_prod['product_name'] = buy_prod['product'].apply(lambda x: products[x])
    G.add_weighted_edges_from(zip(buy_prod['buyer_name'].values, buy_prod['product_name'].values, 
                                  buy_prod['weight'].values))
    print(f'Constructed graph, found {len(G)} out of {len(firms)+len(products)} nodes')
    
    n2v = Node2Vec(G, dimensions=emb_dim, walk_length=30, num_walks=200, workers=4)
    model = n2v.fit(window=10, min_count=1, batch_words=4)
    prod_embs = []
    for p in products:
        if p in model.wv:
            prod_embs.append(model.wv[p])
        else:
            print('missing', p)
            prod_embs.append(np.zeros(emb_dim))
    with open(out_file, 'wb') as f:
        pickle.dump(prod_embs, f)
    

def predict_product_relations_with_node2vec(emb_file, products):
    """
    Create matrix of predicted relationships between products based on cosine similarity in node2vec embeddings.
    """
    with open(emb_file, 'rb') as f:
        prod_embs = pickle.load(f)
    m = np.zeros((len(products), len(products)))
    for i, p1 in enumerate(products):
        norm1 = np.linalg.norm(prod_embs[i])
        if norm1 > 0:
            for j, p2 in enumerate(products):
                norm2 = np.linalg.norm(prod_embs[j])
                if i != j and norm2 > 0:
                    cos_sim = (prod_embs[i] @ prod_embs[j]) / (norm1 * norm2)
                    m[i,j] = cos_sim
    return m


def test_inventory_module_for_multiple_seeds(num_seeds, train_transactions, firms, products, prod2idx, 
                                             prod_graph, module_args, prod_emb=None):
    """
    Helper function to test inventory module over multiple seeds.
    """
    mats = []
    maps = []
    for s in range(num_seeds):
        module_args['seed'] = s
        _, _, inv_m, _ = predict_product_relations_with_inventory_module(
            train_transactions, firms, products, prod2idx, prod_graph, module_args, 
            show_weights=False, prod_emb=prod_emb)
        inv_map = mean_average_precision(prod_graph, prod2idx, inv_m, verbose=False)
        print(f'seed={s}: MAP={inv_map:.4f}')
        mats.append(inv_m)
        maps.append(inv_map)
    return mats, maps


def gridsearch_on_hyperparameters(num_seeds=1):
    """
    Gridsearch over scaling between debt penalty vs consumption reward.
    """
    with open('./synthetic_data.pkl', 'rb') as f:
        firms, products, prod_graph, firm2prods, prod2firms, inputs2supplier = pickle.load(f)
    prod2idx = {p:i for i,p in enumerate(products)}
    
    transactions = pd.read_csv(f'./synthetic_standard.csv')
    print(f'Loaded all transactions -> {len(transactions)} transactions')    
    train_max_ts = transactions.time.quantile(0.7).astype(int)
    train_transactions = transactions[transactions.time <= train_max_ts]
    print(f'Num transactions: overall={len(transactions)}, train={len(train_transactions)}')  # should match tgnpl experiments
    print(f'Running inventory module, random init, experiments with {num_seeds} seeds')
    
    debt_penalty = 5
    for scaling in np.arange(0.2, 1.1, 0.2):
        consumption_reward = debt_penalty * scaling
        module_args = {'num_firms': len(firms), 'num_prods': len(products), 'learn_att_direct': True,
                       'debt_penalty': debt_penalty, 'consumption_reward': consumption_reward}
        mats, maps = test_inventory_module_for_multiple_seeds(num_seeds, train_transactions, firms, products, prod2idx, 
                                                              prod_graph, module_args)
        print(f'Scaling={scaling}: MAP mean={np.mean(maps):.4f}, std={np.std(maps):.4f}')


def compare_methods_on_dataset(dataset_name, try_corr=True, emb_name=None, save_results=False, num_seeds=1):
    """
    Evaluate production learning methods on standard synthetic data with different settings:
    varying amounts of missingness in transactions, varying amounts of missingness in firms.
    """
    with open('./synthetic_data.pkl', 'rb') as f:
        firms, products, prod_graph, firm2prods, prod2firms, inputs2supplier = pickle.load(f)
    firm2idx = {f:i for i,f in enumerate(firms)}
    prod2idx = {p:i for i,p in enumerate(products)}
    results = {}
    
    transactions = pd.read_csv(dataset_name)
    train_max_ts = transactions.time.quantile(0.7).astype(int)
    train_transactions = transactions[transactions.time <= train_max_ts]
    print(f'Num transactions: overall={len(transactions)}, train={len(train_transactions)}')  # should match tgnpl experiments
    print(f'Running inventory module, random init, experiments with {num_seeds} seeds')

    module_args = {'num_firms': len(firms), 'num_prods': len(products), 'learn_att_direct': True}
    mats, maps = test_inventory_module_for_multiple_seeds(num_seeds, train_transactions, firms, products, prod2idx, 
                                                         prod_graph, module_args)
    results['inventory module, direct'] = mats
    print(f'Inventory module, direct: MAP mean={np.mean(maps):.4f}, std={np.std(maps):.4f}')

    if try_corr:
        corr_m = predict_product_relations_with_corr(train_transactions, products)
        results['temporal correlations'] = corr_m
        corr_map = mean_average_precision(prod_graph, prod2idx, corr_m)
        print(f'Temporal correlations: MAP={corr_map:.4f}')

        module_args = {'num_firms': len(firms), 'num_prods': len(products), 'learn_att_direct': True,
                       'init_weights': corr_m}  # don't need seed here since weights are set
        losses, maps, inv_m, mod = predict_product_relations_with_inventory_module(
            train_transactions, firms, products, prod2idx, prod_graph, module_args, show_weights=False)
        results['inventory module, direct, init corr'] = inv_m
        inv_map = mean_average_precision(prod_graph, prod2idx, inv_m, verbose=False)
        print(f'Inventory module, init with temporal corr: MAP={inv_map:.4f}')
    
    if emb_name is not None:
        node2vec_m = predict_product_relations_with_node2vec(emb_name, products)
        results['node2vec'] = node2vec_m
        node2vec_map = mean_average_precision(prod_graph, prod2idx, node2vec_m)
        print(f'node2vec cosine sim: MAP={node2vec_map:.4f}')
        
        module_args = {'num_firms': len(firms), 'num_prods': len(products), 'learn_att_direct': True,
                       'init_weights': node2vec_m}  # don't need seed here since weights are set
        losses, maps, inv_m, mod = predict_product_relations_with_inventory_module(
            train_transactions, firms, products, prod2idx, prod_graph, module_args, show_weights=False)
        results['inventory module, direct, init node2vec'] = inv_m
        inv_map = mean_average_precision(prod_graph, prod2idx, inv_m, verbose=False)
        print(f'Inventory module, init with node2vec embs: MAP={inv_map:.4f}')
        
        with open(emb_name, 'rb') as f:
            prod_emb = pickle.load(f)
        assert len(prod_emb) == len(products)
        module_args = {'num_firms': len(firms), 'num_prods': len(products), 'learn_att_direct': False,
                       'emb_dim': len(prod_emb[0])}
        mats, maps = test_inventory_module_for_multiple_seeds(num_seeds, train_transactions, firms, products, prod2idx, 
                                                         prod_graph, module_args, prod_emb=prod_emb)
        results['inventory module + node2vec emb'] = mats
        print(f'Inventory module, with node2vec embs: MAP mean={np.mean(maps):.4f}, std={np.std(maps):.4f}')
    
    if save_results:
        dataset_prefix = dataset_name.split('.', 1)[0]
        fn = f'{dataset_prefix}_results.pkl'
        with open(fn, 'wb') as f:
            pickle.dump(results, f)
        print('Saved results at', fn)
        
        
if __name__ == "__main__":    
    # gridsearch_on_hyperparameters()
#     datasets = ['synthetic_standard.csv', 'supply_shocks.csv', 'missing_20_pct_firms.csv']
#     embs = ['prod_embs_synthetic_std.pkl', 'prod_embs_synthetic_shocks.pkl', 'prod_embs_synthetic_missing.pkl']
#     for d, e in zip(datasets, embs):
#         print(f'===== {d} =====')
#         compare_methods_on_dataset(d, emb_name=e, save_results=True, num_seeds=10)
#         print()
    test_inventory_module_on_sem_data()