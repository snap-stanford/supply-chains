from collections import Counter
import json
import matplotlib.pyplot as plt
import networkx as nx
# from node2vec import Node2Vec
import numpy as np
import os
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import pairwise_distances
import torch
import time

from inventory_module import TGNPLInventory, mean_average_precision
from synthetic_data import *

PATH_TO_DATASETS = f'/lfs/local/0/{os.getlogin()}/supply-chains/TGB/tgb/datasets/'
SEM_DATA_NAME = 'tgbl-hypergraph_sem_23_subset'
SEM_CODES = [901210, 902780, 903141] # , 903180]

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
            print(f'{prod} (id: {prod2idx[prod]}) -> found {len(prod_txns)} transactions')
        if len(prod_txns) >= min_nonzero:
            candidates = {}
            for s_idx, supp_df in prod_txns.groupby('source'):  # iterate through suppliers
                supp_ts = convert_txns_to_timeseries(supp_df, min_t, max_t, time_col='ts', amount_col='weight')
                if (supp_ts > 0).sum() >= min_nonzero:  # has at least min_nonzero nonzero timesteps
                    buy_df = transactions[transactions['target'] == s_idx]  # all txns where s is buying
                    for b_idx, buy_df_s in buy_df.groupby('product'):  # group by products bought
                        buy_ts = convert_txns_to_timeseries(buy_df_s, min_t, max_t, time_col='ts', amount_col='weight')
                        if (buy_ts > 0).sum() >= min_nonzero:
                            best_corr, best_lag = get_best_corr_with_lag(buy_ts, supp_ts)
                            candidates[b_idx] = candidates.get(b_idx, []) + [best_corr]
            for b_idx, corrs in candidates.items():
                m[p_idx, b_idx] = np.mean(corrs)
    return m


def predict_product_relations_with_pmi(transactions, firms, prod2idx, products_to_test=None, verbose=False):
    """
    Create matrix of predicted relationships between products based on Pointwise Mutual Information (PMI).
    """
    assert np.isin(['ts', 'source', 'target', 'product', 'weight'], transactions.columns).all()
    if products_to_test is None:
        products_to_test = prod2idx.keys()
    prob_supply = transactions.groupby('product')['source'].nunique() / len(firms)  # prob that a firm supplies each product
    prob_buy = transactions.groupby('product')['target'].nunique() / len(firms)  # prob that a firm buys each product
    m = np.zeros((len(prod2idx), len(prod2idx)))
    for prod in products_to_test:
        p_idx = prod2idx[prod]
        prod_txns = transactions[transactions['product'] == p_idx]  
        if verbose:
            print(f'{prod} (id: {prod2idx[prod]}) -> found {len(prod_txns)} transactions')
        suppliers = prod_txns['source'].unique()  # suppliers of this product
        buy_df = transactions[transactions['target'].isin(suppliers)]
        for b_idx, buy_df_b in buy_df.groupby('product'):
            prob_buy_b_supply_p = buy_df_b['target'].nunique() / len(firms)  # prob that a firm buys b and supplies p
            pmi = prob_buy_b_supply_p / (prob_buy[b_idx] * prob_supply[p_idx])
            m[p_idx, b_idx] = np.log(pmi)
    return m
        

def predict_product_relations_with_inventory_module(transactions, firms, products, prod2idx, prod_graph, 
                                                    module_args, num_epochs=100, show_weights=True,
                                                    prod_emb=None, gpu=0, products_to_test=None, patience=5):
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
    timesteps = transactions.ts.unique()
    no_improvement = 0
    if products_to_test is not None and len(products_to_test) < 10:
        return_per_prod = True
    else:
        return_per_prod = False
    for ep in range(num_epochs):
        start_time = time.time()
        loss, debt_loss, consump_rwd = 0, 0, 0
        for t in timesteps:
            to_keep = transactions.ts.values == t
            src = torch.Tensor(transactions['source'].values[to_keep]).long().to(device)
            dst = torch.Tensor(transactions['target'].values[to_keep]).long().to(device)
            prod = torch.Tensor(transactions['product'].values[to_keep]).long().to(device) + len(firms)
            ts = torch.Tensor(transactions['ts'].values[to_keep]).long().to(device)
            msg = torch.Tensor(transactions['weight'].values[to_keep].reshape(-1, 1)).to(device)
            
            opt.zero_grad()
            loss, debt_loss, consump_rwd = module(src, dst, prod, ts, msg, prod_emb=prod_emb)
            loss.backward(retain_graph=False)
            opt.step()
            module.detach()
            loss += float(loss)
            debt_loss += float(debt_loss)
            consump_rwd += float(consump_rwd)
        module.reset()  # reset inventory
        losses.append(float(loss))
            
        weights = module._get_prod_attention(prod_emb=prod_emb).cpu().detach().numpy()
        if show_weights and (ep % 5) == 0:
            pos = plt.imshow(weights)
            plt.colorbar(pos)
            plt.title(f'Ep {ep}')
            plt.show()
            
        avg_prec = mean_average_precision(prod_graph, prod2idx, weights, 
                        products_to_test=products_to_test, return_per_prod=return_per_prod)
        duration = time.time()-start_time
        if return_per_prod:
            mean_avg_prec = np.mean(list(avg_prec.values()))
            print(f'Ep {ep}: MAP={mean_avg_prec:0.4f}, loss={loss:0.4f}, debt_loss={debt_loss:0.4f}, consump_rwd={consump_rwd:0.4f} [time={duration:0.2f}s]')
            for p in products_to_test:
                print(f'{p} -> MAP={avg_prec[p]:0.4f}')
        else:
            mean_avg_prec = avg_prec
            print(f'Ep {ep}: MAP={mean_avg_prec:0.4f}, loss={loss:0.4f}, debt_loss={debt_loss:0.4f}, consump_rwd={consump_rwd:0.4f} [time={duration:0.2f}s]')
        if len(maps) > 0:
            if mean_avg_prec > np.max(maps):  # new best
                no_improvement = 0
            else: 
                no_improvement += 1
        maps.append(mean_avg_prec)
        
        if no_improvement > patience:
            print(f'Early stopping since MAP did not improve for {patience} epochs')
            break
    return losses, maps, weights, module
    

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
    train_max_ts = transactions.ts.quantile(0.7).astype(int)
    train_transactions = transactions[transactions.ts <= train_max_ts]
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


def prep_synthetic_data_for_prod_learning(name):
    """
    """
    with open('./synthetic_data.pkl', 'rb') as f:
        firms, products, prod_graph, firm2prods, prod2firms, inputs2supplier = pickle.load(f)
    firm2idx = {f:i for i,f in enumerate(firms)}
    prod2idx = {p:i for i,p in enumerate(products)}
    results = {}
    
    transactions = pd.read_csv(dataset_name)
    train_max_ts = transactions.ts.quantile(0.7).astype(int)
    train_transactions = transactions[transactions.ts <= train_max_ts]
    print(f'Num transactions: overall={len(transactions)}, train={len(train_transactions)}')  # should match tgnpl experiments
    print(f'Running inventory module, random init, experiments with {num_seeds} seeds')

    
def compare_methods_on_synthetic_data(data_dir, dataset_name, try_corr=True, emb_name=None, save_results=False, 
                                      num_seeds=1):
    """
    Evaluate production learning methods on synthetic data.
    """
    
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
        

def load_SEM_data():
    """
    Load SEM data.
    """
    data_dir = os.path.join(PATH_TO_DATASETS, SEM_DATA_NAME.replace('-', '_'))
    # make prod_graph
    all_parts = []
    for code in SEM_CODES:
        parts = pd.read_csv(os.path.join(data_dir, f'{code}_parts.csv'))
        print(f'{code} -> num parts={len(parts)}')
        parts = parts.rename(columns={'prod_hs6':'dest', 'part_hs6':'source'})
        all_parts.append(parts[['source', 'dest', 'description']])
    prod_graph = pd.concat(all_parts)
    
    with open(os.path.join(data_dir, f'{SEM_DATA_NAME}_meta.json'), "r") as f:
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
    transactions = pd.read_csv(os.path.join(data_dir, f'{SEM_DATA_NAME}_edgelist.csv'))
    assert (transactions.ts.values == sorted(transactions.ts.values)).all()
    transactions['product'] = transactions['product']-num_firms
    train_max_ts = transactions.ts.quantile(0.7).astype(int)
    print('Train max ts:', train_max_ts)
    train_transactions = transactions[transactions.ts <= train_max_ts]
    print(f'Num txns: {len(transactions)}, num train: {len(train_transactions)}')  # should match tgnpl experiments
    return train_transactions, firms, firm2idx, products, prod2idx, prod_graph
    
    
def compare_methods_on_sem_data(methods=[]):
    """
    Run method for learning production functions on SEM data.
    """
    for m in methods:
        assert m in ['random', 'corr', 'pmi', 'inventory', 'inventory-corr']
    transactions, firms, firm2idx, products, prod2idx, prod_graph = load_SEM_data()
    if 'random' in methods:
        avg_precs = {p:[] for p in SEM_CODES}
        avg_precs['mean'] = []
        print(f'\nTesting random matrices of size ({len(products)}, {len(products)})')
        for i in range(100):
            np.random.seed(i)
            rand_mat = np.random.random((len(products), len(products)))
            rand_map = mean_average_precision(prod_graph, prod2idx, rand_mat, products_to_test=SEM_CODES, 
                                              return_per_prod=True)
            for p in SEM_CODES:
                avg_precs[p].append(rand_map[p])
            avg_precs['mean'].append(np.mean(list(rand_map.values())))
        maps = avg_precs['mean']
        print(f'MAP: mean={np.mean(maps):0.4f}, std={np.std(maps):0.4f}')
        for p in SEM_CODES:
            p_avg_precs = avg_precs[p]
            print(f'{p} -> mean AP={np.mean(p_avg_precs):0.4f}, std={np.std(p_avg_precs):0.4f}')
    
    if 'corr' in methods:  # get correlations for all pairs of products
        start_time = time.time()
        print('\nPredicting production functions with temporal correlations...')
        corr_m = predict_product_relations_with_corr(transactions, prod2idx, products_to_test=SEM_CODES, verbose=True)
        with open('corr_sem.pkl', 'wb') as f:
            pickle.dump(corr_m, f)
        duration = time.time()-start_time
        print(f'Finished computing all correlations: time={duration:0.2f}s')
        corr_map = mean_average_precision(prod_graph, prod2idx, corr_m, products_to_test=SEM_CODES, return_per_prod=True)
        print(f'MAP: {np.mean(list(corr_map.values())):0.4f}')
        for p in SEM_CODES:
            print(f'{p} -> AP={corr_map[p]:0.4f}')
            
    if 'pmi' in methods:
        print('\nPredicting production functions with PMI...')
        pmi_m = predict_product_relations_with_pmi(transactions, firms, prod2idx, products_to_test=SEM_CODES,
                                       verbose=True)
        pmi_map = mean_average_precision(prod_graph, prod2idx, pmi_m, products_to_test=SEM_CODES, return_per_prod=True)
        print(f'MAP: {np.mean(list(pmi_map.values())):0.4f}')
        for p in SEM_CODES:
            print(f'{p} -> AP={pmi_map[p]:0.4f}')
            
    if 'node2vec' in methods:
        assert os.path.isfile('node2vec_sem.pkl')
        print('\nPredicting production functions with node2vec...')
        node2vec_m = predict_product_relations_with_node2vec('node2vec_sem.pkl', products)
        node2vec_map = mean_average_precision(prod_graph, prod2idx, node2vec_m, products_to_test=SEM_CODES, 
                                              return_per_prod=True)
        print(f'MAP: {np.mean(list(node2vec_map.values())):0.4f}')
        for p in SEM_CODES:
            print(f'{p} -> AP={node2vec_map[p]:0.4f}')
    
    if 'inventory' in methods:
        print('\nPredicting production functions with inventory module, direct attention, random init...')
        module_args = {'num_firms': len(firms), 'num_prods': len(products), 'learn_att_direct': True}
        losses, maps, weights, module = predict_product_relations_with_inventory_module(
            transactions, firms, products, prod2idx, prod_graph, module_args, show_weights=False,
            products_to_test=SEM_CODES)
        with open('inv_sem_direct.pkl', 'wb') as f:
            pickle.dump((losses, maps, weights), f)
    
    if 'inventory-corr' in methods:
        assert os.path.isfile('corr_sem.pkl')
        with open('corr_sem.pkl', 'rb') as f:
            corr_m = pickle.load(f)
        print('\nLoaded corr matrix from corr_sem.pkl')
        print('Predicting production functions with inventory module, direct attention, initialized by correlations...')
        module_args = {'num_firms': len(firms), 'num_prods': len(products), 'learn_att_direct': True,
                       'init_weights': corr_m, 'debt_penalty':5, 'consumption_reward':5}
        losses, maps, weights, module = predict_product_relations_with_inventory_module(
            transactions, firms, products, prod2idx, prod_graph, module_args, show_weights=False,
            products_to_test=SEM_CODES)
        with open('inv_sem_direct_init_corr.pkl', 'wb') as f:
            pickle.dump((losses, maps, weights), f)
        
        
if __name__ == "__main__":    
    # gridsearch_on_hyperparameters()
#     datasets = ['synthetic_standard.csv', 'supply_shocks.csv', 'missing_20_pct_firms.csv']
#     embs = ['prod_embs_synthetic_std.pkl', 'prod_embs_synthetic_shocks.pkl', 'prod_embs_synthetic_missing.pkl']
#     for d, e in zip(datasets, embs):
#         print(f'===== {d} =====')
#         compare_methods_on_synthetic_data(d, emb_name=e, save_results=True, num_seeds=10)
#         print()
    compare_methods_on_sem_data(methods=['corr', 'pmi', 'node2vec'])   
    