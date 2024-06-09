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
import time

from inventory_module import TGNPLInventory, mean_average_precision
from synthetic_data import *

PATH_TO_DATASETS = f'/lfs/local/0/{os.getlogin()}/supply-chains/TGB/tgb/datasets/'
SEM_DATA_NAME = 'tgbl-hypergraph_sem_23_subset'
SEM_CODES = [901210, 902780, 903141] # , 903180]
ALL_METHODS = ['random', 'corr', 'pmi', 'node2vec', 'inventory', 'inventory-corr', 
               'inventory-node2vec', 'inventory-emb', 'inventory-tgnpl']


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
    

def train_node2vec_on_firm_product_graph(transactions, firms, products, out_file, emb_dim=64):
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
    

def predict_product_relations_with_node2vec(emb_file, prod2idx, products_to_test=None):
    """
    Create matrix of predicted relationships between products based on cosine similarity in node2vec embeddings.
    """
    with open(emb_file, 'rb') as f:
        prod_embs = pickle.load(f)
    if products_to_test is None:
        products_to_test = prod2idx.keys()
    m = np.zeros((len(prod2idx), len(prod2idx)))
    for p1 in products_to_test:
        i = prod2idx[p1]
        norm1 = np.linalg.norm(prod_embs[i])
        if norm1 > 0:
            for p2, j in prod2idx.items():
                norm2 = np.linalg.norm(prod_embs[j])
                if i != j and norm2 > 0:
                    cos_sim = (prod_embs[i] @ prod_embs[j]) / (norm1 * norm2)
                    m[i,j] = cos_sim
    return m


def predict_product_relations_with_inventory_module(transactions, firms, products, prod2idx, prod_graph, 
                                                    module_args, num_epochs=100, show_weights=False,
                                                    prod_emb=None, products_to_test=None, patience=10):
    """
    Train inventory module on transactions, return final attention weights.
    """
    assert np.isin(['ts', 'source', 'target', 'product', 'weight'], transactions.columns).all()
    # initialize inventory module
    if module_args.get('device', None) is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module_args['device'] = device
    device = module_args['device']
    module = TGNPLInventory(**module_args)                
    opt = torch.optim.Adam(module.parameters())
    if prod_emb is not None:
        assert not module.learn_att_direct
        prod_emb = torch.Tensor(prod_emb).to(device)
        assert module.emb_dim == prod_emb.shape[1]
        
    # MAP before training
    weights = module._get_prod_attention(prod_emb=prod_emb).cpu().detach().numpy()
    prod_map = mean_average_precision(prod_graph, prod2idx, weights, products_to_test=products_to_test)
    print(f'Before training: MAP={prod_map:0.4f}')
    
    # train inventory module
    timesteps = transactions.ts.unique()
    losses = []
    maps = []
    no_improvement = 0
    best_weights = None
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
            
        prod_map = mean_average_precision(prod_graph, prod2idx, weights, products_to_test=products_to_test)
        duration = time.time()-start_time
        print(f'Ep {ep}: MAP={prod_map:0.4f}, loss={loss:0.4f}, debt_loss={debt_loss:0.4f}, consump_rwd={consump_rwd:0.4f} [time={duration:0.2f}s]')
        if len(maps) == 0 or prod_map > np.max(maps):
            no_improvement = 0
            best_weights = weights.copy()
        else: 
            no_improvement += 1
        maps.append(prod_map)
        
        if no_improvement >= patience:
            print(f'Early stopping since MAP did not improve for {patience} epochs')
            break
    print(f'Best MAP: {np.max(maps):0.4f}')
    return losses, maps, best_weights, module


def test_inventory_module_for_multiple_seeds(num_seeds, train_transactions, firms, products, prod2idx, 
                                             prod_graph, module_args, prod_emb=None, products_to_test=None):
    """
    Helper function to test inventory module over multiple seeds.
    """
    maps = []
    mats = []
    for s in range(num_seeds):
        print(f'\n=== SEED {s} ===')
        module_args['seed'] = s
        _, maps_s, mat_s, _ = predict_product_relations_with_inventory_module(
            train_transactions, firms, products, prod2idx, prod_graph, module_args, 
            prod_emb=prod_emb, products_to_test=products_to_test)
        maps.append(np.max(maps_s))
        mats.append(mat_s)
    print(f'\nOver {num_seeds} seeds: mean MAP={np.mean(maps):0.4f}, std={np.std(maps):0.4f}')
    return maps, mats


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


def load_synthetic_data(name, only_train=True):
    """
    Load synthetic data.
    """
    with open('./synthetic_data.pkl', 'rb') as f:
        firms, products, prod_graph, firm2prods, prod2firms, inputs2supplier = pickle.load(f)
    firm2idx = {f:i for i,f in enumerate(firms)}
    prod2idx = {p:i for i,p in enumerate(products)}
    results = {}
    
    data_name = f'tgbl-hypergraph_synthetic_{name}'
    data_dir = os.path.join(PATH_TO_DATASETS, data_name.replace('-', '_'))
    # load transactions
    transactions = pd.read_csv(os.path.join(data_dir, f'{data_name}_edgelist.csv'))
    assert (transactions.ts.values == sorted(transactions.ts.values)).all()
    assert ((transactions['source'] >= 0) & (transactions['source'] < len(firms))).all()
    assert ((transactions['target'] >= 0) & (transactions['target'] < len(firms))).all()
    assert ((transactions['product'] >= len(firms)) & (transactions['product'] < len(firms)+len(products))).all()    
    print(f'Loaded {len(transactions)} transactions')
    transactions['product'] = transactions['product']-len(firms)
    if only_train:
        train_max_ts = transactions.ts.quantile(0.7).astype(int)
        print('Train max ts:', train_max_ts)
        transactions = transactions[transactions.ts <= train_max_ts]
        print(f'Num train transactions: {len(transactions)}')  # should match link pred experiments
    return transactions, firms, firm2idx, products, prod2idx, prod_graph


def load_SEM_data(only_train=True):
    """
    Load SEM data.
    """
    data_dir = os.path.join(PATH_TO_DATASETS, SEM_DATA_NAME.replace('-', '_'))
    # load metadata
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
            
    # load prod_graph
    all_source = []
    all_dest = []
    for code in SEM_CODES:
        assert code in prod2idx
        parts_df = pd.read_csv(os.path.join(data_dir, f'{code}_parts.csv'))
        parts = parts_df['part_hs6'].unique()  # parts can be duplicated
        found_part = [p in prod2idx for p in parts]
        print(f'{code} -> num rows={len(parts_df)}, num parts={len(parts)}, found {np.mean(found_part):0.4f} in transactions')
        all_source.extend(list(parts))
        all_dest.extend([code] * len(parts))
    prod_graph = pd.DataFrame({'source': all_source, 'dest': all_dest})
    
    # load transactions
    transactions = pd.read_csv(os.path.join(data_dir, f'{SEM_DATA_NAME}_edgelist.csv'))
    assert (transactions.ts.values == sorted(transactions.ts.values)).all()
    print(f'Loaded {len(transactions)} transactions')
    transactions['product'] = transactions['product']-num_firms
    if only_train:
        train_max_ts = transactions.ts.quantile(0.7).astype(int)
        print('Train max ts:', train_max_ts)
        transactions = transactions[transactions.ts <= train_max_ts]
        print(f'Num train transactions: {len(transactions)}')  # should match link pred experiments
    return transactions, firms, firm2idx, products, prod2idx, prod_graph
    
    
def compare_methods_on_data(dataset, methods, synthetic_type=None, num_seeds=1, gpu=0, save_results=False):
    """
    Evaluate production learning methods on data.
    """
    for m in methods:
        assert m in ALL_METHODS
    assert dataset in ['synthetic', 'sem']
    if dataset == 'synthetic':
        assert synthetic_type is not None and synthetic_type in ['std', 'shocks', 'missing']
        transactions, firms, firm2idx, products, prod2idx, prod_graph = load_synthetic_data(synthetic_type)
        products_to_test = None
        postfix = 'synthetic_' + synthetic_type
    else:
        assert synthetic_type is None
        transactions, firms, firm2idx, products, prod2idx, prod_graph = load_SEM_data()
        products_to_test = SEM_CODES
        postfix = 'sem'
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    
    if 'corr' in methods:  # get correlations for all pairs of products
        print('\nPredicting production functions with temporal correlations...')
        corr_m = predict_product_relations_with_corr(transactions, prod2idx, products_to_test=products_to_test)
        if save_results:
            with open(f'corr_{postfix}.pkl', 'wb') as f:
                pickle.dump(corr_m, f)
        corr_map = mean_average_precision(prod_graph, prod2idx, corr_m, products_to_test=products_to_test)
        print(f'MAP: {corr_map:0.4f}')
            
    if 'pmi' in methods:
        print('\nPredicting production functions with PMI...')
        pmi_m = predict_product_relations_with_pmi(transactions, firms, prod2idx, products_to_test=products_to_test)
        pmi_map = mean_average_precision(prod_graph, prod2idx, pmi_m, products_to_test=products_to_test)
        print(f'MAP: {pmi_map:0.4f}')
            
    if 'node2vec' in methods:
        assert os.path.isfile(f'node2vec_embs_{postfix}.pkl')
        print('\nPredicting production functions with node2vec...')
        node2vec_m = predict_product_relations_with_node2vec(f'node2vec_embs_{postfix}.pkl', prod2idx, 
                                                             products_to_test=products_to_test)
        node2vec_map = mean_average_precision(prod_graph, prod2idx, node2vec_m, products_to_test=products_to_test)
        print(f'MAP: {node2vec_map:0.4f}')
    
    if 'inventory' in methods:
        print('\nPredicting production functions with inventory module, direct attention, random init...')
        module_args = {'num_firms': len(firms), 'num_prods': len(products), 'learn_att_direct': True,
                       'device': device}
        maps, mats = test_inventory_module_for_multiple_seeds(num_seeds, transactions, firms, products, prod2idx, 
                                                 prod_graph, module_args, products_to_test=products_to_test)
        if save_results:
            with open(f'inventory_{postfix}.pkl', 'wb') as f:
                pickle.dump(mats[np.argmax(maps)], f)
    
    if 'inventory-corr' in methods:
        print('\nPredicting production functions with inventory module, direct attention, initialized by correlations...')
        assert os.path.isfile(f'corr_{postfix}.pkl')
        with open(f'corr_{postfix}.pkl', 'rb') as f:
            corr_m = pickle.load(f)
        print(f'Loaded corr matrix from corr_{postfix}.pkl')
        module_args = {'num_firms': len(firms), 'num_prods': len(products), 'learn_att_direct': True,
                       'init_weights': corr_m, 'device': device}  # no randomness since weights are initialized
        losses, maps, weights, module = predict_product_relations_with_inventory_module(
            transactions, firms, products, prod2idx, prod_graph, module_args, products_to_test=products_to_test)
        if save_results:
            with open(f'inventory-corr_{postfix}.pkl', 'wb') as f:
                pickle.dump(weights, f)
            
    if 'inventory-node2vec' in methods:
        print('\nPredicting production functions with inventory module with node2vec embeddings...')
        assert os.path.isfile(f'node2vec_embs_{postfix}.pkl')
        with open(f'node2vec_embs_{postfix}.pkl', 'rb') as f:
            node2vec_embs = np.array(pickle.load(f))
        print(f'Loaded node2vec embs from node2vec_embs_{postfix}.pkl')
        module_args = {'num_firms': len(firms), 'num_prods': len(products), 'learn_att_direct': False,
                       'emb_dim': node2vec_embs.shape[1], 'device': device}
        maps, mats = test_inventory_module_for_multiple_seeds(num_seeds, transactions, firms, products, prod2idx, 
            prod_graph, module_args, prod_emb=node2vec_embs, products_to_test=products_to_test)
        if save_results:
            with open(f'inventory-node2vec_{postfix}.pkl', 'wb') as f:
                pickle.dump(mats[np.argmax(maps)], f)
            
    if 'inventory-emb' in methods:
        print('\nPredicting production functions with inventory module with product embeddings...')
        assert os.path.isfile(f'prod_embs_{postfix}.pkl')
        with open(f'prod_embs_{postfix}.pkl', 'rb') as f:
            prod_embs, transform = pickle.load(f)
            assert prod_embs.shape[0] == len(products)
            assert transform.shape == (prod_embs.shape[1], prod_embs.shape[1])
        init_bilinear = transform if dataset == 'sem' else None
        module_args = {'num_firms': len(firms), 'num_prods': len(products), 'learn_att_direct': False,
                       'emb_dim': prod_embs.shape[1], 'device': device, 'init_bilinear': init_bilinear}
        maps, mats = test_inventory_module_for_multiple_seeds(num_seeds, transactions, firms, products, prod2idx, 
            prod_graph, module_args, prod_emb=prod_embs, products_to_test=products_to_test)
        if save_results:
            with open(f'inventory-emb_{postfix}.pkl', 'wb') as f:
                pickle.dump(mats[np.argmax(maps)], f)
        
    if 'inventory-tgnpl' in methods:
        print('\nPredicting production functions with inventory module with TGN-PL product embeddings...')
        assert os.path.isfile(f'tgnpl_embs_{postfix}.pkl')
        with open(f'tgnpl_embs_{postfix}.pkl', 'rb') as f:
            prod_embs = pickle.load(f)
            assert prod_embs.shape[0] == len(products)
        module_args = {'num_firms': len(firms), 'num_prods': len(products), 'learn_att_direct': False,
                       'emb_dim': prod_embs.shape[1], 'device': device}
        maps, mats = test_inventory_module_for_multiple_seeds(num_seeds, transactions, firms, products, prod2idx, 
            prod_graph, module_args, prod_emb=prod_embs, products_to_test=products_to_test)
        if save_results:
            with open(f'inventory-tgnpl_{postfix}.pkl', 'wb') as f:
                pickle.dump(mats[np.argmax(maps)], f)
        
        
if __name__ == "__main__":        
    # gridsearch_on_hyperparameters()
    # transactions, firms, firm2idx, products, prod2idx, prod_graph = load_SEM_data()
    # train_node2vec_on_firm_product_graph(transactions, firms, products, 'node2vec_embs_sem.pkl')
    # compare_methods_on_data('synthetic', ALL_METHODS, synthetic_type='missing', num_seeds=10, gpu=1)
    # compare_methods_on_data('sem', ALL_METHODS, num_seeds=10, gpu=1)
    # compare_methods_on_data('synthetic', ['inventory-node2vec'], synthetic_type='std', num_seeds=10, gpu=4, save_results=True)
    # compare_methods_on_data('sem', ['inventory-emb'], num_seeds=10, gpu=3, save_results=True)
    compare_methods_on_data('sem', ['inventory-emb'], num_seeds=10, gpu=3, save_results=True)
