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

DATA = 'tgbl-hypergraph_synthetic'
DATA_DIR = f"/lfs/turing1/0/{os.getlogin()}/supply-chains/TGB/tgb/datasets/{DATA.replace('-', '_')}/"

###################################################
# Functions to generate synthetic data
###################################################
def make_product_graph(num_exog=5, num_consumer=5, num_inner_layers=4, num_per_layer=10,
                       min_inputs=2, max_inputs=4, min_units=1, max_units=4, seed=0):
    """
    Generate DAG of products (edge is part-product relation).
    """
    np.random.seed(seed)
    num_prods = num_exog + num_consumer + (num_inner_layers*num_per_layer)
    prods = [f'product{i}' for i in range(num_prods)]
    prod_pos = np.random.random(size=(num_prods, 2))
    
    prod_graph = []
    start_idx = num_exog
    prev_prods = prods[:num_exog]
    prev_pos = prod_pos[:num_exog]
    layers = [0] * num_exog
    for l in range(num_inner_layers+1):
        if l < num_inner_layers:
            curr_prods = prods[start_idx:start_idx+num_per_layer]  # inner layer
            curr_pos = prod_pos[start_idx:start_idx+num_per_layer]
        else:
            curr_prods = prods[start_idx:]  # consumer prods
            curr_pos = prod_pos[start_idx:]
        
        dists = pairwise_distances(curr_pos, prev_pos)
        utils = np.exp(-dists)
        for i, p in enumerate(curr_prods):
            num_inputs = np.random.randint(min_inputs, max_inputs+1)
            sorted_prev = np.argsort(dists[i])
            inputs = [prev_prods[j] for j in sorted_prev[:num_inputs]]  # use closest products as inputs
            units = np.random.randint(min_units, max_units+1, size=num_inputs)
            prod_graph.extend(list(zip(inputs, [p] * num_inputs, units, [l+1] * num_inputs)))  # source, dest, units, layer
            layers.append(l+1)
            
        start_idx += num_per_layer
        prev_prods = curr_prods
        prev_pos = curr_pos
    
    prod_graph = pd.DataFrame(prod_graph, columns=['source', 'dest', 'units', 'layer'])
    print(f'Made product graph: {num_prods} products, {len(prod_graph)} part-product edges')
    consumer_prods = set(prod_graph.dest.values) - set(prod_graph.source.values)  # consumer products, not used as input    
    print(f'{len(consumer_prods)} consumer products (ie, not inputs for anything)')
    return prods, layers, prod_pos, prod_graph


def make_supplier_product_graph(max_firms, products, prod_pos, seed=0):
    """
    Assign products to supplier firms.
    """
    np.random.seed(seed)
    firms = [f'firm{i}' for i in range(max_firms)]
    firm_pos = np.random.random(size=(max_firms, 2))
    dists = pairwise_distances(prod_pos, firm_pos)
    utils = np.exp(-dists)
    
    prod2firms = {}  # product to supplier firms
    firm2prods = {f:[] for f in firms}  # firm to products supplied
    min_suppliers_per_prod = int(np.round(max_firms/100))
    max_suppliers_per_prod = min_suppliers_per_prod*4
    print(f'Num suppliers per product: {min_suppliers_per_prod}-{max_suppliers_per_prod}')

    for i, p in enumerate(products):
        num_suppliers = np.random.randint(min_suppliers_per_prod, max_suppliers_per_prod+1)
        sorted_firms = np.argsort(dists[i])
        suppliers = [firms[j] for j in sorted_firms[:num_suppliers]]  # use closest firms as suppliers
        prod2firms[p] = suppliers
        for f in suppliers:
            firm2prods[f].append(p)
    to_keep = [j for j,f in enumerate(firms) if len(firm2prods[f]) > 0]  # keep firms with at least one product
    firms = list(np.array(firms)[to_keep])
    firm_pos = firm_pos[to_keep]
    print(f'Keeping {len(firms)} out of {max_firms} firms with at least one product')
    return firms, firm_pos, firm2prods, prod2firms
    
    
def make_supplier_buyer_graph(firms, prod_graph, firm2prods, prod2firms, seed=0):
    """
    Assign supplier firms to buyer firms.
    """
    np.random.seed(seed)
    inputs2supplier = {}  # (buyer firm, product) to supplier firm
    num_buyers_per_firm = {f:0 for f in firms}
    for f in firms:
        inputs_needed = prod_graph[prod_graph['dest'].isin(firm2prods[f])].source.unique()  # all inputs needed by f
        for src in inputs_needed:
            suppliers = prod2firms[src]  # all possible suppliers for this input
            utils = np.array([num_buyers_per_firm[s] for s in suppliers]) + 1  
            supp = np.random.choice(suppliers, p=utils/np.sum(utils))  # preferential attachment
            inputs2supplier[(f, src)] = supp
            num_buyers_per_firm[supp] = num_buyers_per_firm[supp] + 1
    print(f'Assigned suppliers to (buyer, product): {len(inputs2supplier)} edges')
    return inputs2supplier
    
    
def generate_static_graphs(max_firms, seed=0):
    """
    Generate static graphs: product-product, firm-product, firm-firm.
    """
    products, layers, prod_pos, prod_graph = make_product_graph(seed=seed)
    firms, firm_pos, firm2prods, prod2firms = make_supplier_product_graph(max_firms, products, prod_pos, seed=seed)    
    inputs2supplier = make_supplier_buyer_graph(firms, prod_graph, firm2prods, prod2firms, seed=seed)
    return firms, products, prod_graph, firm2prods, prod2firms, inputs2supplier
    

def generate_initial_conditions(firms, products, prod_graph, prod2firms, init_inv=0, init_supply=100, init_demand=1):
    """
    Generate initial conditions for simulation. No need for seed because deterministic.
    """
    # initialize inventories
    inventories = np.ones((len(firms), len(products))) * init_inv
    
    # initialize supply for exogenous products
    exog_prods = set(prod_graph.source.values) - set(prod_graph.dest.values)  # exogenous products, has no inputs
    assert sorted(exog_prods) == sorted(products[:len(exog_prods)])
    exog_supp = {}  # maps (firm, product) to supply
    for p in exog_prods:
        for f in prod2firms[p]:
            exog_supp[(f,p)] = init_supply  # start with steady supply of exogeneous products
    
    # initialize demand for consumer products
    consumer_prods = set(prod_graph.dest.values) - set(prod_graph.source.values)  # consumer products, not used as input
    curr_orders = {}  # maps (supplier, product) to list of orders as (buyer, amount)
    for p in products:
        for f in prod2firms[p]:
            if p in consumer_prods and init_demand > 0:
                curr_orders[(f,p)] = [('consumer', init_demand)]  # start with steady demand for consumer products
            else:
                curr_orders[(f,p)] = []
            
    return inventories, curr_orders, exog_supp
    
    
def generate_demand_schedule(num_timesteps, prod_graph, prod2firms, seed=0, min_demand=0.5, init_demand=2):
    """
    Generate demand schedule for consumer products.
    """
    np.random.seed(seed)
    consumer_prods = sorted(set(prod_graph.dest.values) - set(prod_graph.source.values))  # consumer products, not used as input
    prod_types = ['weekend', 'weekday', 'uniform']
    
    prod2demand = {p:init_demand for p in consumer_prods}
    demand_schedule = {}  # t -> (firm, product) -> demand
    for t in range(num_timesteps):
        demand_schedule_t = {}
        for i, p in enumerate(consumer_prods):
            prev_demand = prod2demand[p]
            curr_demand = max(min_demand, prev_demand + np.random.normal(loc=0, scale=0.1))  # random drift
            prod2demand[p] = curr_demand
            
            prod_type = prod_types[i%3]
            if prod_type == 'uniform':  # same every day
                demand = curr_demand
            elif prod_type == 'weekday':  # higher on weekdays, lower on weekends
                if (t%7) < 5:
                    demand = curr_demand * 2
                else:
                    demand = curr_demand * 0.5
            else:  # higher on weekends, lower on weekdays
                if (t%7) < 5:
                    demand = curr_demand * 0.5
                else:
                    demand = curr_demand * 2
            
            for f in prod2firms[p]:
                fp_demand = np.random.poisson(demand)
                demand_schedule_t[(f,p)] = fp_demand
        demand_schedule[t] = demand_schedule_t
    return demand_schedule
    

def generate_exog_schedule_with_shocks(num_timesteps, prod_graph, prod2firms, seed=0, 
                                       default_supply=1e6, shock_supply=100, shock_prob=0.001, 
                                       shock_probs=None, recovery_rate=1.25):
    """
    Generate schedule of supply for exogenous products with possible shocks to supply.
    """
    np.random.seed(seed)
    exog_prods = sorted(set(prod_graph.source.values) - set(prod_graph.dest.values))  # exog products, only used as input
    if shock_probs is None:
        expected_num_shocks = len(exog_prods) * num_timesteps * shock_prob
    else:
        expected_num_shocks = np.sum(len(exog_prods) * shock_probs)
    print(f'Found {len(exog_prods)} exogenous products -> expected num shocks = {expected_num_shocks:0.3f}')
    # default_supply = shock_supply * recovery_rate^k 
    # log default_supply = log shock_supply + k log recovery_rate
    # (log default_supply - log shock_supply) / log recovery_rate = k
    time_to_recovery = (np.log(default_supply) - np.log(shock_supply)) / np.log(recovery_rate)
    print(f'Default supply = {default_supply}, shock supply = {shock_supply}, recovery rate = {recovery_rate} -> timesteps to recovery = {time_to_recovery: 0.2f}')
    
    prod2supply = {p:default_supply for p in exog_prods}
    exog_schedule = {}  # t -> (firm, product) -> supply
    for t in range(num_timesteps):
        exog_supp_t = {}
        for p in exog_prods:
            prev_supp = prod2supply[p]
            prob = shock_prob if shock_probs is None else shock_probs[t]
            if np.random.rand() < prob:   # shock occurred
                print(f'Shock to {p} at time {t}')
                curr_supp = shock_supply
            elif prev_supp < default_supply:  # in recovery
                curr_supp = min(default_supply, prev_supp*recovery_rate)
            else:  # at default supply
                assert prev_supp == default_supply
                curr_supp = default_supply
            prod2supply[p] = curr_supp
            for f in prod2firms[p]:
                fp_supp = np.random.poisson(curr_supp)
                exog_supp_t[(f, p)] = fp_supp
        exog_schedule[t] = exog_supp_t
    return exog_schedule
    
    
def generate_transactions(num_timesteps, inventories, curr_orders, exog_supp,  # time-varying info
                          firms, products, firm2idx, prod2idx,  # nodes + indexing
                          prod_graph, firm2prods, prod2firms, inputs2supplier,  # static graphs
                          exog_schedule=None, demand_schedule=None, gamma=0.8, num_decimals=5, 
                          seed=0, debug=False):
    """
    Generate transactions using agent-based model to determine firm's actions per timestep.
    Inputs:
        inventories: firm x product, shape: (n_firms x n_products)
        curr_orders: dict of (supplier, product) -> list of (buyer, amount)
        exog_supp: current supply per (firm, product) for all exogenous products
        exog_schedule: dict of t -> supply per (firm, product) for all exogenous products
        demand_schedule: dict of t -> demand per (firm, product) for all consumer products
    
    Returns:
        transactions: a pd DataFrame of supplier_id, buyer_id, product_id, amount, time
    """
    np.random.seed(seed)
    all_transactions = []
    pending = np.zeros(inventories.shape)  # n_firms x n_products, firms put in order for product but haven't received
    consumer_prods = set(prod_graph.dest.values) - set(prod_graph.source.values)  # consumer products, not used as input 
    prod_mat = get_prod_mat(prod_graph, prod2idx)
    if gamma < 1:
        print(f'Using same-supplier probability gamma = {gamma}')
    
    for t in range(num_timesteps):      
        # add new demand from consumers
        if demand_schedule is not None:
            demand_schedule_t = demand_schedule[t]
            for p in consumer_prods:
                for f in prod2firms[p]:
                    curr_orders[(f, p)].append(('consumer', demand_schedule_t[(f,p)]))
        
        # get new transactions at time t
        transactions_t = []
        all_inputs_needed = np.zeros(inventories.shape)  # n_firms x n_products
        rand_idx = np.random.choice(len(firms), replace=False, size=len(firms))
        firm_order = np.array(firms)[rand_idx]
        for f in firm_order:
            exog_supp_t = exog_supp if exog_schedule is None else exog_schedule[t]  # supply of exog products at time t
            inventories, curr_orders, transactions_completed, inputs_needed = simulate_actions_for_firm(
                f, inventories, curr_orders, exog_supp_t, firms, products, firm2idx, prod2idx, prod_mat, firm2prods, 
                prod2firms, inputs2supplier, debug=debug)
            transactions_t += transactions_completed
            all_inputs_needed[firm2idx[f]] = inputs_needed
            
        # update firms' inventories and pending based on completed transactions, record transactions
        if len(transactions_t) > 0:
            s_idxs, b_idxs, p_idxs, amts = list(zip(*transactions_t))
            buyer_product_mat = csr_matrix((amts, (b_idxs, p_idxs)), shape=(len(firms), len(products))).toarray()
            inventories = np.round(inventories+buyer_product_mat, num_decimals)  # add to buyers' inventories
            pending = np.round(pending-buyer_product_mat, num_decimals)  # subtract from pending
            
            transactions_df = pd.DataFrame(transactions_t, columns=['supplier_id', 'buyer_id', 'product_id', 'amount'])
            transactions_df['time'] = t
            all_transactions.append(transactions_df)
        
        # make new orders based on inputs needed, inventories, and pending
        all_inputs_needed = np.clip(all_inputs_needed - inventories - pending, 0, None)  # what is still needed
        all_inputs_needed = np.round(all_inputs_needed, num_decimals)
        for f in firm_order:  # allow firms to make new orders in random order - order matters here
            f_idx = firm2idx[f]
            inputs_needed = all_inputs_needed[f_idx]
            for p_idx in inputs_needed.nonzero()[0]:
                p = products[p_idx]
                if gamma < 1:
                    suppliers = prod2firms[p]
                    uni_prob = (1-gamma) / len(suppliers)  # with prob 1-gamma, randomly choose a supplier
                    probs = [gamma+uni_prob if s == inputs2supplier[(f,p)] else uni_prob for s in suppliers]
                    assert np.isclose(1, np.sum(probs))
                    s = np.random.choice(suppliers, p=probs)
                else:
                    s = inputs2supplier[(f,p)]  # deterministic
                curr_orders[(s, p)].append((f, inputs_needed[p_idx]))
                pending[f_idx, p_idx] += inputs_needed[p_idx]
                
        num_orders = np.sum([len(v) for v in curr_orders.values()])
        print(f't={t}: generated {len(transactions_t)} transactions, {num_orders} current orders')
        if num_orders == 0:
            print('No remaining orders, ending simulation')
            break
        
    all_transactions = pd.concat(all_transactions)
    assert len(all_transactions) == len(all_transactions.drop_duplicates(['supplier_id', 'buyer_id', 'product_id', 'time']))
    return all_transactions
        

def simulate_actions_for_firm(f, inventories, curr_orders, exog_supp,  # time-varying info
                              firms, products, firm2idx, prod2idx,  # nodes + indexing
                              prod_mat, firm2prods, prod2firms, inputs2supplier,  # static graphs
                              debug=False):
    """
    Simulate actions for a given firm f following agent-based model. No need for seed because deterministic.
    Inputs: see above
    Returns:
        inventories: same shape as before, modified with f's consumption subtracted
        curr_orders: same shape as before, modified with orders *to* f subtracted
        transactions_completed: transactions completed by f, where firm is suppler; list of 
                                (supplier_id, buyer_id, product_id, amount)
        inputs_needed: inputs needed to fulfill f's unfinished orders; shape: (n_products)
    """    
    inventories = inventories.copy()  # copy bc it'll be modified
    curr_orders = curr_orders.copy()  # copy bc it'll be modified
    amount_supplied = {}  # maps (buyer_id, product_id) to amount supplied in this round
    inputs_needed = np.zeros(len(products))  # what inputs are still needed for unfulfilled orders
    f_idx = firm2idx[f]
    
    for p in firm2prods[f]:  # order matters here because inventory changes
        if len(curr_orders[(f,p)]) == 0:
            if debug:
                print(f'No orders for {p}, skipping')
        else:
            p_idx = prod2idx[p]
            if debug:
                print(f'Processing {p}...')
                print_firm_and_product_status(f_idx, p_idx, inventories, curr_orders, inputs_needed, firms, products)            
            
            # complete as many orders as possible, first-in-first-out
            fp_orders = curr_orders[(f,p)]
            p_inputs = prod_mat[p_idx]  # inputs required to make p, as vector of length products
            order_num = 0
            if np.sum(p_inputs) == 0:  # exogenous product
                supply = exog_supp[(f,p)]
                for buyer, amt in fp_orders:
                    assert buyer != 'consumer'  # consumers don't buy exogenous products directly
                    bp_key = firm2idx[buyer], p_idx
                    if amt <= supply:
                        supply -= amt
                        amount_supplied[bp_key] = amount_supplied.get(bp_key, 0) + amt
                        order_num += 1 
                    else:
                        break
            else:
                for buyer, amt in fp_orders:
                    order_inputs_needed = amt * p_inputs  # inputs required to complete order
                    if (inventories[f_idx] >= order_inputs_needed).all():  # need to have enough of all inputs
                        inventories[f_idx] -= order_inputs_needed  # subtract from inventory
                        if buyer != 'consumer':
                            bp_key = firm2idx[buyer], p_idx
                            amount_supplied[bp_key] = amount_supplied.get(bp_key, 0) + amt
                        order_num += 1 
                    else:
                        break
            if debug:
                print(f'Got through {order_num} out of {len(fp_orders)} orders')
            curr_orders[(f,p)] = fp_orders[order_num:]
            
            # record what inputs are still needed for unfulfilled orders
            remaining = np.sum([amt for buyer, amt in curr_orders[(f,p)]])
            inputs_needed += remaining * p_inputs
                
            if debug: 
                print('After processing...')
                print_firm_and_product_status(f_idx, p_idx, inventories, curr_orders, inputs_needed, firms, products)
    
    transactions_completed = []
    for (b_idx, p_idx), amt in amount_supplied.items():
        transactions_completed.append((f_idx, b_idx, p_idx, amt))
    return inventories, curr_orders, transactions_completed, inputs_needed


def print_firm_and_product_status(f_idx, p_idx, inventories, curr_orders, inputs_needed, firms, products, 
                                  print_inv=False):
    """
    Print status of a given firm and product. Used for debgging.
    """
    if print_inv:
        # don't always print inventory because it can be very long
        curr_inv = inventories[f_idx]  # length n_products
        dict_curr_inv = {products[p_idx]:curr_inv[p_idx] for p_idx in np.nonzero(curr_inv)[0]}
        print('firm inventory', dict_curr_inv)
    
    print('orders for firm+product', curr_orders[(firms[f_idx], products[p_idx])])
    dict_inputs_needed = {products[s_idx]:inputs_needed[s_idx] for s_idx in np.nonzero(inputs_needed)[0]}
    print('inputs needed by firm', dict_inputs_needed)
    
    
def print_curr_orders(curr_orders, firms, products, max_print=10):
    """
    Print current orders. Used for debugging.
    """
    for (f,p), orders in curr_orders.items():
        print(f, p, orders[:max_print])
    buyer_idx, supplier_idx, prod_idx = np.nonzero(curr_orders)
    for b, s, p in zip(buyer_idx[:max_print], supplier_idx[:max_print], prod_idx[:max_print]):
        buyer = firms[b] if b < len(firms) else 'consumer'
        print(f'{buyer} ordered {curr_orders[b,s,p]:.5f} of {products[p]} from {firms[s]}')
    print()
    

###################################################
# Functions to analyze synthetic data
###################################################
def get_supply_chain_for_product(prod, prod_graph):
    """
    Get supply chain for a given product (ie, backwards BFS).
    """
    curr_layer = {prod}
    next_layer = set()
    seen = set()
    layers = []
    while len(curr_layer) > 0:
        layers.append(curr_layer)
        for p in curr_layer:
            parents = prod_graph[prod_graph['dest'] == p].source.values
            next_layer = next_layer.union(set(parents) - seen)
            seen.add(p)
        curr_layer = next_layer
        next_layer = set()
    return layers

def get_prod_mat(prod_graph, prod2idx):
    """
    Convert prod_graph, a pd DataFrame, to a matrix.
    """
    row_idx = [prod2idx[p] for p in prod_graph['dest'].values]
    col_idx = [prod2idx[p] for p in prod_graph['source'].values]
    m = csr_matrix((prod_graph['units'].values, (row_idx, col_idx)), shape=(len(prod2idx), len(prod2idx))).toarray()
    return m

def get_stats_on_firm_network(inputs2supplier):
    """
    Construct firm-firm network, report network stats.
    """
    edges = []
    for (buyer, product), supplier in inputs2supplier.items():
        edges.append((supplier, buyer))
    edges = list(set(edges))  # only keep unique edges
    print(f'Num supplier-buyer-product relations: {len(inputs2supplier)}; num supplier-buyer relations: {len(edges)}')
    G = nx.DiGraph()
    G.add_edges_from(edges)
    nodes = list(G.nodes())
    print(f'Found {len(nodes)} nodes')
    
    # degree: should be power law
    in_deg = [G.in_degree(f) for f in nodes]
    out_deg = [G.out_degree(f) for f in nodes]
    fig, axes = plt.subplots(1,2, figsize=(10, 4))
    axes[0].hist(in_deg, bins=30)
    axes[0].set_title('In-degree', fontsize=12)
    axes[1].hist(out_deg, bins=30)
    axes[1].set_title('Out-degree', fontsize=12)
    plt.show()
    
    # GWCC (should be most nodes) and GSCC (should be around half of nodes)
    gwcc = max(nx.weakly_connected_components(G), key=len)
    gwcc_prop = len(gwcc) / len(nodes)
    print(f'GWCC: {len(gwcc)}, {gwcc_prop: 0.3f} of firms')
    gscc = max(nx.strongly_connected_components(G), key=len)
    gscc_prop = len(gscc) / len(nodes)
    print(f'GSCC: {len(gscc)}, {gscc_prop: 0.3f} of firms')
    
    # degree assortativity: should be negative
    r = nx.degree_assortativity_coefficient(G)
    print(f'Degree assortativity: {r: 0.3f}')
    
    # triangles: should be low
    c = nx.average_clustering(G.to_undirected())
    print(f'Average clustering coef: {c: 0.3f}')
    

def convert_txns_to_timeseries(txns_df, min_t, max_t):
    """
    Convert a dataframe of transactions to a time series.
    """
    sums_per_t = txns_df.groupby('time')['amount'].sum()
    ts = []
    for t in range(min_t, max_t+1):
        if t in sums_per_t.index:
            ts.append(sums_per_t.loc[t])
        else:
            ts.append(0)
    return ts

def get_temporal_corr(buy_ts, supp_ts, lag=0, corr_func=pearsonr):
    """
    Get correlation between timeseries for buying a product vs supplying 
    another product, with possible lag between buying and supplying.
    """
    if lag > 0:
        lagged_buy_ts = buy_ts[:len(buy_ts)-lag]
        lagged_supp_ts = supp_ts[lag:]
    else:
        lagged_buy_ts = buy_ts
        lagged_supp_ts = supp_ts
    assert len(lagged_buy_ts) == len(lagged_supp_ts)
    return corr_func(lagged_buy_ts, lagged_supp_ts)

def get_best_corr_with_lag(buy_ts, supp_ts, max_lag=7):
    """
    Get best correlation between timeseries for buying a product vs supplying 
    another product, using the best possible lag between buying and supplying.
    """
    best_corr = -1
    best_lag = -1
    for lag in range(7):
        r,p = get_temporal_corr(buy_ts, supp_ts, lag=lag, corr_func=pearsonr)
        if r > best_corr:
            best_corr = r
            best_lag = lag
    return best_corr, best_lag


def eval_timeseries_for_product(prod, transactions, firms, products, firm2idx, prod2idx,
                                prod_graph, firm2prods, prod2firms, make_plots=True):
    """
    For a given product, iterate through its suppliers and plot time series of supplying this product vs
    buying other products, including its true parts. Evaluate correlation between time series.
    """
    p_idx = prod2idx[prod]
    true_sources = sorted(prod_graph[prod_graph['dest'] == prod]['source'].values)
    if len(true_sources) == 0:
        print('This is an exogenous product, no inputs to evaluate')
        return
    src2dests = {}
    num_layers = []
    for src in true_sources:
        src2dests[src] = set(prod_graph[prod_graph['source'] == src].dest.values)
        layers = get_supply_chain_for_product(src, prod_graph)
        num_layers.append(len(layers))
    assert len(set(num_layers)) == 1  # should all be the same layer, by construction
    print(f'True sources (layer {num_layers[0]}):', true_sources)
    
    # compare supply time series and buy time series per supplier
    conditions2corrs = {}
    min_t = np.min(transactions.time.values)
    max_t = np.max(transactions.time.values)
    time_range = range(min_t, max_t+1)
    for s in prod2firms[prod]:
        s_idx = firm2idx[s]
        print(f'SUPPLIER: {s}')
        supp_df = transactions[transactions['supplier_id'] == s_idx]  # all txns where s is supplying
        if p_idx in supp_df.product_id.values:
            all_prods_supplied = {products[p_idx] for p_idx in supp_df['product_id'].unique()}
            buy_df = transactions[transactions['buyer_id'] == s_idx]  # all txns where s is buying
            all_prods_bought = {products[p_idx] for p_idx in buy_df['product_id'].unique()}
            supp_ts = convert_txns_to_timeseries(supp_df[supp_df['product_id'] == p_idx], min_t, max_t)
            
            # get time series for true inputs
            pos_corrs = []
            if make_plots:
                fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
                fig.suptitle(s, fontsize=12)
                ax = axes[0]
                ax.plot(time_range, supp_ts, label=f'supply {prod}')
            for src in true_sources:
                buy_ts = convert_txns_to_timeseries(buy_df[buy_df['product_id'] == prod2idx[src]], min_t, max_t)
                best_corr, best_lag = get_best_corr_with_lag(buy_ts, supp_ts)
                pos_corrs.append(best_corr)
                dest_prods_supplied_by_s = all_prods_supplied.intersection(src2dests[src])
                print(f'src={src}: best corr = {best_corr:.3f}, lag = {best_lag}; input to {len(dest_prods_supplied_by_s)} product(s) supplied by firm')
                conds = (len(true_sources), num_layers[0], len(dest_prods_supplied_by_s))
                conditions2corrs[conds] = conditions2corrs.get(conds, []) + [best_corr]
                if make_plots:
                    ax.plot(time_range, buy_ts, label=f'buy {src}')

            # get time series for other product bought
            other_prods_bought = all_prods_bought - set(true_sources)
            neg_corrs = []
            if len(other_prods_bought) > 0: 
                print(f'Found {len(other_prods_bought)} OTHER products bought by {s}')
                for b in other_prods_bought:
                    buy_ts = convert_txns_to_timeseries(buy_df[buy_df['product_id'] == prod2idx[b]], min_t, max_t)
                    best_corr, best_lag = get_best_corr_with_lag(buy_ts, supp_ts)
                    neg_corrs.append(best_corr)
                print('Neg corrs:', np.round(neg_corrs, 3))
            if make_plots:
                ax.legend()
                ax = axes[1]
                ymin, ymax = ax.get_ylim()
                ax.vlines(pos_corrs, ymin, ymax, color='red')
                ax.vlines(neg_corrs, ymin, ymax, color='grey')
                plt.show()            
        else:
            print(f'Never supplied {prod} in transactions; skipping')
    return conditions2corrs


def measure_temporal_variation_in_triplets(transactions, verbose=False):
    """
    Measure how much variation there is over days in the set of triplets that appear.
    """
    seen = set()
    seen_prev = set()
    min_t = transactions.time.min()
    max_t = transactions.time.max()
    t_range = range(min_t, max_t+1)
    new_to_seen = []
    seen_missing = []
    new_to_seen_prev = []
    seen_prev_missing = []
    for t in t_range:
        txns = transactions[transactions.time == t]
        unique_triplets = set(zip(txns.supplier_id, txns.buyer_id, txns.product_id))
        assert len(unique_triplets) == len(txns)
        if verbose: 
            print(f't={t} -> {len(unique_triplets)} triplets')
        
        if len(seen) > 0 and len(seen_prev) > 0:
            new_triplets = unique_triplets - seen
            prop_new = len(new_triplets)/len(unique_triplets)
            new_to_seen.append(prop_new)
            missing_triplets = seen - unique_triplets
            prop_missing = len(missing_triplets)/len(seen)
            seen_missing.append(prop_missing)
            if verbose:
                print(f'Relative to all triplets: {len(new_triplets)} ({prop_new*100:0.2f}%) are new')
                print(f'Out of all triplets: {len(missing_triplets)} ({prop_missing*100:0.2f}%) not seen today')

            new_triplets = unique_triplets - seen_prev
            prop_new = len(new_triplets)/len(unique_triplets)
            new_to_seen_prev.append(prop_new)
            missing_triplets = seen_prev - unique_triplets
            prop_missing = len(missing_triplets)/len(seen_prev)
            seen_prev_missing.append(prop_missing)
            if verbose:
                print(f'Relative to yesterday\'s triplets: {len(new_triplets)} ({prop_new*100:0.2f}%) are new')
                print(f'Out of yesterday\'s triplets: {len(missing_triplets)} ({prop_missing*100:0.2f}%) not seen today')
        
        seen = seen.union(unique_triplets)
        seen_prev = unique_triplets
    
    plt.plot(t_range[1:], new_to_seen, label='New triplet, all')
    plt.plot(t_range[1:], new_to_seen_prev, label='New triplet, rel to t-1')
    plt.plot(t_range[1:], seen_missing, label='Missing triplet, all')
    plt.plot(t_range[1:], seen_prev_missing, label='Missing triplet, rel to t-1')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()
    
    
def check_negative_sampling(data_name, neg_samples=18, split='val'):
    """
    Check results from negative sampling.
    """
    data_dir = f"/lfs/turing1/0/{os.getlogin()}/supply-chains/TGB/tgb/datasets/{data_name.replace('-', '_')}/"
    edgelist_df = pd.read_csv(os.path.join(data_dir, f'{data_name}_edgelist.csv'))
    with open(os.path.join(data_dir, f'{data_name}_{split}_ns.pkl'), 'rb') as f:
        eval_ns = pickle.load(f)
    with open(os.path.join(data_dir, f'{data_name}_meta.json'), 'r') as f:
        meta = json.load(f)

    max_train = meta['train_max_ts'] 
    max_val = meta['val_max_ts'] 
    train_df = edgelist_df[edgelist_df.ts <= max_train]
    val_df = edgelist_df[(edgelist_df.ts > max_train) & (edgelist_df.ts <= max_val)]
    test_df = edgelist_df[edgelist_df.ts > max_val]
    print(f'Train len={len(train_df)}, Val len={len(val_df)}, Test len={len(test_df)}')

    train_triples = set(zip(train_df['source'].values, train_df['target'].values, train_df['product'].values))
    print(f'{len(train_triples)} unique train triples')
    if split == 'train':
        eval_keys = set(zip(train_df['source'].values, train_df['target'].values, 
                            train_df['product'].values, train_df['ts'].values))
        assert len(eval_ns) == len(train_df)
    elif split == 'val':
        eval_keys = set(zip(val_df['source'].values, val_df['target'].values, 
                            val_df['product'].values, val_df['ts'].values))
        assert len(eval_ns) == len(val_df)
    else:
        eval_keys = set(zip(test_df['source'].values, test_df['target'].values, 
                            test_df['product'].values, test_df['ts'].values))
        assert len(eval_ns) == len(test_df)

    num_historicals = []
    num_overlap_historicals = []
    for pos, negs in eval_ns.items():
        assert pos in eval_keys
        n_historical = 0
        n_perturb = 0
        n_overlap_historical = 0
        negs = [tuple(negs[i]) for i in range(neg_samples)]  # convert to list of tuples
        assert len(set(negs)) == neg_samples  # should be unique
        for neg in negs:
            assert (neg[0], neg[1], neg[2], pos[-1]) not in eval_keys
            overlap = np.sum([int(pos[i] == neg[i]) for i in range(3)])
            if neg in train_triples:
                n_historical += 1
                if overlap > 0:
                    n_overlap_historical += 1
            if overlap == 2:
                n_perturb += 1
        assert n_historical >= (neg_samples/2)  # at least 50% should be historical
        assert n_perturb >= (neg_samples/2)  # at least 50% should have one node perturbed
        num_historicals.append(n_historical)
        num_overlap_historicals.append(n_overlap_historical)

    for i in sorted(set(num_historicals)):
        num_data_points = np.sum(np.array(num_historicals) == i)
        print(f'Num historicals = {i} -> {num_data_points} ({num_data_points/len(eval_ns):0.3f})')
        
    for i in sorted(set(num_overlap_historicals)):
        num_data_points = np.sum(np.array(num_overlap_historicals) == i)
        print(f'Num overlap historicals = {i} -> {num_data_points} ({num_data_points/len(eval_ns):0.3f})')