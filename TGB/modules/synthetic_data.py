from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, spearmanr
import torch

from inventory_module import TGNPLInventory


DATA = 'tgbl-hypergraph_synthetic'
DATA_DIR = f"/lfs/turing1/0/{os.getlogin()}/supply-chains/TGB/tgb/datasets/{DATA.replace('-', '_')}/"

###################################################
# Functions to generate synthetic data
###################################################
def generate_static_graphs(max_firms, seed=0, min_units=1, max_units=4):
    """
    Generate static graphs: product-product, firm-product, firm-firm.
    """
    np.random.seed(seed)
    # get product graph
    prod_graph = pd.read_table(os.path.join(DATA_DIR, 'product_graph.psv'), sep="|")
    prod_graph['units'] = np.random.randint(low=min_units, high=max_units+1, size=len(prod_graph))  # add units per source, dest
    G = nx.DiGraph()
    edges = list(zip(prod_graph.source.values, prod_graph.dest.values))
    G.add_edges_from(edges)
    assert len(list(nx.simple_cycles(G))) == 0  # check for cycles, there should be none
    products = list(nx.topological_sort(G))  # topological order
    
    # assign products to supplier firms
    firms = [f'firm{i}' for i in range(max_firms)]
    prod2firms = {}  # product to supplier firms
    firm2prods = {f:[] for f in firms}  # firm to products supplied
    num_prods_per_firm = np.zeros(len(firms))
    min_suppliers_per_prod = int(np.round(max_firms/100))
    max_suppliers_per_prod = min_suppliers_per_prod*4
    print(f'Num suppliers per product: {min_suppliers_per_prod}-{max_suppliers_per_prod}')
    for p in products:
        num_suppliers = np.random.randint(min_suppliers_per_prod, max_suppliers_per_prod+1)
        firm_probs = (num_prods_per_firm+1) / (np.sum(num_prods_per_firm)+len(firms))
        suppliers = np.random.choice(firms, size=num_suppliers, replace=False, p=firm_probs)  # preferential attachment
        prod2firms[p] = suppliers
        for f in suppliers:
            firm2prods[f].append(p)
            num_prods_per_firm[int(f[4:])] += 1  # first four letters are 'firm'
    firms = [f for f in firms if len(firm2prods[f]) > 0]  # keep firms with at least one product
    print(f'Keeping {len(firms)} out of {max_firms} firms with at least one product')
    firms = sorted(firms, key=lambda x: len(firm2prods[f]), reverse=True)  # sort from most to least products
    
    # assign supplier firms for each firm, based on products supplied 
    inputs2supplier = {}  # (buyer firm, product) to supplier firm
    for f in firms:
        inputs_needed = prod_graph[prod_graph['dest'].isin(firm2prods[f])].source.unique()  # all inputs needed by f
        for src in inputs_needed:
            supplier = np.random.choice(prod2firms[src])
            inputs2supplier[(f, src)] = supplier
    
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
    assert sorted(consumer_prods) == sorted(products[-len(consumer_prods):])
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
                                       default_supply=1000, shock_supply=10, shock_prob=0.001, 
                                       shock_probs=None, recovery_rate=1.2):
    """
    Generate schedule of supply for exogenous products with possible shocks to supply.
    """
    np.random.seed(seed)
    exog_prods = sorted(set(prod_graph.source.values) - set(prod_graph.dest.values))  # exog products, only used as input
    expected_num_shocks = len(exog_prods) * num_timesteps * shock_prob
    print(f'Found {len(exog_prods)} exogenous products -> expected num shocks = {expected_num_shocks:0.3f}')
    # default_supply = shock_supply * recovery_rate^k 
    # log default_supply = log shock_supply + k log recovery_rate
    # (log default_supply - log shock_supply) / log recovery_rate = k
    time_to_recovery = (np.log(default_supply) - np.log(shock_supply)) / np.log(recovery_rate)
    print(f'Recovery rate = {recovery_rate} -> num timesteps to recovery = {time_to_recovery: 0.2f}')
    
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
                          exog_schedule=None, demand_schedule=None, num_decimals=5, seed=0, debug=False):
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
                s = inputs2supplier[(f,p)]
                curr_orders[(s, p)].append((f, inputs_needed[p_idx]))
                pending[f_idx, p_idx] += inputs_needed[p_idx]
                
        num_orders = np.sum([len(v) for v in curr_orders.values()])
        print(f't={t}: generated {len(transactions_t)} transactions, {num_orders} current orders')
        if num_orders == 0:
            print('No remaining orders, ending simulation')
            break
        
    all_transactions = pd.concat(all_transactions)
    orig_len = len(all_transactions)
    # it's possible for there to be multiple of the same triplet on a day if a firm processes multiple orders
    # for the same buyer, product -> sum over amount
    all_transactions = all_transactions.groupby(['supplier_id', 'buyer_id', 'product_id', 'time']).sum().reset_index()
    print(f'Grouped by supplier, buyer, product, and time -> reduced {orig_len} transactions to {len(all_transactions)}')
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
    transactions_completed = []  # list of (supplier_id, buyer_id, product_id, amount)
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
                    if amt <= supply:
                        supply -= amt
                        transactions_completed.append((f_idx, firm2idx[buyer], p_idx, amt))
                        order_num += 1 
                    else:
                        break
            else:
                for buyer, amt in fp_orders:
                    order_inputs_needed = amt * p_inputs  # inputs required to complete order
                    if (inventories[f_idx] >= order_inputs_needed).all():  # need to have enough of all inputs
                        inventories[f_idx] -= order_inputs_needed  # subtract from inventory
                        if buyer != 'consumer':
                            transactions_completed.append((f_idx, firm2idx[buyer], p_idx, amt))
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
            if make_plots:
                ax.legend()
                ax = axes[1]
                ax.hist(neg_corrs, color='blue', bins=20)
                ymin, ymax = ax.get_ylim()
                ax.vlines(pos_corrs, ymin, ymax, color='red')
                plt.show()            
        else:
            print(f'Never supplied {prod} in transactions; skipping')
    return conditions2corrs


###################################################
# Functions to evaluate against ground-truth production functions
###################################################
def predict_product_relations_with_corr(transactions, products):
    """
    Create matrix of predicted relationships between products based on temporal correlation.
    """
    min_t = np.min(transactions.time.values)
    max_t = np.max(transactions.time.values)
    m = np.zeros((len(products), len(products)))
    for p_idx in range(len(products)):
        prod_txns = transactions[transactions['product_id'] == p_idx]  
        if len(prod_txns) > 0:  # should be 0 for consumer products
            candidates = {}
            for s_idx, supp_df in prod_txns.groupby('supplier_id'):  # iterate through suppliers
                supp_ts = convert_txns_to_timeseries(supp_df, min_t, max_t)
                buy_df = transactions[transactions['buyer_id'] == s_idx]  # all txns where s is buying
                for b_idx, buy_df_s in buy_df.groupby('product_id'):  # group by products bought
                    buy_ts = convert_txns_to_timeseries(buy_df, min_t, max_t)
                    best_corr, best_lag = get_best_corr_with_lag(buy_ts, supp_ts)
                    candidates[b_idx] = candidates.get(b_idx, []) + [best_corr]
            for b_idx, corrs in candidates.items():
                m[p_idx, b_idx] = np.mean(corrs)
    return m


def predict_product_relations_with_inventory_module(transactions, firms, products, prod2idx, prod_graph, 
                                                    debt_penalty=None, consumption_reward=None,
                                                    init_weights=None, num_epochs=50, patience=5, 
                                                    show_weights=True):
    """
    Train inventory module with direct attention on synthetic data, return final attention weights.
    """
    # initialize inventory module
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    module_args = {'num_firms': len(firms), 'num_prods': len(products), 'learn_att_direct': True,
                   'device': device}
    if debt_penalty is not None and consumption_reward is not None:
        module_args['debt_penalty'] = debt_penalty
        module_args['consumption_reward'] = consumption_reward
    if init_weights is not None:
        module_args['init_weights'] = init_weights
    module = TGNPLInventory(**module_args)                
    opt = torch.optim.Adam(module.parameters())
    
    # train inventory module
    losses = []
    maps = []
    min_t = transactions.time.min()
    max_t = transactions.time.max()
    for ep in range(num_epochs):
        for t in range(min_t, max_t+1):
            to_keep = transactions.time.values == t
            src = torch.Tensor(transactions.supplier_id.values[to_keep]).long().to(device)
            dst = torch.Tensor(transactions.buyer_id.values[to_keep]).long().to(device)
            prod = torch.Tensor(transactions.product_id.values[to_keep]).long().to(device) + len(firms)
            msg = torch.Tensor(transactions.amount.values[to_keep].reshape(-1, 1)).to(device)
            
            opt.zero_grad()
            loss, debt_loss, cons_loss = module(src, dst, prod, msg)
            loss.backward(retain_graph=False)
            opt.step()
            module.detach()
            losses.append((float(loss), float(debt_loss), float(cons_loss)))
            
        weights = module._get_prod_attention().cpu().detach().numpy()
        mean_avg_pr = mean_average_precision(prod_graph, prod2idx, weights)
        maps.append(mean_avg_pr)
        
        if show_weights and (ep % 5) == 0:
            pos = plt.imshow(weights)
            plt.colorbar(pos)
            plt.title(f'Ep {ep}')
            plt.show()
        module.reset()  # reset inventory
    return losses, maps, weights

    
def mean_average_precision(prod_graph, prod2idx, pred_mat, verbose=False):
    """
    Compute mean average precision over products, given a prediction matrix where rows are products
    and columns are their predicted inputs.
    """
    avg_prec_per_prod = []
    for p, p_idx in prod2idx.items():
        inputs = [prod2idx[s] for s in prod_graph[prod_graph['dest'] == p].source.values]  # true inputs
        is_consumer_prod = len(prod_graph[prod_graph['source'] == p]) == 0  
        if len(inputs) > 0:  # skip exogenous products since they have no inputs
            ranking = list(np.argsort(-pred_mat[p_idx]))
            total = 0
            for s_idx in inputs:
                k = ranking.index(s_idx)+1  # get the rank of input s_idx; rank and index are off-by-one
                prec_at_k = np.mean(np.isin(ranking[:k], inputs))  # how many of top k are true inputs
                total += prec_at_k
            avg_prec = total/len(inputs)
            if verbose:
                print(f'{p} (consumer prod: {is_consumer_prod}): avg precision={avg_prec:0.4f}')
            avg_prec_per_prod.append(avg_prec)
    return np.mean(avg_prec_per_prod)


def gridsearch_on_hyperparameters():
    """
    Gridsearch over scaling between debt penalty vs consumption reward.
    """
    with open('./synthetic_data.pkl', 'rb') as f:
        firms, products, prod_graph, firm2prods, prod2firms, inputs2supplier, demand_schedule = pickle.load(f)
    prod2idx = {p:i for i,p in enumerate(products)}
    
    transactions = pd.read_csv(f'./standard_setting_all_transactions.csv')
    print(f'Loaded all transactions -> {len(transactions)} transactions')
    corr_m = predict_product_relations_with_corr(transactions, products)
    corr_map = mean_average_precision(prod_graph, prod2idx, corr_m)
    print(f'Temporal correlations: MAP={corr_map:.4f}')
        
    debt_penalty = 10
    for scaling in np.arange(0.1, 1.1, 0.1):
        consumption_reward = debt_penalty * scaling
        losses, maps, inv_m = predict_product_relations_with_inventory_module(
            transactions, firms, products, prod2idx, prod_graph, init_weights=corr_m, 
            debt_penalty=debt_penalty, consumption_reward=consumption_reward, 
            show_weights=False)
        print(f'Debt penalty = {debt_penalty}, scaling = {scaling} -> final MAP={maps[-1]:0.4f}, best MAP={np.max(maps):0.4f}')
        print('Last 5 MAPs:', np.round(maps[-5:], 3))
        print()
            
    
def compare_methods_across_standard_data_settings(data_settings, num_rand_inits=10, 
                                                  debt_penalty=None, consumption_reward=None):
    """
    Evaluate production learning methods on standard synthetic data with different settings:
    varying amounts of missingness in transactions, varying amounts of missingness in firms.
    """
    with open('./synthetic_data.pkl', 'rb') as f:
        firms, products, prod_graph, firm2prods, prod2firms, inputs2supplier, demand_schedule = pickle.load(f)
    prod2idx = {p:i for i,p in enumerate(products)}
    
    print(f'For all experiments, using debt_penalty={debt_penalty} and consumption_reward={consumption_reward}')
    for setting in data_settings:
        transactions = pd.read_csv(f'./standard_setting_{setting}.csv')
        print(f'Setting: {setting} -> {len(transactions)} transactions')

        corr_m = predict_product_relations_with_corr(transactions, products)
        corr_map = mean_average_precision(prod_graph, prod2idx, corr_m)
        print(f'Temporal correlations: MAP={corr_map:.4f}')
        
        # when inventory module is randomly initialized
        final_maps = []
        for i in range(num_rand_inits):
            torch.manual_seed(i)
            losses, maps, inv_m = predict_product_relations_with_inventory_module(
                transactions, firms, products, prod2idx, prod_graph, 
                debt_penalty=debt_penalty, consumption_reward=consumption_reward,
                show_weights=False)
            print(i, f'Inventory module: final MAP={maps[-1]:0.4f}, best MAP={np.max(maps):0.4f}')
            final_maps.append(maps[-1])
        print(f'Inventory module: mean MAP={np.mean(final_maps):0.4f}, std MAP={np.std(final_maps):0.4f}')
        
        # no randomness, only need to run once
        losses, maps, inv_m = predict_product_relations_with_inventory_module(
                transactions, firms, products, prod2idx, prod_graph, init_weights=corr_m, 
                debt_penalty=debt_penalty, consumption_reward=consumption_reward, 
                show_weights=False)
        print(f'Inventory module, init with corr: final MAP={maps[-1]:0.4f}, best MAP={np.max(maps):0.4f}')

        
if __name__ == "__main__":    
    # summarize results with multiple runs
    settings = ['all_transactions', '08_transactions', '05_transactions', '09_firms', '07_firms']
    compare_methods_across_standard_data_settings(settings, debt_penalty=10, consumption_reward=1)
    # gridsearch_on_hyperparameters()