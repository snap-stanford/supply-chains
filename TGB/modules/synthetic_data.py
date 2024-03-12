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
    

def generate_initial_conditions(firms, products, firm2idx, prod2idx, prod_graph, prod2firms, 
                                init_inv=0, init_supply=100, init_demand=1):
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
    curr_orders = np.zeros((len(firms)+1, len(firms), len(products)))  # buyer, supplier, product; entry is amount
    consumer_prods = set(prod_graph.dest.values) - set(prod_graph.source.values)  # consumer products, not used as input
    assert sorted(consumer_prods) == sorted(products[-len(consumer_prods):])
    for p in consumer_prods:
        for f in prod2firms[p]:
            curr_orders[len(firms), firm2idx[f], prod2idx[p]] = init_demand  # start with steady demand for consumer products
            
    return inventories, curr_orders, exog_supp
    
    
def generate_demand_schedule(num_timesteps, prod_graph, prod2firms, seed=0):
    """
    Generate demand schedule for consumer products.
    """
    np.random.seed(seed)
    consumer_prods = set(prod_graph.dest.values) - set(prod_graph.source.values)  # consumer products, not used as input
    prod_types = []
    num_weeks = np.ceil(num_timesteps / 7)
    demand_schedule = {}  # (firm, product, time) -> consumer demand
    for i, p in enumerate(consumer_prods):
        # get total demand per t
        prod_type = ['weekend', 'weekday', 'uniform'][i % 3]  # iterate through these options
        if prod_type == 'weekend':
            demand = np.repeat([10, 10, 2, 2, 2, 2, 2], num_weeks)[:num_timesteps]
        elif prod_type == 'weekday':
            demand = np.repeat([2, 2, 10, 10, 10, 10, 10], num_weeks)[:num_timesteps]
        else:
            demand = np.ones(num_timesteps) * 5
        prod_types.append(prod_type)
        
        # assign demand to suppliers
        supplier_firms = prod2firms[p]
        pvals = np.ones(len(supplier_firms)) / len(supplier_firms)  # each supplier is equally likely
        for t, d in enumerate(demand):
            counts = np.random.multinomial(d, pvals)
            for s, ct in zip(supplier_firms, counts):
                demand_schedule[(s, p, t)] = ct
    print('Finished generating demand schedule:', Counter(prod_types))
    return demand_schedule
    
    
def generate_transactions(num_timesteps, inventories, curr_orders, exog_supp,  # time-varying info
                          firms, products, firm2idx, prod2idx,  # nodes + indexing
                          prod_graph, firm2prods, prod2firms, inputs2supplier,  # static graphs
                          demand_schedule=None, num_decimals=5, use_random_firm_order=False,
                          seed=0, debug=False):
    """
    Generate transactions using agent-based model to determine firm's actions per timestep.
    """
    np.random.seed(seed)
    all_transactions = []
    consumer_prods = set(prod_graph.dest.values) - set(prod_graph.source.values)  # consumer products, not used as input
    for t in range(num_timesteps):
        transactions_t = []
        future_inputs_needed = np.zeros(inventories.shape)  # n_firms x n_products
        # should be order-invariant: a firm's actions at time t shouldn't be affected by other firms' actions
        # firm's actions depend on its inventory and orders *to* the firm
        if use_random_firm_order:  # to test order invariance, try this
            rand_idx = np.random.choice(len(firms), replace=False, size=len(firms))
            firm_order = np.array(firms)[rand_idx]
            print(firm_order[:10])
        else:
            firm_order = firms
        for f in firm_order:
            inventories, curr_orders, transactions_completed, inputs_needed = simulate_actions_for_firm(
                f, inventories, curr_orders, exog_supp, firms, products, firm2idx, prod2idx, prod_graph, firm2prods, 
                prod2firms, inputs2supplier, debug=debug)
            transactions_t += transactions_completed
            future_inputs_needed[firm2idx[f]] = inputs_needed
            
        # update firms' inventories (add to buyers) based on completed transactions, record transactions
        if len(transactions_t) > 0:
            s_idxs, b_idxs, p_idxs, amts = list(zip(*transactions_t))
            inventories += csr_matrix((amts, (b_idxs, p_idxs)), shape=inventories.shape).toarray()  # add to buyers' inventories
            transactions_df = pd.DataFrame(transactions_t, columns=['supplier_id', 'buyer_id', 'product_id', 'amount'])
            transactions_df['time'] = t
            all_transactions.append(transactions_df)
        
        # make new orders based on inventories and future inputs needed
        future_inputs_needed = np.clip(future_inputs_needed - inventories, 0, None)  # what is still needed
        b_idxs, p_idxs = np.nonzero(future_inputs_needed)
        for b_idx, p_idx in zip(b_idxs, p_idxs):
            s_idx = firm2idx[inputs2supplier[(firms[b_idx], products[p_idx])]]
            curr_orders[b_idx, s_idx, p_idx] = max(curr_orders[b_idx, s_idx, p_idx], future_inputs_needed[b_idx, p_idx])
        
        # add new demand from consumers
        if demand_schedule is not None:
            for p in consumer_prods:
                for f in prod2firms[p]:
                    demand = demand_schedule[t] if t in demand_schedule else demand_schedule[(f, p, t)]
                    curr_orders[len(firms), firm2idx[f], prod2idx[p]] += demand
        
        # avoid rounding errors
        inventories = np.round(inventories, num_decimals)
        curr_orders = np.round(curr_orders, num_decimals)
        num_orders = len(np.nonzero(curr_orders)[0])
        print(f't={t}: generated {len(transactions_t)} transactions, {num_orders} current orders')
        # print_curr_orders(curr_orders, firms, products)
        if num_orders == 0:
            print('No remaining orders, ending simulation')
            break
        
    all_transactions = pd.concat(all_transactions)
    return all_transactions
        

def simulate_actions_for_firm(f, inventories, curr_orders, exog_supp,  # time-varying info
                              firms, products, firm2idx, prod2idx,  # nodes + indexing
                              prod_graph, firm2prods, prod2firms, inputs2supplier,  # static graphs
                              debug=False):
    """
    Simulate actions for a given firm f following agent-based model. No need for seed because deterministic.
    Inputs:
        f: name of firm
        inventories: firm x product, shape: (n_firms x n_products)
        curr_orders: buyer x supplier x product, shape: (n_firms+1 x n_firms x n_products); 
                     the +1 is to represent consumer as buyer
        exog_supp: current supply per (firm, product) for all exogenous products
    
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
    inputs_needed = np.zeros(len(products))
    f_idx = firm2idx[f]
    for p in firm2prods[f]:  # use order from earlier product sampling; order matters bc inventory is changed
        if debug: 
            print('Processing', p)
        p_idx = prod2idx[p]
        if np.sum(curr_orders[:, f_idx, p_idx]) == 0:
            if debug:
                print('No orders for firm+product, skipping')
        else:
            if debug:
                print('Before processing...')
                print_firm_and_product_status(f_idx, p_idx, inventories, curr_orders, inputs_needed, firms, products)
            
            # get maximum amount of p that f could make
            inputs_p = prod_graph[prod_graph['dest'] == p]
            input2units = dict(zip(inputs_p.source, inputs_p.units))
            if debug: 
                print('inputs for product', input2units)
            if len(input2units) == 0:  # exogenous
                p_max = exog_supp[(f, p)]
            else:
                p_max = []
                for s, units in input2units.items():
                    p_max.append(inventories[f_idx, prod2idx[s]]/units)
                    if debug: 
                        print(f'Based on {s}, could make {p_max[-1]:.2f} of {p}')
                p_max = np.min(p_max)

            # produce output, subtract from inventory
            fp_orders = curr_orders[:, f_idx, p_idx]
            p_ordered = np.sum(fp_orders)  # total amount ordered
            p_out = min(p_ordered, p_max)  # min of amount ordered and max amount f could produce
            if debug: 
                print(f'Amount ordered: {p_ordered:.2f}; Amount made: {p_out:.2f}')
            for s, units in input2units.items():
                inventories[f_idx, prod2idx[s]] -= p_out * units

            # allocate output to buyers, subtract from orders to f
            allocated_per_buyer = p_out * (fp_orders / np.sum(fp_orders))  # proportional
            for b_idx in np.nonzero(allocated_per_buyer)[0]:
                if b_idx < len(firms):  # otherwise, indicates consumer demand
                    transactions_completed.append((f_idx, b_idx, p_idx, allocated_per_buyer[b_idx]))
                curr_orders[b_idx, f_idx, p_idx] -= allocated_per_buyer[b_idx]

            # record what inputs are still needed for unfulfilled orders
            remaining = np.sum(curr_orders[:, f_idx, p_idx])
            for s, units in input2units.items():
                inputs_needed[prod2idx[s]] += remaining * units
                
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
    
    orders_to_f = curr_orders[:, f_idx, p_idx]  # length n_firms+1
    dict_orders_to_f = {}
    for b_idx in np.nonzero(orders_to_f)[0]:
        buyer = firms[b_idx] if b_idx < len(firms) else 'consumer'
        dict_orders_to_f[buyer] = orders_to_f[b_idx]
    print('orders for firm+product', dict_orders_to_f)
    
    dict_inputs_needed = {products[s_idx]:inputs_needed[s_idx] for s_idx in np.nonzero(inputs_needed)[0]}
    print('inputs needed by firm', dict_inputs_needed)
    
    
def print_curr_orders(curr_orders, firms, products, max_print=10):
    """
    Print current orders. Used for debugging.
    """
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
# Functions to evaluate against ground-truth
# production functions
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
                                                    init_m=None, num_epochs=50, visualize_weights=True):
    """
    Train inventory module with direct attention on synthetic data, return learned 
    """
    # initialize inventory module
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    module = TGNPLInventory(len(firms), len(products), learn_att_direct=True, device=device)
    if init_m is not None:  # set initial weights of inventory module
        assert init_m.shape == (len(products), len(products))
        module.att_weights.data = torch.Tensor(init_m).to(device)
        
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
            losses.append(float(loss))
            
        weights = module._get_prod_attention().cpu().detach().numpy()
        mean_avg_pr = mean_average_precision(prod_graph, prod2idx, weights, verbose=False)
        maps.append(mean_avg_pr)

        if visualize_weights and (ep % 5) == 0:
            pos = plt.imshow(weights)
            plt.colorbar(pos)
            plt.title('Ep %d' % ep)
            plt.show()
        module.reset()  # reset inventory
    return losses, maps, weights

    
def mean_average_precision(prod_graph, prod2idx, pred_mat, verbose=True):
    """
    Compute mean average precision over products.
    """
    avg_prec_per_prod = []
    for p, p_idx in prod2idx.items():
        inputs = [prod2idx[s] for s in prod_graph[prod_graph['dest'] == p].source.values]
        is_consumer_prod = len(prod_graph[prod_graph['source'] == p]) == 0
        if len(inputs) > 0:
            ranking = list(np.argsort(-pred_mat[p_idx]))
            total = 0
            for s_idx in inputs:
                k = ranking.index(s_idx)+1  # rank and index are off-by-one
                prec_at_k = np.mean(np.isin(ranking[:k], inputs))  # how many of top k are true inputs
                total += prec_at_k
            avg_prec = total/len(inputs)
            if verbose:
                print(f'{p} (consumer prod: {is_consumer_prod}): avg precision={avg_prec:0.4f}')
            avg_prec_per_prod.append(avg_prec)
    return np.mean(avg_prec_per_prod)


if __name__ == "__main__":
    with open('./synthetic_data.pkl', 'rb') as f:
        firms, products, prod_graph, firm2prods, prod2firms, inputs2supplier, demand_schedule = pickle.load(f)
    prod2idx = {p:i for i,p in enumerate(products)}
    
    # summarize results with multiple runs
    settings = ['all_transactions', '08_transactions', '05_transactions', '09_firms', '07_firms']
    for setting in settings:
        transactions = pd.read_csv(f'./standard_setting_{setting}.csv')
        print(f'Setting: {setting} -> {len(transactions)} transactions')

        corr_m = predict_product_relations_with_corr(transactions, products)
        corr_map = mean_average_precision(prod_graph, prod2idx, corr_m, verbose=False)
        print(f'Temporal correlations: MAP={corr_map:.4f}')
        
        final_maps = []
        for i in range(10):
            torch.manual_seed(i)
            losses, maps, inv_m = predict_product_relations_with_inventory_module(
                transactions, firms, products, prod2idx, prod_graph, visualize_weights=False)
            print(i, f'Inventory module: final MAP={maps[-1]:0.4f}, best MAP={np.max(maps):0.4f}')
            final_maps.append(maps[-1])
        print(f'Inventory module: mean MAP={np.mean(final_maps):0.4f}, std MAP={np.std(final_maps):0.4f}')
        
        # no randomness, only need to run once
        losses, maps, inv_m = predict_product_relations_with_inventory_module(
                transactions, firms, products, prod2idx, prod_graph, init_m=corr_m, visualize_weights=False)
        print(i, f'Inventory module, init with corr: final MAP={maps[-1]:0.4f}, best MAP={np.max(maps):0.4f}')