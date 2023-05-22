from constants_and_utils import *
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


class MultiTierSC():
    """
    A class to construct a multi-tier supply chain.
    """
    def __init__(self):
        # index contains all supplier, buyer, hs_code triplets in the data (since Jan 2019)
        self.index = pd.read_csv("s3://supply-web-data-storage/CSV/index_hs6.csv")
        self.all_companies = set(self.index.buyer_t.unique()).union(self.index.supplier_t.unique())
        
    def get_supply_chain(self, hs_codes, tiers, as_nx=False, nx_kwargs=None):
        """
        Constructs supply chain started at hs_codes. 
        Returns companies, products, supplier-product edges, and product-buyer edges.
        Process:
        1. Find the suppliers up to the desired tier.
        2. Include ALL edges and products adjacent to these suppliers.
        """
        assert tiers >= 1
        # Find tier 1 suppliers
        prod_df = pd.DataFrame()
        for hs6 in hs_codes:
            prod_df = pd.concat([prod_df, self.index[self.index['hs6'] == hs6]], axis=0)
        prod_df = prod_df.loc[:,['supplier_t','hs6','bill_count']]
        suppliers = prod_df[prod_df.supplier_t.str.len() > 0].supplier_t.unique()
        all_suppliers = set(suppliers)
        
        # Find suppliers above tier 1
        new_suppliers = set(suppliers)
        for t in range(1, tiers):
            df = self.index[self.index['buyer_t'].isin(new_suppliers)].copy()
            new_suppliers = set(df[df.supplier_t.str.len() > 0].supplier_t.unique())
            new_suppliers = new_suppliers - all_suppliers
            all_suppliers = all_suppliers.union(new_suppliers)
        
        # Find supplier-product edges
        prod_df = self.index[self.index['supplier_t'].isin(all_suppliers)].copy()
        prod_df = prod_df.loc[:,['supplier_t','hs6','bill_count']]
        assert all(prod_df.supplier_t.str.len() > 0)
        prod_df = prod_df[prod_df.hs6.str.len() > 0].reset_index(drop=True)
        
        # Find product-buyer edges
        proc_df = self.index[self.index['buyer_t'].isin(all_suppliers)].copy()
        proc_df = proc_df.loc[:,['hs6','buyer_t','bill_count']]
        assert all(proc_df.buyer_t.str.len() > 0)
        proc_df = proc_df[proc_df.hs6.str.len() > 0].reset_index(drop=True)
        
        found_companies = set(prod_df.supplier_t.unique()).union(set(proc_df.buyer_t.unique()))
        found_products = set(prod_df.hs6.unique()).union(set(proc_df.hs6.unique()))
        print('Found %d companies overall, %d companies with edges, %d products with edges, %d edges' % (
            len(all_suppliers), len(found_companies), len(found_products), len(prod_df) + len(proc_df)))
        
        # Return as networkx graph
        if as_nx:
            if nx_kwargs is None:
                nx_kwargs = {}
            G = make_networkx_graph_from_edges(prod_df, proc_df, **nx_kwargs)
            assert len(G.nodes) == (len(found_companies) + len(found_products))
            return G
        return found_companies, found_products, prod_df, proc_df
                
    def compute_pmis(self, max_degree=1):
        """
        Computes pointwise mutual information (PMI) for each pair of (product_sold, product_bought).
        """
        # get pairs of supplier, product
        supplier_pairs = self.index.groupby(['supplier_t', 'hs6']).size().reset_index().rename(columns={'supplier_t': 'company', 'hs6': 'product_sold'})
        # get pairs of buyer, product
        buyer_pairs = self.index.groupby(['buyer_t', 'hs6']).size().reset_index().rename(columns={'buyer_t': 'company', 'hs6': 'product_bought'})
        # get triplets of company, product sold, product bought
        triplets = pd.merge(supplier_pairs[['company', 'product_sold']], buyer_pairs[['company', 'product_bought']], how='inner', 
                            left_on='company', right_on='company')
        test_company = triplets.iloc[0].company
        num_products_sold = len(supplier_pairs[supplier_pairs.company == test_company])
        num_products_bought = len(buyer_pairs[buyer_pairs.company == test_company])
        assert (num_products_sold * num_products_bought) == len(triplets[triplets.company == test_company])
        # compute p(sell product A and buy product B)
        product_pairs = triplets.groupby(['product_sold', 'product_bought']).size().rename('num_product_pair').reset_index()
        product_pairs['p_product_pair'] = product_pairs.num_product_pair / len(self.all_companies)
        
        # compute p(sell product) and p(buy product)
        num_suppliers_and_buyers = self.index.groupby('hs6')[['supplier_t', 'buyer_t']].nunique()
        prob_supply_and_buy = (num_suppliers_and_buyers / len(self.all_companies)).rename(columns={'supplier_t': 'p_product_sold', 'buyer_t': 'p_product_bought'})
        product_pairs = pd.merge(product_pairs, prob_supply_and_buy.p_product_sold, how='left', left_on='product_sold', right_index=True)
        product_pairs = pd.merge(product_pairs, prob_supply_and_buy.p_product_bought, how='left', left_on='product_bought', right_index=True)
        product_pairs['p_multiplied'] = product_pairs['p_product_sold'] * product_pairs['p_product_bought']
        product_pairs['pmi'] = np.log(product_pairs['p_product_pair'] / product_pairs['p_multiplied'])
        for k in range(2, max_degree+1):
            # pmi^k is known to alleviate issues with low frequency
            product_pairs[f'pmi^{k}'] = np.log((product_pairs['p_product_pair']**k) / product_pairs['p_multiplied'])
        return product_pairs
    
    
def make_networkx_graph_from_edges(prod_df, proc_df, directed=False, flip_direction=False):
    """
    Convert dataframes of supplier-product and product-buyer to networkx graph.
    """
    prod_df['supplier_t'] = prod_df.supplier_t.apply(lambda x: '%s_COMPANY' % x)  # tag as company
    prod_df['hs6'] = prod_df.hs6.apply(lambda x: '%s_PRODUCT' % x)  # tag as product
    prod_df.columns = ['src', 'dst', 'edge_attr']
    proc_df['hs6'] = proc_df.hs6.apply(lambda x: '%s_PRODUCT' % x)
    proc_df['buyer_t'] = proc_df.buyer_t.apply(lambda x: '%s_COMPANY' % x)
    proc_df.columns = ['src', 'dst', 'edge_attr']
    all_edges = pd.concat([prod_df, proc_df], axis=0).reset_index(drop=True)
    if directed:
        if flip_direction:
            # product -> supplier, buyer -> product (better for predicting product->parts)
            G = nx.from_pandas_edgelist(all_edges, source='dst', target='src', edge_attr='edge_attr',
                                        create_using=nx.DiGraph)
        else:
            # product -> buyer, supplier -> product (better for predicting parts->product)
            G = nx.from_pandas_edgelist(all_edges, source='src', target='dst', edge_attr='edge_attr',
                                        create_using=nx.DiGraph)
    else:
        G = nx.from_pandas_edgelist(all_edges, source='src', target='dst', edge_attr='edge_attr')
    return G

    
def get_pagerank_ordering(G, kwargs=None):
    """
    Runs Pagerank on the graph, using the provided kwargs (eg, personalization).
    Returns a dataframe of nodes in the graph ordered by PageRank score.
    Assumes that nodes in graph are represented as <node_name>_<node_type>, where <node_name> is 
    either the company's name or product's HS code and node_type is either "PRODUCT" or "COMPANY".
    See make_networkx_graph_from_edges for how G is constructed.
    """
    if kwargs is None:
        kwargs = {}
    pr = nx.pagerank(G, **kwargs)
    pr_df = []
    for n, score in pr.items():
        node_name, node_type = n.rsplit('_', 1)
        if node_type == 'PRODUCT':
            label = 1 if node_name in BATTERY_PARTS else 0
        else:
            assert node_type == 'COMPANY'
            label = -1
        pr_df.append({'node_name': node_name, 'node_type': node_type, 'label': label, 'pr': score})
    pr_df = pd.DataFrame(pr_df, columns=list(pr_df[-1].keys())).sort_values('pr', ascending=False)
    pr_df['rank'] = np.arange(len(pr_df))
    product_pr_df = pr_df[pr_df.node_type == 'PRODUCT']
    parts_found = product_pr_df[product_pr_df.node_name.isin(BATTERY_PARTS)]
    if len(parts_found) < len(BATTERY_PARTS):
        print('Warning: only found %d out of %d parts in graph' % (len(parts_found), len(BATTERY_PARTS)))
    return pr_df

    
def precision_and_recall_at_k(ordered_items, positives, k):
    """
    Computes precision@k and recall@k.
    """
    top_k = ordered_items[:k]
    num_positive = np.sum(np.isin(top_k, positives))
    precision = num_positive / k
    recall = num_positive / len(positives)
    return precision, recall


def compare_orderings_across_k(orderings, labels, title=''):
    """
    Visualizes precision, recall, and F1 @ k based on BATTERY_PARTS for different 
    product orderings. 
    Assumes orderings is a dict mapping from label to ordering.
    """
    assert all([l in orderings for l in labels])
    num_products = len(orderings[labels[0]])
    assert all([len(orderings[l]) == num_products for l in labels])
    ks = range(10, len(BATTERY_PARTS)+1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.subplots_adjust(wspace=0.3)
    fig.suptitle(title, fontsize=12)
    for label in labels:
        ordering = orderings[label]
        prec_vec = []
        rec_vec = []
        for k in ks:
            prec, rec = precision_and_recall_at_k(ordering, BATTERY_PARTS, k)
            prec_vec.append(prec)
            rec_vec.append(rec)
        prec_vec, rec_vec = np.array(prec_vec), np.array(rec_vec)
        axes[0].plot(ks, prec_vec, label=label)
        axes[1].plot(ks, rec_vec)
        f1_vec = 2 * (prec_vec * rec_vec) / (prec_vec + rec_vec)
        axes[2].plot(ks, f1_vec)

    axes[0].set_ylabel('precision @ k', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[1].set_ylabel('recall @ k', fontsize=12)
    axes[2].set_ylabel('F1 @ k', fontsize=12)
    for ax in axes:
        ax.set_xlabel('k', fontsize=12)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=12)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(0, ymax)
    plt.show()
    
    
def get_deduped_records_for_company_and_hs(companies, hs_code, rs, mode, by_date=True):
    """
    Get deduped records using primary key for a list of companies (either supplier or buyer)
    and hs_code.
    """
    assert mode in {'supplier', 'buyer'}
    company_str = f"{mode}_t like '{companies[0]}'" + "".join([f" or {mode}_t like '{company}'" for company in companies[1:]])
    query = f"select {PRIMARY_KEY}, COUNT(*) as count, COUNT(DISTINCT id) as num_ids from logistic_data where hs_code like '{hs_code}%' and ({company_str}) GROUP BY {PRIMARY_KEY}"
    if by_date:  # group by date
        query = f"select date, COUNT(*) as bill_count, SUM(quantity) as total_quantity, SUM(amount) as total_amount, SUM(weight) as total_weight from ({query}) GROUP BY date"
    print(query)
    
    df = rs.query_df(query)
    assert len(df) == len(df.drop_duplicates())
    print('Found %d rows' % len(df))
    df['month'] = df.date.apply(lambda x: str(x).rsplit('-', 1)[0])
    df['datetime'] = pd.to_datetime(df.date)
    df['month_datetime'] = pd.to_datetime(df.month)
    df = df.sort_values('datetime').set_index('datetime')
    return df