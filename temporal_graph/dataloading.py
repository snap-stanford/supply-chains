"""
This file is for alchemizing the .csv file of edges produced in extract_graph_data.py into Temporal
Graph data structures from the PyTorch Geometric library (see https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/signal.html)
"""

import datetime
import numpy as np
from torch_geometric_temporal import DynamicHeteroGraphTemporalSignal, DynamicGraphTemporalSignal
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix
import time
import torch
import os 

def get_days_between(date_1, date_2, date_format = "%Y-%m-%d"):
    """
    calculates the number of days between two dates (default format YY-MM-DD)
    """
    date_1_time = datetime.datetime.strptime(date_1, date_format)
    date_2_time = datetime.datetime.strptime(date_2, date_format)
    return (date_2_time - date_1_time).days

def get_forward_date(date_1, num_days_ahead, date_format = "%Y-%m-%d"):
    """
    retrieves the date (default format YY-MM-DD) <num_days_ahead> days ahead of date_1. If <num_days_ahead>
    is negative, then it will retrieve a date before date_1. 
    """
    date_2 = datetime.datetime.strptime(date_1, date_format) + datetime.timedelta(days = num_days_ahead)
    return date_2.strftime(date_format)

def node_normalize(temporal_node_data, norms = None, epsilon = 10**(-10)):
    """
    *temporal_node_data is of the shape [some number of timestamps, num_nodes, 2]
    or [num_nodes, 2]. Norms is a dictionary with keys {"mean","std"} and the 
    corresponding values each have shape [num_nodes, 2]
    """
    if (norms == None): return temporal_node_data
    #winsorize the data 

    return (temporal_node_data - norms["mean"]) / (norms["std"] + epsilon)
    #return (temporal_node_data - np.min(temporal_node_data, axis = (0,1))) / (np.max(temporal_node_data, axis = (0,1)) - np.min(temporal_node_data, axis = (0,1)))

def node_unnormalize(normalized_node_data, norms = None, epsilon = 10**(-10)):
    """ analagous parameter recommendations as normalize() above """
    if (norms == None): return normalized_node_data
    return normalized_node_data * (norms["std"] + epsilon) + norms["mean"]

class SupplyChainDataset(object):
    
    def __init__(self, csv_file, start_date, length_timestamps = 1, 
                 metric = "total_amount", edge_type = "material", lags = 5):
        """
        Constructor method for the temporal Supply Chain dataset, which formats the .csv from 
        extract_graph_data.py into graph-based data structures from torch_geometric_temporal
        
        Args:
            edges_csv_file (str): the path to the CSV file storing transaction edges
            start_date (str): start date from data was gathered (date corresponding to a time_stamp of 0)
            length_timestamps (int): number of days aggregated per time stamp
            metric (str): the metric (out of total_amount, total_quantity, bill_count, total_weight) 
                          to assign to each edge as its "weight"
            edge_type (str): Out of {'material', 'cash', 'both'}. By default, edges are drawn from supplier -> buyer (flow of 
            material). Reverse edges from buyer -> supplier (flow of cash) can also be drawn. 
                        
            lags (int): Number of past time_stamps to use for predicting the next time_stamp  

        Returns:
            None
        """
        all_metrics = {"total_amount", "total_quantity", "bill_count", "total_weight"}
        if metric not in all_metrics:
            raise ValueError(f"metric must be in {all_metrics}")
        self.start_date = start_date
        self.length_ts = length_timestamps
        if (edge_type not in ["material","cash","both"]):
            raise ValueError("edge_type must be in {'material','cash','both'}")
        self.edge_type = edge_type
        self.lags = lags
        
        #read the dataframe and excise missing supplier & buyer nodes, as well as values
        df = pd.read_csv(csv_file)
        row_names = ["time_stamp","supplier_t","buyer_t","hs6", metric]
        df = df[row_names][(~df[metric].isna()) & (df[metric] > 0) & (df["supplier_t"] != "") & (df["buyer_t"] != "")]
        companies = list(set(df["supplier_t"]).union(set(df["buyer_t"])))
        self.num_nodes, self.num_edges, self.num_products = len(companies), len(df), len(set(df["hs6"]))
        self.prevalent_timestamps = set(df["time_stamp"])
        self.max_ts = max(self.prevalent_timestamps)
        
        #establish node IDs by mapping firms to a unique indices, and obtain edge and node features
        self.company_to_nodeID = {company_t: node_id for node_id, company_t in enumerate(companies)}
        self.nodeID_to_company = {value:key for key,value in self.company_to_nodeID.items()}
        self.create_edge_attributes(df, metric)
        self.create_node_features(df, metric)
        
    def create_edge_attributes(self, df, metric):
        """procure edge information by using two parallel dictionaries to store edge indices and weights"""
        self.edge_index_dict = {} 
        self.edge_weight_dict = {}
        row_names = ["time_stamp","supplier_t","buyer_t","hs6", metric]
        rows = [list(df[name]) for name in row_names]
        for time_stamp, supplier, buyer, product, amount in zip(*rows):
            time_stamp = int(time_stamp)
            product = str(int(product)).zfill(6)
            
            #directed edge from supplier to buyer (flow of material)
            edge_material_flow = [self.company_to_nodeID[supplier], self.company_to_nodeID[buyer]]
            if (time_stamp not in self.edge_index_dict):
                self.edge_index_dict[time_stamp] = {product: [edge_material_flow]}
                self.edge_weight_dict[time_stamp] = {product: [amount]}   
            
            elif (product not in self.edge_index_dict[time_stamp]):
                self.edge_index_dict[time_stamp][product] = [edge_material_flow]
                self.edge_weight_dict[time_stamp][product] = [amount]
            
            else: 
                self.edge_index_dict[time_stamp][product].append(edge_material_flow)
                self.edge_weight_dict[time_stamp][product].append(amount)
        
    def create_node_features(self, df, metric):
        """get total input/output for each node at each timestep 
        input: supplier gets amount, buyer gets product, output: supplier loses amount, buyer loses product"""
        input_entity = "supplier_t" if metric == "total_amount" else "buyer_t" 
        output_entity = "buyer_t" if metric == "total_amount" else "supplier_t"
        
        df_node_input = df.groupby(by = ["time_stamp", input_entity]).sum(numeric_only = True).reset_index()
        df_node_output = df.groupby(by = ["time_stamp", output_entity]).sum(numeric_only = True).reset_index()
        node_input_rows = [list(df_node_input[row]) for row in ["time_stamp",input_entity,metric]]
        node_output_rows = [list(df_node_output[row]) for row in ["time_stamp",output_entity,metric]]
        
        self.node_features = np.zeros(shape = (self.max_ts + 1, self.num_nodes, 2), dtype = np.float32)
        for time_stamp, firm, metric in zip(*node_input_rows):
            self.node_features[int(time_stamp), self.company_to_nodeID[firm], 0] = metric
        for time_stamp, firm, metric in zip(*node_output_rows):
            self.node_features[int(time_stamp), self.company_to_nodeID[firm], 1] = metric
              
    def get_edge_index_dict(self, time_stamp):
        if (time_stamp not in self.prevalent_timestamps):
            return None 
        edges_dict = self.edge_index_dict[time_stamp]
        unique_products = list(edges_dict.keys())
        index_dict = {}
        #constructs forward edges from supplier to buyer (flow of material)
        if (self.edge_type == "material" or self.edge_type == "both"):
            index_dict.update({("firm",(product,"material"),"firm"): np.transpose(
                np.array(edges_dict[product])) for product in unique_products})
        
        #constructs reverse edges from buyer to supplier (flow of cash)
        if (self.edge_type == "cash" or self.edge_type == "both"):
            index_dict.update({("firm",(product, "cash"), "firm"): np.flip(np.transpose(
                np.array(edges_dict[product])), axis = 0).copy() for product in unique_products})
        return index_dict

    def get_edge_weight_dict(self, time_stamp):
        if (time_stamp not in self.prevalent_timestamps):
            return None
        weight_dict = self.edge_weight_dict[time_stamp]
        unique_products = list(weight_dict.keys()) #different products that delineate edge relations
        edge_weight_dict = {}
        #assigns weight attributes to forward edges from supplier to buyer (flow of material)
        if (self.edge_type == "material" or self.edge_type == "both"):
            edge_weight_dict.update({("firm",(product,"material"),"firm"): np.array(
                weight_dict[product]) for product in unique_products})
                                    
        #assigns weight attributes to reverse edges from buyer to supplier (flow of cash)
        if (self.edge_type == "cash" or self.edge_type == "both"):
            edge_weight_dict.update({("firm",(product, "cash"), "firm"): np.array(
                weight_dict[product]) for product in unique_products})
        return edge_weight_dict
    
    def get_date_range(self,time_stamp):
        lower_date = get_forward_date(self.start_date, self.length_ts * time_stamp)
        upper_date = get_forward_date(self.start_date, self.length_ts * (time_stamp + 1) - 1)
        return [lower_date, upper_date]
    
    def segment_timestamps(self, current_date, prior_days, next_days):
        """
        retrieve two lists of time_stamps corresponding to the <prior_days> days before <current_date>,  
        and the following <next_days> days of the dataset, respectively. 
        """
        days_after_start = get_days_between(self.start_date, current_date)
        prior_earliest_ts = np.ceil((days_after_start - prior_days + 1) / self.length_ts)
        prior_latest_ts = (days_after_start + 1) // self.length_ts - 1
        prior_timestamps = list(range(int(prior_earliest_ts), int(prior_latest_ts) + 1, 1))
        
        next_earliest_ts = np.ceil((days_after_start + 1) / self.length_ts)
        next_latest_ts = (days_after_start + 1 + next_days) // self.length_ts - 1
        next_timestamps = list(range(int(next_earliest_ts), int(next_latest_ts) + 1, 1))
        return prior_timestamps, next_timestamps
    
    def get_node_norms(self, time_stamps):
        """ obtain mean and standard deviation of raw node features over the time dimension """
        if (len(time_stamps) <= 1):
            return None 
        node_features = self.node_features[time_stamps]
        mean, std = np.mean(node_features), np.std(node_features)
        return {"mean": mean, "std": std}
    
    def getTemporalGraph(self, time_stamps, norms = None):
        """
        given a list of time_stamps, produces a PyG temporal graph covering time-stamped transaction edges 
        between node firms. Here, edge weights are the metric (e.g. total_amount, bill_count) given
        to the constructor method, node targets are the total input/output of each firm at each time. For now,
        node features are the concatenated input/output of the past <self.lags> days of each firm. 
        
        The edge relations are tuples of the form ("firm", (product, flow direction), "firm"), where flow 
        direction is "material" for supplier -> buyer and "cash" for buyer -> supplier. Date_ranges corresponds 
        to the interval of dates aggregated for each time_stamp. Use self.company_to_nodeID and 
        self.nodeID_to_company to convert between node indices and firm names.
        """
        date_ranges = [{"firm": np.array(self.get_date_range(ts))} for ts in time_stamps]
        edge_index_dicts = [self.get_edge_index_dict(ts - self.lags) for ts in time_stamps]
        edge_weight_dicts = [self.get_edge_weight_dict(ts - self.lags) for ts in time_stamps]
        
        feature_dicts = [{"firm": node_normalize(self.node_features[ts - self.lags: ts], norms) if (
            self.lags <= ts <= self.max_ts) else None} for ts in time_stamps]
        target_dicts = [{"firm": node_normalize(self.node_features[ts], norms) if (
            0 <= ts <= self.max_ts) else None} for ts in time_stamps]
        
        tempGraph = DynamicHeteroGraphTemporalSignal(edge_index_dicts = edge_index_dicts,
                                            edge_weight_dicts = edge_weight_dicts,
                                            feature_dicts = feature_dicts,
                                            target_dicts = target_dicts,
                                            date_ranges = date_ranges)
        return tempGraph
    
    def loadData(self, current_date, prior_days, next_days, normalize = True ):
        """
        This data function will provide two temporal graphs, the first as past input to a forecasting model,
        and the second as the ground truth corresponding to future predictions (see getTemporalGraph above for graph info)
        
        note: for well-defined behavior, you'll want the difference (in days) between self.start_date 
        and current_date to be one less than a multiple of self.length_ts (length of timestamps)
        i.e. -1 modulo self.length_ts. Same goes for the integer values of prior_days and next_days. 
        
        Args:
            current_date (str): The current_date from which we aim to predict the next days directly after
            prior_days (int): The number of past days to extract, including current_date  
            next_days (int): The number of days after current_date that we want to predict
        
        Returns:
            priorGraph (DynamicHeteroGraphTemporalSignal): a temporal Graph of product-stratified edges between firm nodes
                    for transactions occurring up to (prior_days - 1) days before the current_date (i.e. current_date is included)                       
            nextGraph (DynamicHeteroGraphTemporalSignal): a temporal Graph of product-stratified edges between firm nodes 
                    for transactions occurring up to next_days after the current_date (i.e. current_date is NOT included)
        """
        prior_timestamps, next_timestamps = self.segment_timestamps(current_date, prior_days, next_days)
        #set normalization values based on lagged prior time_stamps (no intersection with next days)
        lagged_priors = [ts - self.lags for ts in prior_timestamps if 0 <= ts - self.lags <= self.max_ts]
        norms = self.get_node_norms(lagged_priors) if normalize == True else None
        
        priorGraph = self.getTemporalGraph(prior_timestamps, norms)
        nextGraph = self.getTemporalGraph(next_timestamps, norms)
        return priorGraph, nextGraph, norms
    
#revise based on refactored class (use homogenuous supplier -> buyer edges)
class FirmDataset(SupplyChainDataset):
    def __init__(self, csv_file, start_date, length_timestamps = 1, 
                 metric = "total_amount", lags = 5):
        """
        same parameters as ancestor class (SupplyChainDataset). A modified version that aggregates
        transactions between firms over all products, i.e. no product stratification for edges. 
        """
        all_metrics = {"total_amount", "total_quantity", "bill_count", "total_weight"}
        if metric not in all_metrics:
            raise ValueError(f"Metric must be {all_metrics}")
        self.start_date = start_date
        self.length_ts = length_timestamps
        self.lags = lags
        #for edge_type, if metric == total_amount, then draw buyer -> supplier, otherwise supplier -> buyer 
        self.edge_type = "cash" if metric == "total_amount" else "material"
        
        #read the dataframe and excise missing supplier & buyer nodes, as well as values
        df = pd.read_csv(csv_file)
        row_names = ["time_stamp","supplier_t","buyer_t", metric]
        df = df[row_names][(~df[metric].isna()) & (df[metric] > 0) & (df["supplier_t"] != "") & (df["buyer_t"] != "")]
        
        #aggregate over all products 
        df = df.groupby(by = ["time_stamp","supplier_t","buyer_t"]).sum().reset_index()
        companies = list(set(df["supplier_t"]).union(set(df["buyer_t"])))
        self.num_nodes = len(companies)
        self.prevalent_timestamps = set(df["time_stamp"])
        self.max_ts = max(self.prevalent_timestamps)
        self.company_to_nodeID = {company_t: node_id for node_id, company_t in enumerate(companies)}
        self.nodeID_to_company = {value:key for key,value in self.company_to_nodeID.items()}
        
        self.create_edge_attributes(df, metric)
        self.create_node_features(df, metric)
        self.node_features = np.log(self.node_features + 1)
        self.edge_weight_dict = {key: np.log(np.array(value) + 1) for key, value in self.edge_weight_dict.items()}
    
    def create_edge_attributes(self, df, metric):
        self.edge_index_dict = {}
        self.edge_weight_dict = {}
        row_names = ["time_stamp","supplier_t","buyer_t", metric]
        rows = [list(df[name]) for name in row_names]
        for time_stamp, supplier, buyer, amount in zip(*rows):
            time_stamp = int(time_stamp)
            if (self.edge_type == "cash"):
                edge_transactions = [self.company_to_nodeID[buyer], self.company_to_nodeID[supplier]] 
            else:
                edge_transactions = [self.company_to_nodeID[supplier], self.company_to_nodeID[buyer]]
            if (time_stamp not in self.edge_index_dict):
                self.edge_index_dict[time_stamp] = [edge_transactions]
                self.edge_weight_dict[time_stamp] = [amount]
            else:
                self.edge_index_dict[time_stamp].append(edge_transactions)
                self.edge_weight_dict[time_stamp].append(amount)
            
    #override ancestor class to use homogeneous edges
    def get_edge_index_dict(self, time_stamp):
        if (time_stamp not in self.prevalent_timestamps):
            return None
        return np.array(self.edge_index_dict[time_stamp]).T
        
    #override ancestor class to use homogeneous edges
    def get_edge_weight_dict(self, time_stamp):
        if (time_stamp not in self.prevalent_timestamps):
            return None
        return np.array(self.edge_weight_dict[time_stamp])
    
    #override ancestor class to use DynamicGraphTemporalSignal
    def getTemporalGraph(self, time_stamps, norms = None):
        date_ranges = [np.array(self.get_date_range(ts)) for ts in time_stamps]
        edge_indices = [self.get_edge_index_dict(ts - self.lags) for ts in time_stamps]
        edge_weights = [self.get_edge_weight_dict(ts - self.lags) for ts in time_stamps]
        features = [node_normalize(self.node_features[ts - self.lags: ts,:], norms).transpose(1,0,2).reshape(-1,10) if (
            self.lags <= ts <= self.max_ts) else None for ts in time_stamps]
        targets = [node_normalize(self.node_features[ts, :], norms) if (
            0 <= ts <= self.max_ts) else None for ts in time_stamps]
         
        tempGraph = DynamicGraphTemporalSignal(edge_indices = edge_indices,
                                               edge_weights = edge_weights,
                                               features = features, 
                                               targets = targets,
                                              date_ranges = date_ranges)
        return tempGraph
    
if __name__ == "__main__":
    """command line testing"""
    start_t = time.time()
    fname = "./storage/daily_transactions_2021.csv"
    sc = SupplyChainDataset(fname, start_date = "2021-01-01", edge_type = "material", lags = 5)
    end_t = time.time()
    print(f"Processed dataset at {fname} in {end_t - start_t:.2f} seconds.")
    print(f"Total Firms: {sc.num_nodes}\nTotal Time-Stamped Supplier \u2192 Buyer Edges: {sc.num_edges}")
    priorGraph, nextGraph, norms = sc.loadData(current_date = "2021-03-10", prior_days = 50, next_days = 10)

    for timestep, snapshot in enumerate(priorGraph):
        #print(snapshot["firm"].x.shape)
        snapshot_map = snapshot.to_dict()
        for key in list(snapshot_map.keys())[:4]:
            print(key, {label: feature if label not in ["x","y"] else feature.shape for label, feature in snapshot_map[key].items()})
            print()
        break