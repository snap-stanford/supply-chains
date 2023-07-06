"""
This file is for alchemizing the .csv file of edges produced in extract_graph_data.py into Temporal
Graph data structures from the PyTorch Geometric library (see https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/signal.html)
"""

import datetime
import numpy as np
from torch_geometric_temporal import DynamicHeteroGraphTemporalSignal
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix

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
    
class SupplyChainDataset(object):
    
    def __init__(self, edges_csv_file, start_date, length_timestamps = 1, metric = "total_amount"):
        """
        Constructor method for the temporal Supply Chain dataset, which formats the .csv from 
        extract_graph_data.py into graph-based data structures from torch_geometric_temporal
        
        Args:
            edges_csv_file (str): the path to the CSV file storing transaction edges
            start_date (str): start date from data was gathered (date corresponding to a time_stamp of 0)
            length_timestamps (int): number of days aggregated per time stamp
            metric (str): the metric (out of total_amount, total_quantity, bill_count, total_weight) 
                          to assign to each edge as its "weight"
        
        Returns:
            None
        """
        all_metrics = {"total_amount", "total_quantity", "bill_count", "total_weight"}
        if metric not in all_metrics:
            raise ValueError(f"Metric must be {all_metrics}")
        self.start_date = start_date
        self.length_ts = length_timestamps
            
        #read the dataframe and excise missing supplier & buyer nodes, as well as values
        df = pd.read_csv(edges_csv_file)
        row_names = ["time_stamp","supplier_t","buyer_t","hs6", metric]
        df = df[row_names][(~df[metric].isna()) & (df[metric] > 0) & (df["supplier_t"] != "") & (df["buyer_t"] != "")]
        self.num_edges_total = len(df)
        companies = list(set(df["supplier_t"]).union(set(df["buyer_t"])))
        self.num_nodes_total = len(companies)
        
        #establish nodes (using an ID dictionary) and heterogeneous edge information (using two parallel
        #dictionaries storing the edge indices and metric / weights, respectively)
        self.company_to_nodeID = {company_t: node_id for node_id, company_t in enumerate(companies)}
        self.nodeID_to_company = {value:key for key,value in self.company_to_nodeID.items()}
        self.edge_index_dict = {} 
        self.edge_weight_dict = {}
        rows = [list(df[name]) for name in row_names] #for index, row in tqdm(df.iterrows()): 
        for time_stamp, supplier, buyer, product, amount in zip(*rows):
            time_stamp = int(time_stamp)
            product = str(int(product)).zfill(6)
            #directed edge from supplier to buyer (flow of material), will create the
            #edge from buyer to supplier (flow of cash) on the spot in getTemporalGraph()
            material_edge = [self.company_to_nodeID[supplier], self.company_to_nodeID[buyer]]
            if (time_stamp not in self.edge_index_dict):
                self.edge_index_dict[time_stamp] = {product: [material_edge]}
                self.edge_weight_dict[time_stamp] = {product: [amount]}
                
            elif (product not in self.edge_index_dict[time_stamp]):
                self.edge_index_dict[time_stamp][product] = [material_edge]
                self.edge_weight_dict[time_stamp][product] = [amount]
                
            else: 
                self.edge_index_dict[time_stamp][product].append(material_edge)
                self.edge_weight_dict[time_stamp][product].append(amount)
        
        self.prevalent_timestamps = set(self.edge_index_dict.keys()) 
        
        #get total input/output for each node at each timestep 
        #input: supplier gets amount, buyer gets product, output: supplier loses amount, buyer loses product
        input_entity = "supplier_t" if metric == "total_amount" else "buyer_t" 
        output_entity = "buyer_t" if metric == "total_amount" else "supplier_t"
        df_node_input = df.groupby(by = ["time_stamp", input_entity]).sum(numeric_only = True).reset_index()
        df_node_output = df.groupby(by = ["time_stamp", output_entity]).sum(numeric_only = True).reset_index()
        
        node_input_rows = [list(df_node_input[row]) for row in ["time_stamp",input_entity,metric]]
        node_output_rows = [list(df_node_output[row]) for row in ["time_stamp",output_entity,metric]]
        self.node_targets = {timestamp: np.zeros((self.num_nodes_total, 2)) for timestamp in self.prevalent_timestamps}
        for time_stamp, firm, metric in zip(*node_input_rows):
            self.node_targets[time_stamp][self.company_to_nodeID[firm],0] = metric 
        for time_stamp, firm, metric in zip(*node_output_rows):
            self.node_targets[time_stamp][self.company_to_nodeID[firm],1] = metric 
        self.node_targets = {timestamp: csr_matrix(self.node_targets[timestamp]) for timestamp in self.prevalent_timestamps}
        
    def get_edge_index_dict(self,time_stamp):
        if (time_stamp not in self.prevalent_timestamps):
            return {"firm": None}
        edges_dict = self.edge_index_dict[time_stamp]
        unique_products = list(edges_dict.keys()) #different edge relation types
        return {("firm",product,"firm"): np.transpose(np.array(edges_dict[product])) for product in unique_products}
    
    def get_edge_weight_dict(self,time_stamp):
        if (time_stamp not in self.prevalent_timestamps):
            return {"firm": None}
        weight_dict = self.edge_weight_dict[time_stamp]
        unique_products = list(weight_dict.keys()) #different edge relation types
        return {("firm",product,"firm"): np.array(weight_dict[product]) for product in unique_products}
    
    def get_date_range(self,time_stamp):
        lower_date = get_forward_date(self.start_date, time_stamp - self.length_ts)
        upper_date = get_forward_date(self.start_date, time_stamp - 1)
        return [lower_date, upper_date]
    
    def getTemporalGraph(self, time_stamps):
        """
        given a list of time_stamps, produces a PyG temporal graph covering time-stamped transaction edges 
        between node firms. Here, edge weights are the metric (e.g. total_amount, bill_count) given
        to the constructor method, and node targets are the total input/output of each firm at each time. 
        The different edge relations correspond to different HS6 products (see get_edge_index_dict above),
        and date_ranges corresponds to the interval of dates aggregated for each time_stamp. Use 
        self.company_to_nodeID and self.nodeID_to_company to convert between node indices and firm names.
        """
        edge_index_dicts = [self.get_edge_index_dict(ts) for ts in time_stamps]
        edge_weight_dicts = [self.get_edge_weight_dict(ts) for ts in time_stamps]
        feature_dicts = [{"firm": None} for _ in range(len(time_stamps))] #not clear what node / firm features should be atm
        target_dicts = [{"firm":self.node_targets[ts].todense().copy()} if ts in self.prevalent_timestamps else None for ts in time_stamps ]
        date_ranges = [{"firm": np.array(self.get_date_range(ts))} for ts in time_stamps]
        
        tempGraph = DynamicHeteroGraphTemporalSignal(edge_index_dicts = edge_index_dicts,
                                                    edge_weight_dicts = edge_weight_dicts,
                                                    feature_dicts = feature_dicts,
                                                    target_dicts = target_dicts,
                                                    date_ranges = date_ranges)
        return tempGraph
        
    def loadData(self, current_date, prior_days, next_days):
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
        days_after_start = get_days_between(self.start_date, current_date)
        prior_earliest_ts = np.ceil((days_after_start - prior_days + 1 + self.length_ts) / self.length_ts) * self.length_ts
        prior_latest_ts = (days_after_start + 1) // self.length_ts * self.length_ts
        prior_timestamps = list(range(int(prior_earliest_ts), int(prior_latest_ts) + 1, self.length_ts))
        
        next_earliest_ts = np.ceil((days_after_start + 1 + self.length_ts) / self.length_ts) * self.length_ts
        next_latest_ts = (days_after_start + 1 + next_days) // self.length_ts * self.length_ts
        next_timestamps = list(range(int(next_earliest_ts), int(next_latest_ts) + 1, self.length_ts))
        
        priorGraph = self.getTemporalGraph(prior_timestamps)
        nextGraph = self.getTemporalGraph(next_timestamps)
        
        return priorGraph, nextGraph

if __name__ == "__main__":
    
    obj = SupplyChainDataset("out.csv", "2022-01-01", 1, "total_amount")
    print(f"Total Firms: {obj.num_nodes_total}\nTotal Time-Stamped Edges: {obj.num_edges_total}")
    priorGraph, nextGraph = obj.loadData(current_date = "2022-01-10", prior_days = 5, next_days = 10)
    


