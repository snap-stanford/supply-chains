"""
This file is for alchemizing the .csv file of edges produced in extract_graph_data.py into Temporal
Graph data structures from the PyTorch Geometric library (see https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/signal.html)
"""

import datetime
import numpy as np
from torch_geometric_temporal import DynamicHeteroGraphTemporalSignal
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix

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
    return (temporal_node_data - norms["mean"]) / (norms["std"] + epsilon)

def node_unnormalize(normalized_node_data, norms = None, epsilon = 10**(-10)):
    """ analagous parameter recommendations as normalize() above """
    if (norms == None): return normalized_node_data
    return normalized_node_data * (norms["std"] + epsilon) + norms["mean"]


class SupplyChainDataset(object):
    
    def __init__(self, edges_csv_file: str, start_date: str, length_timestamps: int = 1, 
                 metric: str = "total_amount", dual_edge: bool = False, lags: int = 5):
        """
        Constructor method for the temporal Supply Chain dataset, which formats the .csv from 
        extract_graph_data.py into graph-based data structures from torch_geometric_temporal
        
        Args:
            edges_csv_file (str): the path to the CSV file storing transaction edges
            start_date (str): start date from data was gathered (date corresponding to a time_stamp of 0)
            length_timestamps (int): number of days aggregated per time stamp
            metric (str): the metric (out of total_amount, total_quantity, bill_count, total_weight) 
                          to assign to each edge as its "weight"
            dual_edge (bool): By default, edges are drawn from supplier -> buyer (flow of material). If True, 
                          this will also draw reverse edges from buyer -> supplier (flow of cash). 
            lags (int): Number of past time_stamps to use for predicting the next time_stamp 

        Returns:
            None
        """
        all_metrics = {"total_amount", "total_quantity", "bill_count", "total_weight"}
        if metric not in all_metrics:
            raise ValueError(f"Metric must be {all_metrics}")
        self.start_date = start_date
        self.length_ts = length_timestamps
        self.dual_edge = dual_edge
        self.lags = lags
            
        #read the dataframe and excise missing supplier & buyer nodes, as well as values
        df = pd.read_csv(edges_csv_file)
        row_names = ["time_stamp","supplier_t","buyer_t","hs6", metric]
        df = df[row_names][(~df[metric].isna()) & (df[metric] > 0) & (df["supplier_t"] != "") & (df["buyer_t"] != "")]
        self.num_edges_total = len(df)
        companies = list(set(df["supplier_t"]).union(set(df["buyer_t"])))
        self.num_nodes_total = len(companies)
        self.num_products_total = len(set(df["hs6"]))
        
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
            #directed edge from supplier to buyer (flow of material)
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
        self.max_ts = max(self.prevalent_timestamps) #latest time_stamp with non-empty data
        
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
        
        #obtain the node features using a past sliding window of firm inputs/outputs
        self.assemble_node_features()
        
    def get_edge_index_dict(self,time_stamp):
        if (time_stamp not in self.prevalent_timestamps):
            return {"firm": None}
        edges_dict = self.edge_index_dict[time_stamp]
        unique_products = list(edges_dict.keys()) #different products that delineate edge relations
        #constructs forward edges from supplier to buyer (flow of material)
        index_dict = {("firm",(product,"material"),"firm"): np.transpose(
            np.array(edges_dict[product])) for product in unique_products}
        #constructs reverse edges from buyer to supplier (flow of cash)
        if self.dual_edge == True: 
            dual_index_dict = {("firm",(product, "cash"), "firm"): np.flip(np.transpose(
                np.array(edges_dict[product])), axis = 0).copy() for product in unique_products}
            index_dict.update(dual_index_dict)
        return index_dict
    
    def get_edge_weight_dict(self,time_stamp):
        if (time_stamp not in self.prevalent_timestamps):
            return {"firm": None}
        weight_dict = self.edge_weight_dict[time_stamp]
        unique_products = list(weight_dict.keys()) #different products that delineate edge relations
        edge_weight_dict = {("firm",(product,"material"),"firm"): np.array(
            weight_dict[product]) for product in unique_products}
        if self.dual_edge == True: 
            dual_index_dict = {("firm",(product, "cash"), "firm"): np.array(
                weight_dict[product]) for product in unique_products}
            edge_weight_dict.update(dual_index_dict)
        return edge_weight_dict
        
    def get_date_range(self,time_stamp):
        lower_date = get_forward_date(self.start_date, time_stamp - self.length_ts)
        upper_date = get_forward_date(self.start_date, time_stamp - 1)
        return [lower_date, upper_date]
    
    def assemble_node_features(self):
        """
        calculates sliding window of past <self.lags> time_stamps, averaging input/output for each firm 
        stores the averaged values in np.array self.node_IO_table, accessed by time_stamp indices
        """
        K = self.lags
        self.node_IO_table = np.zeros(shape = (max(self.prevalent_timestamps) // self.length_ts, self.num_nodes_total, 2))
        
        rolling_sum = np.zeros(shape = (self.num_nodes_total, 2))
        for timestamp in range(self.length_ts, max(self.prevalent_timestamps) + 1, self.length_ts):
            #for the (t)th timestamp, average each firm's values between the (t - K)th and (t - 1)th timestamps
            if (timestamp - self.length_ts in self.prevalent_timestamps):
                rolling_sum += self.node_targets[timestamp - self.length_ts].todense()
            if (timestamp - (K + 1) * self.length_ts in self.prevalent_timestamps):
                rolling_sum -= self.node_targets[timestamp - (K + 1) * self.length_ts].todense()
                
            #divide by the number of time_stamps the amounts are averaged over
            self.node_IO_table[timestamp // self.length_ts - 1] = rolling_sum / min(
                K, timestamp // self.length_ts - 1) if timestamp > self.length_ts else 0
            
    def getTemporalGraph(self, time_stamps):
        """
        given a list of time_stamps, produces a PyG temporal graph covering time-stamped transaction edges 
        between node firms. Here, edge weights are the metric (e.g. total_amount, bill_count) given
        to the constructor method, node targets are the total input/output of each firm at each time. For now,
        node features are the average input/output of the past <self.lags> days of each firm. 
        
        The edge relations are tuples of the form ("firm", (product, flow direction), "firm"), where flow 
        direction is "material" for supplier -> buyer and "cash" for buyer -> supplier. Date_ranges corresponds 
        to the interval of dates aggregated for each time_stamp. Use  self.company_to_nodeID and 
        self.nodeID_to_company to convert between node indices and firm names.
        """
        edge_index_dicts = [self.get_edge_index_dict(ts) for ts in time_stamps]
        edge_weight_dicts = [self.get_edge_weight_dict(ts) for ts in time_stamps]
        feature_dicts = [{"firm":self.node_IO_table[ts // self.length_ts - 1].copy()} if (
            ts > self.length_ts and ts <= max(self.prevalent_timestamps)) else None for ts in time_stamps]
        target_dicts = [{"firm":self.node_targets[ts].todense().copy()} if ts in self.prevalent_timestamps else None for ts in time_stamps ]
        date_ranges = [{"firm": np.array(self.get_date_range(ts))} for ts in time_stamps]
        
        tempGraph = DynamicHeteroGraphTemporalSignal(edge_index_dicts = edge_index_dicts,
                                                    edge_weight_dicts = edge_weight_dicts,
                                                    feature_dicts = feature_dicts,
                                                    target_dicts = target_dicts,
                                                    date_ranges = date_ranges)
        return tempGraph

    def segment_timestamps(self, current_date, prior_days, next_days):
        """
        retrieve two lists of time_stamps corresponding to the <prior_days> days before <current_date>,  
        and the following <next_days> days of the dataset, respectively. 
        """
        days_after_start = get_days_between(self.start_date, current_date)
        prior_earliest_ts = np.ceil((days_after_start - prior_days + 1 + self.length_ts) / self.length_ts) * self.length_ts
        prior_latest_ts = (days_after_start + 1) // self.length_ts * self.length_ts
        prior_timestamps = list(range(int(prior_earliest_ts), int(prior_latest_ts) + 1, self.length_ts))
        
        next_earliest_ts = np.ceil((days_after_start + 1 + self.length_ts) / self.length_ts) * self.length_ts
        next_latest_ts = (days_after_start + 1 + next_days) // self.length_ts * self.length_ts
        next_timestamps = list(range(int(next_earliest_ts), int(next_latest_ts) + 1, self.length_ts))
        return prior_timestamps, next_timestamps
        
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
        prior_timestamps, next_timestamps = self.segment_timestamps(current_date, prior_days, next_days)
        priorGraph = self.getTemporalGraph(prior_timestamps)
        nextGraph = self.getTemporalGraph(next_timestamps)
        
        return priorGraph, nextGraph

class SupplyChainDatasetPredictive(SupplyChainDataset):
    """
    time-lagged version of the Supply Chain Dataset, where node targets are set <self.lags>
    time_stamps into the future, rather than the present values for the corresponding time_stamp
    
    TODO: alternatively, one can still use present values by setting use_present_labels == True, 
    and the node features instead will be comprised of the past <self.lags> days 
    """
    def __init__(self, *args, **kwargs):
        super(SupplyChainDatasetPredictive,self).__init__(*args, **kwargs)
    
    def ts2index(self, time_stamp): #converts from a time_stamp to a place in a storage arraay
        return time_stamp // self.length_ts - 1
    
    # override function in the parent class
    def assemble_node_features(self):
        #storing the input/outputs of each firm at each timestamp in a sparse matrix
        self.node_IO_table = lil_matrix((self.max_ts // self.length_ts, self.num_nodes_total * 2), dtype = np.float32)
        for timestamp in range(self.length_ts, max(self.prevalent_timestamps) + 1, self.length_ts):
            if (timestamp in self.prevalent_timestamps):
                self.node_IO_table[self.ts2index(timestamp)] = self.node_targets[timestamp].reshape(1,-1)
            else:
                self.node_IO_table[self.ts2index(timestamp)] = 0 #missing data for that timestamp 
        
    def get_node_normalizations(self, timestamps: list):
        if (len(timestamps) <= 1): return None
        time_indices = [self.ts2index(ts) for ts in timestamps]
        node_IO_selected = np.array(self.node_IO_table[time_indices].todense()).reshape(-1, self.num_nodes_total, 2)
        mean, std = np.mean(node_IO_selected, axis = 0), np.std(node_IO_selected, axis = 0)
        return {"mean": mean, "std": std}
    
    # override function in the parent class
    def getTemporalGraph(self, time_stamps, norms):
        edge_index_dicts = [self.get_edge_index_dict(ts) for ts in time_stamps]
        edge_weight_dicts = [self.get_edge_weight_dict(ts) for ts in time_stamps]
        
        #set the features to be a panorama of the next <self.lags> - 1 time_stamps, including the current one
        feature_dicts = [{"firm": node_normalize(np.array(self.node_IO_table[self.ts2index(ts): self.ts2index(ts) + self.lags,:].todense()).reshape(
            -1, self.num_nodes_total, 2), norms)} if (self.lags <= self.ts2index(ts) + self.lags <= self.ts2index(self.max_ts) + 1) else None for ts in time_stamps]
        
        #set the targets to be node inputs/outputs <self.lags> time_stamps into the future
        target_dicts = [{"firm": node_normalize(np.array(self.node_IO_table[self.ts2index(ts) + self.lags,:].todense()).reshape(-1,2), norms)} if (
           0 <= self.ts2index(ts) + self.lags <= self.ts2index(self.max_ts)) else None for ts in time_stamps]
        
        date_ranges = [{"firm": np.array(self.get_date_range(ts))} for ts in time_stamps]
        tempGraph = DynamicHeteroGraphTemporalSignal(edge_index_dicts = edge_index_dicts,
                                                    edge_weight_dicts = edge_weight_dicts,
                                                    feature_dicts = feature_dicts,
                                                    target_dicts = target_dicts,
                                                    date_ranges = date_ranges)
        return tempGraph
    
    # override function in the parent class
    def loadData(self, current_date, prior_days, next_days, normalize = True):
        prior_timestamps, next_timestamps = self.segment_timestamps(current_date, prior_days, next_days)
        # set normalization values based on valid prior_timestamps (using for training)
        norms = self.get_node_normalizations([ts for ts in prior_timestamps if 0 <= self.ts2index(ts) <= self.ts2index(self.max_ts)])
        priorGraph = self.getTemporalGraph(prior_timestamps, norms if normalize == True else None)
        nextGraph = self.getTemporalGraph(next_timestamps, norms if normalize == True else None)

        return priorGraph, nextGraph, norms
    
if __name__ == "__main__":
    
    obj = SupplyChainDatasetPredictive("daily_transactions_2019.csv", "2019-01-01", 1, 
                                       metric = "total_amount", lags = 5)
    print(f"Total Firms: {obj.num_nodes_total}\nTotal Time-Stamped Supplier \u2192 Buyer Edges: {obj.num_edges_total}")
    priorGraph, nextGraph, norms = obj.loadData(current_date = "2019-03-10", prior_days = 50, next_days = 10)
    for timestep, snapshot in enumerate(priorGraph):
        print(snapshot)
        break
