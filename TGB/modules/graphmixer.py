import numpy as np
import torch
import torch.nn as nn

from modules.neighbor_loader import LastNeighborLoaderGraphmixer

EMPTY_VALUE = -1 # same as what's filled in all attributes when resetting the neighborloader 

class GraphMixer(nn.Module):

    def __init__(self, node_raw_features: torch.Tensor, edge_feat_dim: int,
                 time_feat_dim: int, num_tokens: int, num_layers: int = 2, token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0, dropout: float = 0.1, time_gap: int = 2000, debug: bool=False):
        """
        TCL model.
        :param node_raw_features: Tensor, shape (num_nodes + 1, node_feat_dim)
        :param edge_feat_dim: int, edge feature dimension (axis=1)
        # :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param num_tokens: int, number of tokens
        :param num_layers: int, number of transformer layers
        :param token_dim_expansion_factor: float, dimension expansion factor for tokens
        :param channel_dim_expansion_factor: float, dimension expansion factor for channels
        :param dropout: float, dropout rate
        """
        super(GraphMixer, self).__init__()

        self.node_raw_features = node_raw_features.float() # this is preset to full data features  

        self.neighbor_sampler = None # assigned later 
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.token_dim_expansion_factor = token_dim_expansion_factor
        self.channel_dim_expansion_factor = channel_dim_expansion_factor
        self.dropout = dropout
        self.num_neighbors = num_tokens
        self.time_gap = time_gap
        self.debug = debug

        self.num_channels = self.edge_feat_dim
        # in GraphMixer, the time encoding function is not trainable
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, parameter_requires_grad=False)
        self.projection_layer = nn.Linear(self.edge_feat_dim + time_feat_dim, self.num_channels)

        self.mlp_mixers = nn.ModuleList([
            MLPMixer(num_tokens=self.num_tokens, num_channels=self.num_channels,
                     token_dim_expansion_factor=self.token_dim_expansion_factor,
                     channel_dim_expansion_factor=self.channel_dim_expansion_factor, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=self.num_channels + self.node_feat_dim, out_features=self.node_feat_dim, bias=True)

    def compute_src_dst_prod_node_temporal_embeddings(self, src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor, prod_node_ids: torch.Tensor, node_interact_times: torch.Tensor, 
                                                        neighbor_loader: LastNeighborLoaderGraphmixer):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: Tensor, shape (batch_size, )
        :param dst_node_ids: Tensor, shape (batch_size, )
        :param prod_node_ids: Tensor, shape (batch_size, )
        :param node_interact_times: Tensor, shape (batch_size, )
        :return:
        """
        self.neighbor_sampler = neighbor_loader

        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.compute_node_temporal_embeddings(node_ids=src_node_ids, node_interact_times=node_interact_times,
                                                                    time_gap=self.time_gap)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.compute_node_temporal_embeddings(node_ids=dst_node_ids, node_interact_times=node_interact_times,
                                                                    time_gap=self.time_gap)
        # Tensor, shape (batch_size, node_feat_dim)
        prod_node_embeddings = self.compute_node_temporal_embeddings(node_ids=prod_node_ids, node_interact_times=node_interact_times,
                                                                    time_gap=self.time_gap)

        return src_node_embeddings, dst_node_embeddings, prod_node_embeddings

    def compute_node_temporal_embeddings(self, node_ids: torch.Tensor, node_interact_times: torch.Tensor, time_gap: int = 2000):
        """
        given node ids node_ids, and the corresponding time node_interact_times, return the temporal embeddings of nodes in node_ids
        :param node_ids: tensor, shape (batch_size, ), node ids
        :param node_interact_times: tensor, shape (batch_size, ), node interaction times
        :param time_gap: NOT YET USED, int, time gap for neighbors to compute node features
        :return:
        """
        # link encoder,
        # get temporal neighbors, including neighbor ids, edge ids, time, and edge feature information
        # neighbor_node_ids, tensor, shape (batch_size, num_neighbors) 
            # each entry in position (i,j) represents the id of the j-th dst node of src node node_ids[i] with an interaction before node_interact_times[i]
            # ndarray, shape (batch_size, num_neighbors)
        # neighbor_edge_ids, tensor, shape (batch_size, num_neighbors)
        # neighbor_times, tensor, shape (batch_size, num_neighbors)
        # neighbor_edge_features, tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        neighbor_node_ids, _, neighbor_edge_ids, neighbor_times, neighbor_edge_features = self.neighbor_sampler(node_ids)

        # Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        nodes_edge_raw_features = neighbor_edge_features
        # Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        if self.debug:
            print("node_interact_times (shape, value)", node_interact_times[:, None].shape, node_interact_times[:, None])
            print("neighbor_times (shape, value)", neighbor_times.shape, neighbor_times)
            print("input to time encoder", node_interact_times[:, None] - neighbor_times)
        nodes_neighbor_time_features = self.time_encoder(timestamps=(node_interact_times[:, None] - neighbor_times).float())

        # ndarray, set the time features to all zeros for the padded timestamp
        nodes_neighbor_time_features[neighbor_node_ids == EMPTY_VALUE] = 0.0
        if self.debug:
            print("output of time encoder (zeroed out if applicable)", nodes_neighbor_time_features.shape, nodes_neighbor_time_features)

        # Tensor, shape (batch_size, num_neighbors, edge_feat_dim + time_feat_dim)
        combined_features = torch.cat([nodes_edge_raw_features, nodes_neighbor_time_features], dim=-1)
        # Tensor, shape (batch_size, num_neighbors, num_channels)
        combined_features = self.projection_layer(combined_features)

        for mlp_mixer in self.mlp_mixers:
            # Tensor, shape (batch_size, num_neighbors, num_channels)
            combined_features = mlp_mixer(input_tensor=combined_features)

        # Tensor, shape (batch_size, num_channels)
        combined_features = torch.mean(combined_features, dim=1)

        # node encoder
        # get temporal neighbors of nodes, including neighbor ids
        # time_gap_neighbor_node_ids, ndarray, shape (batch_size, time_gap)
        # time_gap_neighbor_node_ids, _, _ = self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
        #                                                                                node_interact_times=node_interact_times,
        #                                                                                num_neighbors=time_gap)
        time_gap_neighbor_node_ids = neighbor_node_ids # TODO: simplified version now, assuming args.time_gap == args.num_neighbors

        # Tensor, shape (batch_size, time_gap, node_feat_dim)
        nodes_time_gap_neighbor_node_raw_features = self.node_raw_features[time_gap_neighbor_node_ids]

        # Tensor, shape (batch_size, time_gap)
        valid_time_gap_neighbor_node_ids_mask = (time_gap_neighbor_node_ids > 0).float()
        # note that if a node has no valid neighbor (whose valid_time_gap_neighbor_node_ids_mask are all zero), directly set the mask to -np.inf will make the
        # scores after softmax be nan. Therefore, we choose a very large negative number (-1e10) instead of -np.inf to tackle this case
        # Tensor, shape (batch_size, time_gap)
        valid_time_gap_neighbor_node_ids_mask[valid_time_gap_neighbor_node_ids_mask == 0] = -1e10

        # Tensor, shape (batch_size, time_gap)
        scores = torch.softmax(valid_time_gap_neighbor_node_ids_mask, dim=1)

        # Tensor, shape (batch_size, node_feat_dim), average over the time_gap neighbors
        nodes_time_gap_neighbor_node_agg_features = torch.mean(nodes_time_gap_neighbor_node_raw_features * scores.unsqueeze(dim=-1), dim=1)
        if self.debug:
            print("print nodes_time_gap_neighbor_node_agg_features (expect the same across axis=0 for empty graph)", nodes_time_gap_neighbor_node_agg_features)

        # Tensor, shape (batch_size, node_feat_dim), add features of nodes in node_ids
        output_node_features = nodes_time_gap_neighbor_node_agg_features + self.node_raw_features[node_ids]

        # Tensor, shape (batch_size, node_feat_dim)
        node_embeddings = self.output_layer(torch.cat([combined_features, output_node_features], dim=1))

        return node_embeddings
    
class FeedForwardNet(nn.Module):

    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        """
        two-layered MLP with GELU activation function.
        :param input_dim: int, dimension of input
        :param dim_expansion_factor: float, dimension expansion factor
        :param dropout: float, dropout rate
        """
        super(FeedForwardNet, self).__init__()

        self.input_dim = input_dim
        self.dim_expansion_factor = dim_expansion_factor
        self.dropout = dropout

        self.ffn = nn.Sequential(nn.Linear(in_features=input_dim, out_features=int(dim_expansion_factor * input_dim)),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(in_features=int(dim_expansion_factor * input_dim), out_features=input_dim),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        """
        feed forward net forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        return self.ffn(x)


class MLPMixer(nn.Module):

    def __init__(self, num_tokens: int, num_channels: int, token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0, dropout: float = 0.0):
        """
        MLP Mixer.
        :param num_tokens: int, number of tokens
        :param num_channels: int, number of channels
        :param token_dim_expansion_factor: float, dimension expansion factor for tokens
        :param channel_dim_expansion_factor: float, dimension expansion factor for channels
        :param dropout: float, dropout rate
        """
        super(MLPMixer, self).__init__()

        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_feedforward = FeedForwardNet(input_dim=num_tokens, dim_expansion_factor=token_dim_expansion_factor,
                                                dropout=dropout)

        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(input_dim=num_channels, dim_expansion_factor=channel_dim_expansion_factor,
                                                  dropout=dropout)

    def forward(self, input_tensor: torch.Tensor):
        """
        mlp mixer to compute over tokens and channels
        :param input_tensor: Tensor, shape (batch_size, num_tokens, num_channels)
        :return:
        """
        # mix tokens
        # Tensor, shape (batch_size, num_channels, num_tokens)
        hidden_tensor = self.token_norm(input_tensor.permute(0, 2, 1))
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.token_feedforward(hidden_tensor).permute(0, 2, 1)
        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + input_tensor

        # mix channels
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_norm(output_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_feedforward(hidden_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + output_tensor

        return output_tensor

class TimeEncoder(nn.Module):

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)

        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))

        return output

class MergeLayer(nn.Module):

    def __init__(self, input_dim1: int, input_dim2: int, input_dim3: int, hidden_dim: int, output_dim: int):
        """
        Merge Layer to merge two inputs via: input_dim1 + input_dim2 + input_dim3 -> hidden_dim -> output_dim.
        :param input_dim1: int, dimension of first input
        :param input_dim2: int, dimension of the second input
        :param input_dim3: int, dimension of the third input
        :param hidden_dim: int, hidden dimension
        :param output_dim: int, dimension of the output
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2 + input_dim3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor, input_3: torch.Tensor):
        """
        merge and project the inputs
        :param input_1: Tensor, shape (*, input_dim1)
        :param input_2: Tensor, shape (*, input_dim2)
        :param input_3: Tensor, shape (*, input_dim3)
        :return:
        """
        # Tensor, shape (*, input_dim1 + input_dim2 + input_dim3)
        x = torch.cat([input_1, input_2, input_3], dim=1)
        # Tensor, shape (*, output_dim)
        h = self.fc2(self.act(self.fc1(x)))
        return h