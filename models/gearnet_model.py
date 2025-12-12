"""
GearNet implementation using TorchDrug for protein structure-based embeddings
GearNet is a geometric graph neural network for protein representation learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdrug import core, data, layers
from torchdrug.layers import geometry
from torchdrug.core import Registry as R
import numpy as np
from typing import Dict, List, Optional, Tuple


@R.register("models.GearNetStruct")
class GearNet(nn.Module, core.Configurable):
    """
    GearNet (Geometric graph neural network) implementation for protein structure analysis
    Based on the paper "Protein Representation Learning by Geometric Structure Pretraining"
    """

    def __init__(self,
                 input_dim=256,
                 hidden_dims=[512, 512, 512, 512],
                 num_relation=7,  # number of geometric relations
                 batch_norm=True,
                 short_cut=True,
                 concat_hidden=False,
                 readout="mean",
                 num_mlp_layer=2,
                 activation="relu",
                 dropout=0.1,
                 **kwargs):
        """
        Initialize GearNet model

        Args:
            input_dim: Input node feature dimension
            hidden_dims: List of hidden dimensions for each layer
            num_relation: Number of geometric relations to consider
            batch_norm: Whether to use batch normalization
            short_cut: Whether to use residual connections
            concat_hidden: Whether to concat hidden representations
            readout: Readout function for graph-level prediction
            num_mlp_layer: Number of MLP layers for message passing
            activation: Activation function to use
            dropout: Dropout rate
        """
        super(GearNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_relation = num_relation
        self.batch_norm = batch_norm
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.num_mlp_layer = num_mlp_layer
        self.dropout = dropout

        # Use the appropriate activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()  # default

        # Import GearNet layer inside the initialization to avoid ESM import issues
        try:
            from torchdrug.models.gearnet import GeometryAwareRelationalGraphNeuralNetwork
            self.using_torchdrug_gearnet = True
        except (ImportError, OSError) as e:
            print(f"TorchDrug not available or failed to load: {e}. GearNet will not work.")
            self.using_torchdrug_gearnet = False
            return

        # Initial embedding layer
        self.input_linear = nn.Linear(input_dim, hidden_dims[0])

        # GearNet layers - geometric graph neural network layers
        self.gearnet_layers = nn.ModuleList()
        layer_input_dim = hidden_dims[0]

        for i, hidden_dim in enumerate(hidden_dims):
            # Use the actual GearNet layer from TorchDrug
            layer = GeometryAwareRelationalGraphNeuralNetwork(
                layer_input_dim,
                hidden_dim,
                num_relation,
                batch_norm,
                short_cut,
                concat_hidden,
                num_mlp_layer,
                self.activation,
                dropout
            )
            self.gearnet_layers.append(layer)
            layer_input_dim = hidden_dim if not concat_hidden else layer_input_dim  # Fixed

        # Calculate output dimension
        if concat_hidden:
            self.output_dim = sum(hidden_dims)
        else:
            self.output_dim = hidden_dims[-1]

        # Readout layer
        if readout == "mean":
            self.readout = layers.GlobalMeanPool()
        elif readout == "sum":
            self.readout = layers.GlobalSumPool()
        elif readout == "max":
            self.readout = layers.GlobalMaxPool()
        else:
            self.readout = layers.GlobalMeanPool()  # default

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Forward pass through the GearNet

        Args:
            graph: TorchDrug graph object containing protein structure
            input: Input node features (batch_size, num_nodes, input_dim)
            all_loss: For compatibility with TorchDrug training
            metric: For compatibility with TorchDrug training

        Returns:
            dict: Output containing node embeddings and graph embeddings
        """
        if not self.using_torchdrug_gearnet:
            raise ImportError("TorchDrug not available. GearNet cannot run.")

        # Initialize node features
        hiddens = []
        layer_input = self.input_linear(input)

        for layer in self.gearnet_layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and self.concat_hidden:
                hiddens.append(hidden)
            elif self.short_cut:
                hiddens = [hidden]
            else:
                hiddens = [hidden]
            layer_input = hidden

        # Concatenate all hidden representations if concat_hidden is True
        if self.concat_hidden and len(hiddens) > 1:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1] if hiddens else layer_input

        # Apply readout for graph-level representation
        graph_feature = self.readout(graph, node_feature)

        return {
            "node_feature": node_feature,
            "graph_feature": graph_feature
        }


class GearNetFromCoordinates(nn.Module):
    """
    A wrapper that allows GearNet to work with raw 3D coordinates (N, CA, C)
    This creates a geometric graph from coordinates and feeds it to GearNet
    """

    def __init__(self,
                 hidden_dim=512,
                 pretrained_path=None,
                 freeze=True):
        """
        Initialize GearNet wrapper for coordinate inputs

        Args:
            hidden_dim: Hidden dimension for the model
            pretrained_path: Path to pre-trained GearNet weights
            freeze: Whether to freeze the model during training
        """
        super(GearNetFromCoordinates, self).__init__()

        self.hidden_dim = hidden_dim
        self.freeze = freeze

        # Import GearNet model from TorchDrug inside the initialization
        try:
            from torchdrug.models.gearnet import GeometryAwareRelationalGraphNeuralNetwork
            self.using_torchdrug_gearnet = True
        except (ImportError, OSError) as e:
            print(f"TorchDrug not available or failed to load: {e}. GearNet cannot be created.")
            self.using_torchdrug_gearnet = False
            return

        # Create actual GearNet model with parameters that work across different TorchDrug versions
        # The exact API can vary depending on the version of TorchDrug you have installed
        gearnet_kwargs = {
            'input_dim': 3,  # Input is 3D coordinates
            'hidden_dim': hidden_dim,
            'num_relation': 7,
            'batch_norm': True,
            'short_cut': True,
            'concat_hidden': False,
            'num_mlp_layer': 2,
            'activation': F.relu,
            'dropout': 0.1
        }

        # Attempt to create the model, handling different API versions
        try:
            self.gearnet_model = GeometryAwareRelationalGraphNeuralNetwork(**gearnet_kwargs)
        except TypeError:
            # If some parameters are not accepted, try with minimal parameters
            # Different versions of TorchDrug may have different parameter names
            try:
                # Some versions use 'hidden_dims' (plural) instead of 'hidden_dim'
                gearnet_kwargs_alt = {
                    'input_dim': 3,
                    'hidden_dims': [hidden_dim] * 4,  # Use a list instead
                    'num_relation': 7,
                    'batch_norm': True,
                    'short_cut': True,
                    'concat_hidden': False
                }
                self.gearnet_model = GeometryAwareRelationalGraphNeuralNetwork(**gearnet_kwargs_alt)
            except TypeError:
                # Try with minimal required parameters only
                self.gearnet_model = GeometryAwareRelationalGraphNeuralNetwork(
                    input_dim=3,
                    hidden_dim=hidden_dim,
                    num_relation=7
                )

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        print(f"Successfully created GearNet model with TorchDrug, hidden_dim={hidden_dim}")

    def forward(self, n_coords, ca_coords, c_coords):
        """
        Forward pass taking N, CA, C coordinates and returning embeddings.
        This implementation uses a vectorized and GPU-friendly graph construction.
        """
        if not self.using_torchdrug_gearnet:
            raise ImportError("TorchDrug not available. GearNet cannot run.")

        is_half = ca_coords.dtype == torch.half
        device = ca_coords.device

        if is_half:
            ca_coords = ca_coords.float()

        batch_size, seq_len, _ = ca_coords.shape

        # Stack N, CA, C coordinates to create features
        # In the actual GearNet, these would be used to compute geometric features
        # For now, we'll use CA coordinates as simple node features
        node_features = ca_coords.view(-1, 3)  # (batch_size * seq_len, 3)

        # Create a batch of graphs - one per protein
        # Each protein needs its own edge construction based on its coordinates
        graphs = []
        for i in range(batch_size):
            # Get coordinates for this protein
            ca_coords_single = ca_coords[i]  # (seq_len, 3)

            # Vectorized sequential edges - these are based on sequence adjacency
            arange = torch.arange(seq_len, device=device)
            seq_edges = []
            for offset in [-3, -2, -1, 1, 2, 3]:
                src = arange
                dst = arange + offset
                mask = (dst >= 0) & (dst < seq_len)
                if mask.any():
                    seq_edges.append(torch.stack([src[mask], dst[mask]], dim=1))

            if seq_edges:
                seq_edge_list = torch.cat(seq_edges, dim=0)
                seq_relation_type = torch.zeros(len(seq_edge_list), 1, dtype=torch.long, device=device)
            else:
                # If no sequence edges, we still need at least one edge for the graph
                seq_edge_list = torch.zeros((1, 2), dtype=torch.long, device=device)
                seq_relation_type = torch.zeros((1, 1), dtype=torch.long, device=device)

            # Vectorized k-NN spatial edges - these are based on 3D proximity
            dist_matrix = torch.cdist(ca_coords_single.unsqueeze(0), ca_coords_single.unsqueeze(0)).squeeze(0)  # (seq_len, seq_len)
            k = min(10, seq_len - 1)
            if k > 0:
                topk_dists, topk_indices = torch.topk(dist_matrix, k + 1, largest=False, dim=1)  # (seq_len, k+1)

                src = torch.arange(seq_len, device=device).view(-1, 1).expand(-1, k)  # (seq_len, k)
                dst = topk_indices[:, 1:]  # (seq_len, k) - exclude self-loops

                # Flatten and filter
                src = src.flatten()  # (seq_len * k)
                dst = dst.flatten()  # (seq_len * k)
                mask = (dst >= 0) & (dst < seq_len)
                src = src[mask]
                dst = dst[mask]

                if len(src) > 0:
                    spatial_edge_list = torch.stack([src, dst], dim=1)
                    spatial_relation_type = torch.ones(len(spatial_edge_list), 1, dtype=torch.long, device=device) * 2

                    edge_list = torch.cat([seq_edge_list, spatial_edge_list], dim=0)
                    relation_type = torch.cat([seq_relation_type, spatial_relation_type], dim=0)
                else:
                    edge_list = seq_edge_list
                    relation_type = seq_relation_type
            else:
                edge_list = seq_edge_list
                relation_type = seq_relation_type

            # Create graph for this protein
            graph = data.Graph(
                edge_list=torch.cat([edge_list, relation_type], dim=1),
                num_node=seq_len,
                num_relation=7
            )
            graphs.append(graph)

        # Batch the graphs together
        batched_graph = data.graph_collate(graphs).to(device)

        # Set node features for the entire batch
        batched_graph.node_feature = node_features

        # Forward pass through the actual GearNet model
        with torch.cuda.amp.autocast(enabled=False):
            output = self.gearnet_model(batched_graph, batched_graph.node_feature)

        # Extract node embeddings and reshape to (batch_size, seq_len, hidden_dim)
        node_embeddings = output.get("node_feature", output)
        if isinstance(node_embeddings, dict):
            node_embeddings = node_embeddings.get("node_feature", node_embeddings.get("graph_feature"))

        # Reshape back to batch format
        final_embeddings = node_embeddings.view(batch_size, seq_len, self.hidden_dim)

        if is_half:
            final_embeddings = final_embeddings.half()

        return final_embeddings


def create_pretrained_gearnet(hidden_dim=512, freeze=True):
    """
    Factory function to create a pre-trained GearNet model.
    Note: This implementation uses a randomly initialized GearNet model.
    Loading pre-trained weights is not currently supported in this script.

    Args:
        hidden_dim: Hidden dimension for the model
        freeze: Whether to freeze the model during training

    Returns:
        GearNetFromCoordinates: A configured GearNet wrapper that accepts coordinates
    """
    model = GearNetFromCoordinates(
        hidden_dim=hidden_dim,
        freeze=freeze
    )

    if not model.using_torchdrug_gearnet:
        raise ImportError("TorchDrug is required but not available or failed to load properly.")

    print("Using randomly initialized GearNet. Pre-trained weights are not loaded.")

    return model

import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(graph, save_path="graph.png"):
    """
    Visualizes a torchdrug.data.Graph object.
    """
    edge_list = graph.edge_list.cpu().numpy()

    # Create a networkx graph
    G = nx.Graph()
    for i in range(graph.num_node):
        G.add_node(i)

    for i in range(graph.num_edge):
        u, v, rel = edge_list[i]
        G.add_edge(u, v, relation=rel)

    # Draw the graph
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)

    edge_colors = [d['relation'] for u, v, d in G.edges(data=True)]

    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            edge_color=edge_colors, cmap=plt.cm.get_cmap('viridis'),
            node_size=500, font_size=10)

    plt.title("Graph Structure")
    plt.savefig(save_path)
    print(f"Graph visualization saved to {save_path}")
    plt.close()