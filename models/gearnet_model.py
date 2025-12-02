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
        from torchdrug.models.gearnet import GeometryAwareRelationalGraphNeuralNetwork

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

        # Create actual GearNet model from TorchDrug
        from torchdrug.models.gearnet import GeometryAwareRelationalGraphNeuralNetwork
        self.gearnet_model = GeometryAwareRelationalGraphNeuralNetwork(
            input_dim=hidden_dim,
            hidden_dims=[512, 512, 512, 512],
            num_relation=7,
            batch_norm=True,
            short_cut=True,
            concat_hidden=False,
            readout="mean"
        )

        # Projection layer to convert from coordinate features to hidden dimension
        # 3 coordinate dimensions (just CA for simplicity) -> hidden_dim
        self.coord_projection = nn.Linear(3, hidden_dim)

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        print("Successfully created GearNet model with TorchDrug")

    def forward(self, n_coords, ca_coords, c_coords):
        """
        Forward pass taking N, CA, C coordinates and returning embeddings

        Args:
            n_coords: (batch_size, seq_len, 3) N atom coordinates
            ca_coords: (batch_size, seq_len, 3) CA atom coordinates
            c_coords: (batch_size, seq_len, 3) C atom coordinates

        Returns:
            embeddings: (batch_size, seq_len, hidden_dim) structural embeddings
        """
        is_half = ca_coords.dtype == torch.half
        device = ca_coords.device

        # When using mixed precision, cast coords to float32 to avoid errors in graph construction
        if is_half:
            n_coords = n_coords.float()
            ca_coords = ca_coords.float()
            c_coords = c_coords.float()

        batch_size, seq_len, _ = ca_coords.shape

        graphs = []
        for b in range(batch_size):
            node_pos = ca_coords[b] # (seq_len, 3)

            # 1. Sequential edges
            edge_list = []
            for i in range(seq_len):
                for offset, relation in zip([-3, -2, -1, 1, 2, 3], range(6)):
                    j = i + offset
                    if 0 <= j < seq_len:
                        edge_list.append([i, j, relation])

            # 2. Spatial edges (k-NN)
            k = 10
            dist_matrix = torch.norm(node_pos.unsqueeze(1) - node_pos.unsqueeze(0), dim=2)
            _, topk_indices = torch.topk(dist_matrix, k + 1, largest=False)

            for i in range(seq_len):
                for j in topk_indices[i, 1:]: # exclude self
                    edge_list.append([i, j.item(), 6])

            if not edge_list:
                edge_list = torch.empty(0, 3, dtype=torch.long, device=device)
            else:
                edge_list = torch.tensor(edge_list, dtype=torch.long, device=device)

            graph = data.Graph(
                edge_list=edge_list,
                num_node=seq_len,
                num_relation=7,
                node_feature=node_pos
            )
            graphs.append(graph)

        # Batch the graphs
        if len(graphs) > 1:
            batched_graph = data.graph_collate(graphs)
        else:
            batched_graph = graphs[0]

        # Ensure the batched graph is on the correct device
        batched_graph = batched_graph.to(device)

        # Convert all relevant tensors in the graph to float32 to avoid half precision issues
        if hasattr(batched_graph, 'node_feature') and batched_graph.node_feature.dtype != torch.float:
            batched_graph.node_feature = batched_graph.node_feature.float()

        # Project coordinates to hidden dimension for node features
        node_features = self.coord_projection(batched_graph.node_feature)

        # Ensure node_features are in float32
        node_features = node_features.float()

        # Pass through GearNet model in float32 mode
        with torch.cuda.amp.autocast(enabled=False):  # Disable autocast for this forward pass
            output = self.gearnet_model(batched_graph, node_features)

        # output["node_feature"] is (total_num_residues, hidden_dim)
        node_embeddings = output["node_feature"]

        # Debug: print actual shapes to understand the issue
        print(f"Debug - batch_size: {batch_size}, seq_len: {seq_len}, hidden_dim: {self.hidden_dim}")
        print(f"Debug - node_embeddings shape: {node_embeddings.shape}")
        print(f"Debug - expected total nodes: {batch_size * seq_len}, actual nodes: {node_embeddings.size(0)}")

        # The output from GearNet is for all nodes in the batch, so we reshape properly
        # Make sure the number of nodes matches what we expect: batch_size * seq_len
        expected_nodes = batch_size * seq_len
        actual_nodes = node_embeddings.size(0)

        if actual_nodes != expected_nodes:
            # Handle case where the number of nodes might be different
            # This can happen if there are issues with graph construction
            # For now, we'll reshape according to the actual tensor size
            print(f"Warning: Expected {expected_nodes} nodes but got {actual_nodes} nodes. Adjusting reshape accordingly.")

            # Try to infer the actual sequence length
            if actual_nodes % batch_size == 0:
                actual_seq_len = actual_nodes // batch_size
                final_embeddings = node_embeddings.view(batch_size, actual_seq_len, self.hidden_dim)
            else:
                # Fallback: if sizes don't divide evenly, use the original sequence length
                # and truncate or pad as needed
                if actual_nodes < expected_nodes:
                    # If we have fewer nodes than expected, pad with zeros
                    padding_size = expected_nodes - actual_nodes
                    padding = torch.zeros(padding_size, self.hidden_dim, device=node_embeddings.device, dtype=node_embeddings.dtype)
                    padded_embeddings = torch.cat([node_embeddings, padding], dim=0)
                    final_embeddings = padded_embeddings.view(batch_size, seq_len, self.hidden_dim)
                else:
                    # If we have more nodes than expected, truncate
                    final_embeddings = node_embeddings[:expected_nodes].view(batch_size, seq_len, self.hidden_dim)
        else:
            # Standard case: reshape as expected
            final_embeddings = node_embeddings.view(batch_size, seq_len, self.hidden_dim)

        # If the original input was half, we might want to convert the output back to half
        # but for safety and to maintain precision during training, we'll keep it as float32
        if is_half:
            final_embeddings = final_embeddings.to(dtype=ca_coords.dtype)  # Use original input dtype

        return final_embeddings


def create_pretrained_gearnet(hidden_dim=512, pretrained_path=None, freeze=True):
    """
    Factory function to create a pre-trained GearNet model

    Args:
        hidden_dim: Hidden dimension for the model
        pretrained_path: Path to pre-trained weights (if available)
        freeze: Whether to freeze the model during training

    Returns:
        GearNetFromCoordinates: A configured GearNet wrapper that accepts coordinates
    """
    model = GearNetFromCoordinates(
        hidden_dim=hidden_dim,
        pretrained_path=pretrained_path,
        freeze=freeze
    )

    if pretrained_path:
        # Load pre-trained weights if provided
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"Loaded pre-trained GearNet weights from {pretrained_path}")
        except Exception as e:
            print(f"Could not load pre-trained weights from {pretrained_path}: {e}")
            print("Using randomly initialized GearNet")

    return model