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

        # Import GearNet model from TorchDrug inside the initialization
        from torchdrug.models.gearnet import GeometryAwareRelationalGraphNeuralNetwork

        # Create actual GearNet model with parameters that work across different TorchDrug versions
        # The exact API can vary depending on the version of TorchDrug you have installed
        gearnet_kwargs = {
            'input_dim': 3,  # Input is 3D coordinates
            'hidden_dim': hidden_dim,
            'num_relation': 7,
            'batch_norm': True,
            'short_cut': True,
            'concat_hidden': False
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

            # Create geometric graph edges
            edge_list = []
            for i in range(seq_len):
                # Sequential edges (within ±3 residues)
                for offset in [-3, -2, -1, 1, 2, 3]:
                    j = i + offset
                    if 0 <= j < seq_len:
                        # Relations 0-5 for sequential neighbors
                        relation_type = 0 if offset < 0 else 1  # Use simpler relation types
                        edge_list.append([i, j, relation_type])

            # Spatial edges (k-NN based on 3D distance)
            if seq_len > 1:
                dist_matrix = torch.norm(node_pos.unsqueeze(1) - node_pos.unsqueeze(0), dim=2)
                k = min(10, seq_len - 1)  # Ensure k is not larger than sequence
                if k > 0:
                    _, topk_indices = torch.topk(dist_matrix, k + 1, largest=False)

                    for i in range(seq_len):
                        # Get k nearest neighbors (excluding self)
                        nearest_neighbors = topk_indices[i, 1:k+1]
                        for j in nearest_neighbors:
                            # Relation type 2 for spatial neighbors
                            edge_list.append([i, int(j.item()), 2])

            if edge_list:
                edge_list_tensor = torch.tensor(edge_list, dtype=torch.long, device=device)
            else:
                # If no edges (very short sequence), create self loops to avoid empty graph
                edge_list_tensor = torch.stack([
                    torch.arange(seq_len, device=device),
                    torch.arange(seq_len, device=device),
                    torch.zeros(seq_len, dtype=torch.long, device=device)  # relation type 0
                ], dim=1)

            graph = data.Graph(
                edge_list=edge_list_tensor,
                num_node=seq_len,
                num_relation=7,  # Important: Match the num_relation expected by GearNet
                node_feature=node_pos  # Use CA coordinates as node features directly
            )
            graphs.append(graph)

        # Batch the graphs
        if len(graphs) > 1:
            batched_graph = data.graph_collate(graphs)
        else:
            batched_graph = graphs[0]

        # Ensure the batched graph is on the correct device
        batched_graph = batched_graph.to(device)

        # Pass through GearNet model in float32 mode
        with torch.cuda.amp.autocast(enabled=False):  # Disable autocast for this forward pass
            output = self.gearnet_model(batched_graph, batched_graph.node_feature)

        # output["node_feature"] is (total_num_residues, hidden_dim)
        node_embeddings = output["node_feature"]

        # The output should have the right size for reshaping
        expected_nodes = batch_size * seq_len

        if node_embeddings.size(0) != expected_nodes:
            raise RuntimeError(
                f"GearNet output size mismatch: got {node_embeddings.size(0)}, "
                f"expected {expected_nodes} (batch_size={batch_size}, seq_len={seq_len})"
            )

        # Reshape to expected format: (batch_size, seq_len, hidden_dim)
        final_embeddings = node_embeddings.view(batch_size, seq_len, self.hidden_dim)

        # If the original input was half, convert the output back to half
        if is_half:
            final_embeddings = final_embeddings.to(dtype=ca_coords.dtype)

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