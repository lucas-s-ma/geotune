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

        # Edge construction layer
        self.edge_construction = geometry.EdgeConstruction(
            knn=10, num_relation=7, radius_cutoff=None
        )

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
        batch_size, seq_len, _ = ca_coords.shape
        device = ca_coords.device

        # We will build a graph based on CA atoms for simplicity, as this is common.
        # The nodes of the graph are the residues (represented by CA atoms).
        
        graphs = []
        for b in range(batch_size):
            # Use CA coordinates as node positions
            node_pos = ca_coords[b] # (seq_len, 3)
            
            # Create a graph structure for a single protein
            graph = data.Graph(
                num_node=seq_len,
                node_position=node_pos
            )
            graphs.append(graph)

        # Batch the graphs
        if len(graphs) > 1:
            batched_graph = data.graph_collate(graphs)
        else:
            batched_graph = graphs[0]
            
        batched_graph = batched_graph.to(device)

        # Construct edges and relations
        batched_graph = self.edge_construction(batched_graph)

        # Project coordinates to hidden dimension for node features
        node_features = self.coord_projection(batched_graph.node_position)

        # Pass through GearNet model
        output = self.gearnet_model(batched_graph, node_features)
        
        # output["node_feature"] is (total_num_residues, hidden_dim)
        node_embeddings = output["node_feature"]
        
        # Reshape to (batch_size, seq_len, hidden_dim)
        final_embeddings = node_embeddings.view(batch_size, seq_len, self.hidden_dim)

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