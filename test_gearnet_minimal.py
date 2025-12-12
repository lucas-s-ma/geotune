#!/usr/bin/env python
"""
Minimal test to isolate TorchDrug GearNet hanging issue
"""
import torch
from torchdrug import data
from torchdrug.models.gearnet import GeometryAwareRelationalGraphNeuralNetwork
import time

print("Creating minimal GearNet model...")
model = GeometryAwareRelationalGraphNeuralNetwork(
    input_dim=3,
    hidden_dims=[320, 320, 320, 320],
    num_relation=7,
    batch_norm=True,
    short_cut=True,
    concat_hidden=False
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Using device: {device}")

# Create a tiny test graph
print("\nCreating tiny test graph...")
seq_len = 10
edge_list = []
for i in range(seq_len):
    for offset in [-1, 1]:
        j = i + offset
        if 0 <= j < seq_len:
            edge_list.append([i, j, 0])

edge_list_tensor = torch.tensor(edge_list, dtype=torch.long, device=device)
node_features = torch.randn(seq_len, 3, device=device)

graph = data.Graph(
    edge_list=edge_list_tensor,
    num_node=seq_len,
    num_relation=7,
    node_feature=node_features
)

print(f"Graph created: {graph.num_node} nodes, {graph.num_edge} edges")

# Test forward pass
print("\nTesting forward pass...")
print("Calling model.forward() - THIS IS WHERE IT MIGHT HANG")
start = time.time()

with torch.no_grad():
    try:
        output = model(graph, graph.node_feature)
        elapsed = time.time() - start
        print(f"✓ Forward pass completed in {elapsed:.3f}s")
        print(f"Output shape: {output['node_feature'].shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

print("\nTest complete!")
