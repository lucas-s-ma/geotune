# TorchDrug GearNet Hanging Issue

## Problem Summary

**Date Discovered:** 2025-12-12
**Status:** UNRESOLVED - Using simple structural encoder as workaround
**Severity:** Critical - TorchDrug GearNet hangs indefinitely during forward pass

## Symptoms

1. **Minimal test hangs:** Even a 10-node graph with TorchDrug's `GeometryAwareRelationalGraphNeuralNetwork` hangs
2. **Hangs on both GPU and CPU:** Not a CUDA-specific issue
3. **Hangs inside `model.forward()`:** The call enters TorchDrug code but never returns
4. **No error messages:** Just infinite hang

## Environment Details

- **Python:** 3.10
- **TorchDrug:** 0.2.1
- **PyTorch:** (check with `python -c "import torch; print(torch.__version__)"`)
- **CUDA:** H200 GPU (but also hangs on CPU)
- **Platform:** Linux HPC cluster

## Minimal Reproduction

See `test_gearnet_minimal.py` (if exists) or create:

```python
import torch
from torchdrug import data
from torchdrug.models.gearnet import GeometryAwareRelationalGraphNeuralNetwork

model = GeometryAwareRelationalGraphNeuralNetwork(
    input_dim=3,
    hidden_dims=[320, 320, 320, 320],
    num_relation=7,
    batch_norm=True,
    short_cut=True,
    concat_hidden=False
)

# Create tiny 10-node graph
seq_len = 10
edge_list = [[i, i+1, 0] for i in range(seq_len-1)]
edge_list_tensor = torch.tensor(edge_list, dtype=torch.long)
node_features = torch.randn(seq_len, 3)

graph = data.Graph(
    edge_list=edge_list_tensor,
    num_node=seq_len,
    num_relation=7,
    node_feature=node_features
)

# THIS HANGS:
output = model(graph, graph.node_feature)
```

## Attempted Fixes (None Worked)

1. ✗ Reinstalling TorchDrug
2. ✗ Using CPU instead of GPU
3. ✗ Different graph sizes (tried 10, 50, 100 nodes)
4. ✗ Adding `num_relation=7` to graph constructor

## Current Workaround

**Using SimpleStructuralEncoder** (see `models/simple_structural_encoder.py`):
- Pure PyTorch implementation (no TorchDrug dependency)
- Uses k-NN distances + CA coordinates
- Fast: ~0.1-0.5s per protein vs 1s+ for GearNet
- Enabled by setting `use_simple_encoder=True` in `PretrainedGNNWrapper`

## How to Fix (When You Return to This)

### Option 1: Fix PyTorch Version Compatibility

TorchDrug 0.2.1 may require specific PyTorch versions:

```bash
# Check current PyTorch version
python -c "import torch; print(torch.__version__)"

# Try PyTorch 1.13 (known to work with TorchDrug 0.2.1)
pip install torch==1.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117

# Test
python -c "
from torchdrug.models.gearnet import GeometryAwareRelationalGraphNeuralNetwork
print('Import successful')
"
```

### Option 2: Try Development Version of TorchDrug

```bash
pip uninstall torchdrug -y
pip install git+https://github.com/DeepGraphLearning/torchdrug

# Test with minimal script
```

### Option 3: Check for Known Issues

Visit:
- https://github.com/DeepGraphLearning/torchdrug/issues
- Search for "hang" or "freeze" issues
- Check if H200 GPU has specific issues

### Option 4: Use Alternative GNN Library

Consider replacing TorchDrug with:
- **PyTorch Geometric (PyG):** More actively maintained
- **DGL (Deep Graph Library):** Better performance
- **Custom GNN:** Implement geometric GNN from scratch

## Files Modified for Workaround

1. **models/simple_structural_encoder.py:** Created simple encoder
2. **utils/structure_alignment_utils.py:** Added `use_simple_encoder` parameter
3. **data_pipeline/generate_gearnet_embeddings.py:** Added `use_simple=True` flag
4. **models/gearnet_model.py:** Added extensive debugging (can be removed when fixed)

## How to Switch Back to GearNet (When Fixed)

1. **In `data_pipeline/generate_gearnet_embeddings.py`:**
   ```python
   use_simple = False  # Change from True to False
   ```

2. **Regenerate embeddings:**
   ```bash
   python data_pipeline/generate_gearnet_embeddings.py \
       --processed_dataset_path data/processed \
       --output_dir data/processed/embeddings_gearnet \
       --hidden_dim 320
   ```

3. **Update config to use new embeddings:**
   ```yaml
   data:
     data_path: "data/processed_gearnet"  # Point to new embeddings
   ```

4. **In training, set:**
   ```python
   use_structure_alignment = True
   frozen_gnn = PretrainedGNNWrapper(hidden_dim=esm_hidden_size, use_simple_encoder=False)
   ```

## Notes

- The simple encoder is actually **faster** and may give comparable results
- Consider running ablation study comparing simple encoder vs GearNet when fixed
- Original code worked at ~1 sec/protein before modifications, suggesting environment change
