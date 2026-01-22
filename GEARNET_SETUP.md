# GearNet Integration with Dimension Projection

This document explains how GearNet is integrated into the training pipeline with proper dimension handling following the methodology from **Chen et al. (2025) "Structure-Aligned Protein Language Model"**.

## Problem

Protein Language Models (PLMs) like ESM2 and Graph Neural Networks (GNNs) like GearNet may output embeddings with different dimensions:
- **ESM2-8M**: 320 dimensions
- **ESM2-35M**: 480 dimensions
- **ESM2-150M**: 640 dimensions
- **ESM2-650M**: 1280 dimensions
- **GearNet**: Can be configured to any dimension (typically 512)

## Solution: Projection Layers

Following Chen et al. (2025), we use **separate projection layers** to map embeddings to a shared space:

```
pLM embedding (D_a dimensions) ---W_a---> Shared space (D dimensions)
pGNN embedding (D_g dimensions) --W_g---> Shared space (D dimensions)
```

Where:
- **W_a ∈ ℝ^(D_a×D)**: Projects pLM embeddings from dimension D_a to shared dimension D
- **W_g ∈ ℝ^(D_g×D)**: Projects pGNN embeddings from dimension D_g to shared dimension D
- **D**: Shared projection dimension (default: 512)

This allows the PLM and GNN to have **different output dimensions** while still computing alignment losses.

## Implementation

### 1. StructureAlignmentLoss

The `StructureAlignmentLoss` module now accepts separate dimensions:

```python
structure_alignment_loss = StructureAlignmentLoss(
    hidden_dim=1280,              # ESM2-650M dimension
    pgnn_hidden_dim=512,          # GearNet dimension
    shared_projection_dim=512,    # Shared space dimension
    num_structural_classes=21,
    latent_weight=0.5,
    physical_weight=0.5
)
```

### 2. PretrainedGNNWrapper

The wrapper now exposes its output dimension via the `output_dim` property:

```python
frozen_gnn = PretrainedGNNWrapper(
    hidden_dim=512,               # GearNet hidden dimension
    use_simple_encoder=False      # Set to True for fallback encoder
)

print(f"GNN output dimension: {frozen_gnn.output_dim}")
```

### 3. Training Script Usage

The training scripts automatically handle dimension mismatches:

```python
# 1. Create GNN module first
frozen_gnn = PretrainedGNNWrapper(
    hidden_dim=512,
    use_simple_encoder=False  # Use GearNet
).to(device)

# 2. Create alignment loss with separate dimensions
structure_alignment_loss = StructureAlignmentLoss(
    hidden_dim=esm_hidden_size,           # From ESM2 model
    pgnn_hidden_dim=frozen_gnn.output_dim,  # From GearNet
    shared_projection_dim=512
).to(device)
```

## Enabling GearNet

To enable GearNet (instead of the simple encoder fallback):

### Option 1: Modify Training Script

In `scripts/train.py` or `scripts/train_constrained.py`:

```python
frozen_gnn = PretrainedGNNWrapper(
    hidden_dim=512,
    use_simple_encoder=False  # Change from True to False
).to(device)
```

### Option 2: Generate GearNet Embeddings

Pre-compute GearNet embeddings with the desired dimension:

```bash
python data_pipeline/generate_gearnet_embeddings.py \
    --processed_dataset_path data/processed \
    --output_dir data/gearnet_embeddings \
    --hidden_dim 512 \
    --chunk_size 50
```

**Important**: If using pre-computed embeddings, the `hidden_dim` used during embedding generation must match the `hidden_dim` passed to `PretrainedGNNWrapper`.

## Configuration Examples

### Example 1: ESM2-650M + GearNet-512

```python
# ESM2-650M has hidden_size = 1280
# GearNet configured with hidden_dim = 512

frozen_gnn = PretrainedGNNWrapper(hidden_dim=512, use_simple_encoder=False)
structure_alignment_loss = StructureAlignmentLoss(
    hidden_dim=1280,              # ESM2-650M
    pgnn_hidden_dim=512,          # GearNet
    shared_projection_dim=512
)
```

Projection layers:
- `pLM_projection`: Linear(1280, 512)
- `pGNN_projection`: Linear(512, 512)

### Example 2: ESM2-150M + GearNet-512

```python
# ESM2-150M has hidden_size = 640
# GearNet configured with hidden_dim = 512

frozen_gnn = PretrainedGNNWrapper(hidden_dim=512, use_simple_encoder=False)
structure_alignment_loss = StructureAlignmentLoss(
    hidden_dim=640,               # ESM2-150M
    pgnn_hidden_dim=512,          # GearNet
    shared_projection_dim=512
)
```

Projection layers:
- `pLM_projection`: Linear(640, 512)
- `pGNN_projection`: Linear(512, 512)

### Example 3: Matching Dimensions

```python
# ESM2-650M has hidden_size = 1280
# GearNet configured with hidden_dim = 1280 (to match)

frozen_gnn = PretrainedGNNWrapper(hidden_dim=1280, use_simple_encoder=False)
structure_alignment_loss = StructureAlignmentLoss(
    hidden_dim=1280,              # ESM2-650M
    pgnn_hidden_dim=1280,         # GearNet (matching)
    shared_projection_dim=512
)
```

Projection layers:
- `pLM_projection`: Linear(1280, 512)
- `pGNN_projection`: Linear(1280, 512)

## Benefits

1. **Flexibility**: PLM and GNN can use their optimal dimensions independently
2. **No Pre-processing Mismatch**: GearNet embeddings don't need to match ESM2 dimensions
3. **Follows SOTA**: Implementation matches Chen et al. (2025) methodology
4. **Backward Compatible**: If dimensions match, behaves identically to original implementation

## Troubleshooting

### Error: "size mismatch for pGNN_projection.weight"

This error occurs when loading a checkpoint that was trained with different GNN dimensions. Solutions:

1. **Retrain from scratch** with the new dimensions
2. **Remove projection layers** from checkpoint and reinitialize them:
   ```python
   state_dict = torch.load(checkpoint)
   # Remove old projection layers
   state_dict = {k: v for k, v in state_dict.items()
                 if 'pGNN_projection' not in k and 'pLM_projection' not in k}
   model.load_state_dict(state_dict, strict=False)
   ```

### TorchDrug Hanging Issue

If GearNet hangs during forward pass (known issue with some TorchDrug versions):

1. Set `use_simple_encoder=True` as fallback
2. See `TORCHDRUG_ISSUE.md` for debugging steps
3. Check TorchDrug and PyTorch version compatibility

## References

- Chen et al. (2025). "Structure-Aligned Protein Language Model". arXiv:2505.16896
- Zhang et al. (2023). "Protein Representation Learning by Geometric Structure Pretraining" (GearNet)

## Key Equation from Chen et al. (2025)

Similarity score between residue i from protein b1 and residue j from protein b2:

```
δ(i, b1, j, b2) = s · (pLM(a_b1; θ)_i W_a^⊤) (pGNN(g_b2)_j W_g)
```

Where:
- `pLM(a_b1; θ)_i ∈ ℝ^D_a`: pLM embedding for residue i
- `pGNN(g_b2)_j ∈ ℝ^D_g`: pGNN embedding for residue j
- `W_a ∈ ℝ^(D_a×D)`: pLM projection matrix
- `W_g ∈ ℝ^(D_g×D)`: pGNN projection matrix
- `s`: Learnable temperature parameter
- `D`: Shared projection dimension
