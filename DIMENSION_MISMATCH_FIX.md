# Dimension Mismatch Fix

## Error Summary

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1024x512 and 640x512)
```

This error occurred in the structure alignment loss calculation where:
- GNN embeddings had shape `(batch_size * seq_len, 512)` = `(1024, 512)`
- Expected input dimension was `640` (ESM2-150M hidden size)
- The projection layer expected `(640, 512)` weight matrix but received `(512)` input

## Root Causes

### 1. Wrong Parameter Order in Stub Initialization
**File:** `utils/structure_alignment_utils.py:210`

**Problem:**
```python
return StubGearNetWrapper(hidden_dim, freeze)  # Positional args in wrong order
```

The `StubGearNetWrapper` signature is:
```python
def __init__(self, hidden_dim=512, num_layers=4, freeze=True):
```

When called with `StubGearNetWrapper(640, True)`, the parameters were interpreted as:
- `hidden_dim = 640` ✓
- `num_layers = True` (converted to 1) ✗
- `freeze = True` (default) ✗

**Fix:**
```python
return StubGearNetWrapper(hidden_dim=hidden_dim, freeze=freeze)  # Named args
```

### 2. Hardcoded Output Dimension
**File:** `utils/structure_alignment_utils.py:255`

**Problem:**
```python
self.projection = nn.Linear(hidden_dim, 768)  # Hardcoded to 768!
```

This hardcoded the output to 768 dimensions, which is the hidden size for ESM-1b or ESM2-650M, but not for ESM2-150M (640) or ESM2-35M (480).

**Fix:**
Removed the projection layer entirely. The stub now outputs `hidden_dim` directly, which matches the ESM model's hidden dimension. The `StructureAlignmentLoss` module handles the necessary projections.

### 3. Deprecated API Warning
**File:** `scripts/train.py:119, 158`

**Problem:**
```python
with torch.cuda.amp.autocast(enabled=use_amp):
```

PyTorch deprecated `torch.cuda.amp.autocast` in favor of `torch.amp.autocast('cuda', ...)`.

**Fix:**
```python
with torch.amp.autocast('cuda', enabled=use_amp):
```

## Changes Made

### File: `utils/structure_alignment_utils.py`

1. **Line 210:** Fixed stub initialization to use named parameters
2. **Lines 231-261:** Removed hardcoded projection layer from `StubGearNetWrapper.__init__`
3. **Lines 276-291:** Updated forward method to return embeddings directly without projection

### File: `scripts/train.py`

1. **Line 119:** Updated to `torch.amp.autocast('cuda', enabled=use_amp)`
2. **Line 158:** Updated to `torch.amp.autocast('cuda', enabled=use_amp)`

## Verification

The GNN stub will now:
1. Accept `hidden_dim` parameter matching ESM model (e.g., 640 for ESM2-150M)
2. Output embeddings with shape `(batch_size, seq_len, hidden_dim)`
3. Match the expected input dimension for `StructureAlignmentLoss` projections

## Model-Specific Hidden Dimensions

For reference:
- ESM2-8M: 320
- ESM2-35M: 480
- ESM2-150M: 640
- ESM2-650M: 1280
- ESM2-3B: 2560
- ESM2-15B: 5120

The code now automatically adapts to any ESM model size by using the model's actual `hidden_dim`.

## Testing

After these fixes, the training should proceed without dimension mismatch errors. The stub GNN will output the correct dimension matching your ESM model.
