# TorchDrug Import and Embedding Dimension Fixes

## Issues Fixed

### Issue 1: TorchDrug Still Importing Despite `use_gearnet_stub=True`

**Problem:**
Even when calling `PretrainedGNNWrapper(use_gearnet_stub=True)`, the code was still trying to import TorchDrug, causing NumPy 2.x compatibility errors:

```
AttributeError: _ARRAY_API not found
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```

**Root Cause:**
The `PretrainedGNNWrapper.__init__` method was checking the import BEFORE checking the `use_gearnet_stub` flag:

```python
# OLD CODE - WRONG ORDER
def __init__(self, ..., use_gearnet_stub=False):
    try:
        from models.gearnet_model import create_pretrained_gearnet  # Imports TorchDrug!
        self.backbone = create_pretrained_gearnet(...)
    except (ImportError, AttributeError) as e:
        if use_gearnet_stub:  # Too late, import already failed!
            self.backbone = self._create_stub_gearnet(...)
```

**Fix:**
Check `use_gearnet_stub` FIRST before attempting any imports:

```python
# NEW CODE - CORRECT ORDER
def __init__(self, ..., use_gearnet_stub=False):
    if use_gearnet_stub:
        # Use stub - no TorchDrug import needed
        self.backbone = self._create_stub_gearnet(hidden_dim, freeze)
        print("Using stub implementation (avoiding TorchDrug)")
    else:
        # Only try to import TorchDrug if not using stub
        try:
            from models.gearnet_model import create_pretrained_gearnet
            ...
```

**File:** `utils/structure_alignment_utils.py:174-202`

---

### Issue 2: Pre-computed Embeddings Have Wrong Dimension

**Problem:**
```
Pre-computed embeddings shape: torch.Size([2, 512, 512])
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1024x512 and 640x512)
```

The pre-computed embeddings were generated with `hidden_dim=512`, but ESM2-150M uses `hidden_dim=640`.

**Root Cause:**
The embeddings were pre-computed with a different configuration or model. Each ESM model has a different hidden dimension:

- ESM2-8M: 320
- ESM2-35M: 480
- **ESM2-150M: 640** ← Your model
- ESM2-650M: 1280
- ESM2-3B: 2560

**Fix:**
Disabled loading pre-computed embeddings and generate them on-the-fly with the correct dimension:

```python
# Force disable loading pre-computed embeddings
load_embeddings = False  # Generate on-the-fly with correct dimension
```

**File:** `scripts/train.py:549`

**Future Solution:**
To use pre-computed embeddings, you need to regenerate them with `hidden_dim=640`:

```bash
# Regenerate embeddings with correct dimension
python scripts/precompute_embeddings.py \
    --data_path data/processed \
    --model_name facebook/esm2_t30_150M_UR50D \
    --hidden_dim 640
```

---

### Issue 3: Deprecated PyTorch API

**Problem:**
```
FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated.
Please use `torch.amp.GradScaler('cuda', args...)` instead.
```

**Fix:**
Updated to new PyTorch 2.x API:

```python
# OLD
scaler = torch.cuda.amp.GradScaler()

# NEW
scaler = torch.amp.GradScaler('cuda')
```

**Files:**
- `scripts/train.py:619`
- `scripts/train.py:119, 158` (autocast calls)

---

## Changes Summary

### File: `utils/structure_alignment_utils.py`

**Lines 174-202:** Reordered logic to check `use_gearnet_stub` FIRST:
```python
if use_gearnet_stub:
    # Use stub (avoids TorchDrug import entirely)
    self.backbone = self._create_stub_gearnet(hidden_dim, freeze)
else:
    # Only import TorchDrug if not using stub
    try:
        from models.gearnet_model import create_pretrained_gearnet
        ...
```

### File: `scripts/train.py`

**Lines 549-555:** Disabled pre-computed embeddings:
```python
load_embeddings = False  # Generate on-the-fly with correct dimension
if embeddings_exist:
    print(f"Pre-computed embeddings found but disabled (may have wrong dimension)")
    print(f"Embeddings will be generated on-the-fly with hidden_dim={esm_hidden_size}")
```

**Line 619:** Fixed GradScaler API:
```python
scaler = torch.amp.GradScaler('cuda')
```

**Lines 119, 158:** Fixed autocast API:
```python
with torch.amp.autocast('cuda', enabled=use_amp):
```

---

## Testing

After these fixes, your training should:

1. **NOT import TorchDrug** when using `use_gearnet_stub=True`
2. **NOT load incompatible pre-computed embeddings**
3. **Generate embeddings on-the-fly** with correct dimension (640 for ESM2-150M)
4. **NOT show deprecation warnings** for PyTorch APIs

Expected output:
```
Loading model...
Gradient checkpointing enabled
Trainable parameters: 3,001,280 (1.99% of total)
Using stub implementation for pre-trained GNN (avoiding TorchDrug)
Loading dataset...
Pre-computed embeddings found but disabled (may have wrong dimension)
Embeddings will be generated on-the-fly with hidden_dim=640
Mixed precision training enabled
Starting training for 10 epochs...
```

---

## Performance Notes

**Generating embeddings on-the-fly vs pre-computed:**

| Method | Speed | Memory | Disk Usage |
|--------|-------|--------|------------|
| Pre-computed (correct dim) | Fastest | Low | ~5-10 GB |
| On-the-fly generation | Slower (~20-30% overhead) | Medium | None |

If training speed is critical, regenerate the embeddings with the correct dimension and enable loading them again.

---

## If You Still See TorchDrug Errors

If you still see TorchDrug import errors, it means your code on the SLURM server hasn't been updated. Make sure to:

1. Pull the latest changes: `git pull`
2. Clear Python cache: `find . -type d -name __pycache__ -exec rm -rf {} +`
3. Verify the fix: `grep -A 5 "if use_gearnet_stub:" utils/structure_alignment_utils.py`

You should see the stub check happening FIRST, before any import attempts.
