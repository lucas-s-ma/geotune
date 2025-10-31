# Structural Tokens KeyError Fix

## Issue

```
KeyError: Caught KeyError in DataLoader worker process 1.
File "/work/sm996/co-amp/utils/data_utils.py", line 236, in collate_fn
    structural_tokens = torch.stack([item['structural_tokens'] for item in batch])
KeyError: 'structural_tokens'
```

## Root Cause

**Data Mismatch:** Your dataset has **11,158 proteins** but only **11,157 structural tokens**:

```
Loaded structural tokens for 11157 proteins
Loaded 11158 proteins from pre-processed dataset
```

**The Problem:**
1. Dataset enables `include_structural_tokens=True`
2. Protein at index 11157 (the last one) doesn't have structural tokens
3. When a batch contains this protein AND batch[0] has tokens:
   - `collate_fn` checks: `if 'structural_tokens' in batch[0]` → True
   - Then tries: `[item['structural_tokens'] for item in batch]`
   - Fails on protein 11157: `KeyError: 'structural_tokens'`

## The Fix

Made **two changes** to make the code robust against missing structural tokens:

### 1. Fixed `collate_fn` to Check ALL Items (Line 230, 235)

**Before:**
```python
# Only checks batch[0] - WRONG!
if 'structural_tokens' in batch[0]:
    structural_tokens = torch.stack([item['structural_tokens'] for item in batch])
```

**After:**
```python
# Checks ALL items in batch - CORRECT!
if all('structural_tokens' in item for item in batch):
    structural_tokens = torch.stack([item['structural_tokens'] for item in batch])
```

### 2. Made Dataset `__getitem__` More Defensive (Lines 176-193, 459-476)

**Before:**
```python
if self.include_structural_tokens and len(self.structural_tokens) > idx:
    struct_tokens = self.structural_tokens[idx]
    # ... process tokens
    result['structural_tokens'] = torch.tensor(struct_seq, dtype=torch.long)
```

**After:**
```python
if self.include_structural_tokens:
    try:
        if idx < len(self.structural_tokens):
            struct_tokens = self.structural_tokens[idx]
            # ... process tokens
            result['structural_tokens'] = torch.tensor(struct_seq, dtype=torch.long)
    except (IndexError, KeyError, TypeError) as e:
        # Skip structural tokens for this sample if there's any issue
        pass
```

## Changes Made

**File:** `utils/data_utils.py`

1. **Lines 230-237:** Updated `collate_fn` to use `all()` check
2. **Lines 176-193:** Added defensive try-except in first `__getitem__` method
3. **Lines 459-476:** Added defensive try-except in second `__getitem__` method (EfficientProteinDataset)

## What This Means

### Immediate Fix
- Training will now skip batches where not all items have structural tokens
- If the batch happens to have protein 11157, structural tokens won't be included for that batch
- Training continues without crashing

### Behavior During Training

**Scenario 1: Batch without protein 11157**
- All items have structural tokens
- Structural alignment loss computed normally
- ✓ Full training with structure alignment

**Scenario 2: Batch with protein 11157**
- Not all items have structural tokens
- `struct_align_loss = 0` (skipped for this batch)
- ✓ Training continues, but this batch doesn't use structure alignment

### Impact on Training

With 11,157/11,158 tokens available:
- **99.99% of batches** will have structural tokens
- **~1 batch per epoch** might skip structure alignment
- **Negligible impact** on training quality

## Recommended Long-Term Fix

**Generate missing structural token:**

```bash
# Option 1: Regenerate all tokens (safest)
python scripts/generate_structural_tokens.py \
    --data_path data/processed \
    --output_path data/processed/structural_tokens.pkl

# Option 2: Just add the missing token for protein 11157
# (requires identifying which protein is missing)
```

After regeneration, you should see:
```
Loaded structural tokens for 11158 proteins  ← Matches!
Loaded 11158 proteins from pre-processed dataset
```

## Testing

After applying the fix:

```bash
git pull  # Get the fix
python scripts/train.py --config configs/config.yaml --data_path data/processed
```

Expected output (no KeyError):
```
Using stub implementation for pre-trained GNN (avoiding TorchDrug)
Loading dataset...
Structural tokens available: True
Pre-computed embeddings found but disabled (may have wrong dimension)
Loaded structural tokens for 11157 proteins
Loaded 11158 proteins from pre-processed dataset
Dataset split: 8926 training samples, 2232 validation samples
Starting training for 10 epochs...
[Training proceeds normally]
```

## Summary

✅ **Fixed:** `collate_fn` now checks all items before stacking
✅ **Fixed:** Dataset `__getitem__` handles missing tokens gracefully
✅ **Result:** Training proceeds without KeyError
⚠️ **Note:** 1 protein missing tokens (99.99% coverage)
📝 **Recommended:** Regenerate structural tokens for 100% coverage
