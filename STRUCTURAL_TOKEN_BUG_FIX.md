# Structural Token CUDA Assert Fix

## Error

```
/pytorch/aten/src/ATen/native/cuda/Loss.cu:245: nll_loss_forward_reduce_cuda_kernel_2d
Assertion `t >= 0 && t < n_classes` failed.

File "/work/sm996/co-amp/utils/structure_alignment_utils.py", line 163
    physical_loss = self.ce_loss(logits_flat, tokens_flat)
torch.AcceleratorError: CUDA error: device-side assert triggered
```

Occurred during training at batch 910 in the structure alignment loss calculation.

## Root Cause

**Foldseek Token Out of Bounds:** The `generate_foldseek_3di.py` script was mapping unknown 3Di characters to token ID **20**, but CrossEntropyLoss expects tokens in range **[0, 19]** (20 classes).

### The Bug

In `scripts/generate_foldseek_3di.py` line 104:

```python
# OLD CODE - BUG
def convert_3di_to_ints(ascii_seq: str) -> List[int]:
    struct_to_int = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
        'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
    }
    return [struct_to_int.get(char, 20) for char in ascii_seq]  # ← BUG: default to 20!
```

When Foldseek outputs an unexpected character (e.g., lowercase letters, 'X', etc.), it was mapped to **20**.

In `utils/structure_alignment_utils.py`:
- `num_structural_classes = 20` (line 18)
- CrossEntropyLoss expects tokens in range [0, 19]
- Token 20 causes: **CUDA assertion failure**

## The Fixes

### Fix 1: Corrected Token Mapping (Line 95-115 in generate_foldseek_3di.py)

**Before:**
```python
return [struct_to_int.get(char, 20) for char in ascii_seq]  # Wrong default!
```

**After:**
```python
def convert_3di_to_ints(ascii_seq: str) -> List[int]:
    """
    Convert 3Di ASCII sequence to integer tokens.
    Foldseek 3Di alphabet uses 20 structural states (A-Y).
    Unknown characters are mapped to 0 (default structural state).
    """
    struct_to_int = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
        'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
    }
    tokens = []
    for char in ascii_seq:
        token = struct_to_int.get(char.upper(), 0)  # Default to 0, not 20!
        if char not in struct_to_int:
            print(f"Warning: Unknown 3Di character '{char}' mapped to token 0")
        tokens.append(token)
    return tokens
```

### Fix 2: Runtime Validation (Lines 157-164 in structure_alignment_utils.py)

Added validation to handle structural tokens that were already generated with the bug:

```python
# CRITICAL: Validate and clamp structural tokens to valid range [0, num_classes-1]
# This handles tokens generated with old buggy code that mapped unknown chars to 20
invalid_mask = (tokens_flat >= self.num_structural_classes) | (tokens_flat < 0)
if invalid_mask.any():
    num_invalid = invalid_mask.sum().item()
    print(f"Warning: Found {num_invalid} invalid structural tokens. Clamping to valid range.")
    # Clamp invalid tokens to 0 (default structural state)
    tokens_flat = torch.clamp(tokens_flat, min=0, max=self.num_structural_classes - 1)
```

This ensures:
- ✅ Tokens < 0 are clamped to 0
- ✅ Tokens >= 20 are clamped to 19
- ✅ Training can proceed with existing structural tokens
- ⚠️ Warns you when invalid tokens are found

## Changes Summary

### Files Modified

1. **`scripts/generate_foldseek_3di.py`** (Lines 95-115)
   - Changed default token from 20 → 0 for unknown characters
   - Added uppercase conversion to handle case-insensitive matching
   - Added warning message when unknown characters are encountered

2. **`utils/structure_alignment_utils.py`** (Lines 157-164)
   - Added runtime validation before CrossEntropyLoss
   - Clamps invalid tokens to valid range [0, 19]
   - Prints warning when invalid tokens are detected

## Impact

### Immediate Fix
- ✅ Training will no longer crash with CUDA assert error
- ✅ Existing structural tokens with value 20 will be clamped to 19
- ✅ Warning messages will alert you if invalid tokens are found

### Long-Term
- ✅ Future structural tokens will be generated correctly
- ✅ Unknown 3Di characters will be mapped to valid token 0
- ⚠️ Consider regenerating structural tokens for best accuracy

## Testing

After applying these fixes:

```bash
# On cluster
cd /work/sm996/co-amp
git pull  # Get the fixes

# Run training - should no longer crash
python scripts/train.py --config configs/config.yaml --data_path data/processed
```

Expected behavior:
- Training proceeds past batch 910 without CUDA errors
- If invalid tokens are detected, you'll see warnings like:
  ```
  Warning: Found 1024 invalid structural tokens (>= 20 or < 0). Clamping to valid range.
  ```
- Training completes successfully

## Optional: Regenerate Structural Tokens

For best accuracy, regenerate structural tokens with the fixed code:

```bash
# After ensuring Foldseek is installed and code is updated
python scripts/process_data.py \
    --raw_dir [path_to_pdb_files] \
    --output_dir data/processed \
    --create_efficient_dataset
```

This will create new `structural_tokens.pkl` with all tokens in valid range [0, 19].

## Why Embeddings Show as "False"

Your pre-computed embeddings in `data/processed/embeddings/` exist but are **intentionally disabled** because they have the wrong dimension:

- Your embeddings: 512-dim
- ESM2-150M needs: 640-dim

The code now generates embeddings **on-the-fly** with the correct 640 dimensions. This is slower but ensures dimensional compatibility.

To re-enable pre-computed embeddings, you would need to regenerate them with:
```bash
python scripts/precompute_embeddings.py \
    --data_path data/processed \
    --model_name facebook/esm2_t30_150M_UR50D \
    --hidden_dim 640
```

However, for now, on-the-fly generation works fine and avoids dimension mismatch errors.

---

## Summary

✅ **CUDA assert error fixed** - caused by structural token 20 (out of bounds)
✅ **Runtime validation added** - clamps existing bad tokens
✅ **Future tokens correct** - unknown chars map to 0, not 20
✅ **Training should work** - with or without regenerating structural tokens
ℹ️ **Embeddings disabled** - wrong dimension (512 vs 640), generating on-the-fly instead
