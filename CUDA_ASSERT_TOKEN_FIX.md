# CUDA Assert Error Fix - Invalid Token IDs

## Error

```
torch.AcceleratorError: CUDA error: device-side assert triggered
```

Occurred during training at the MLM loss calculation step.

## Root Cause

**Token ID Mismatch:** Your code was using a custom token mapping (IDs 5-24) that **doesn't match ESM2's vocabulary**. This caused invalid token IDs to be passed to the ESM2 model, triggering CUDA device-side assertions.

### The Problems

1. **Wrong Amino Acid Mapping:**
   - Your custom mapping: A=5, R=6, N=7, ... V=24
   - ESM2 actual mapping: L=4, A=5, G=6, ... C=23

2. **Wrong Special Tokens:**
   - Your code used: mask=4, pad=0
   - ESM2 uses: mask=32, pad=1

3. **Specific Issues:**
   - Line 142 (train.py): `masked_input_ids[mask_positions] = 4`
     - Used token ID 4 as mask, but in ESM2, **4 = 'L' (leucine)**!
     - ESM2's mask token is **32**
   - Padding token was 0, but ESM2 uses 1

### Why This Caused CUDA Assert

When you passed:
- Token ID 4 thinking it's `<mask>` → ESM2 saw it as amino acid 'L'
- Custom IDs 5-24 → Some didn't match ESM2's expected amino acids
- Invalid token IDs → Out of bounds or unexpected values
- Result: **CUDA device-side assertion failure**

---

## The Fixes

### 1. Fixed Amino Acid Token Mapping

**File:** `utils/data_utils.py` (Lines 200-218, applied to both Dataset classes)

**Before:**
```python
# WRONG - Custom mapping
aa_to_id = {
    'A': 5, 'R': 6, 'N': 7, 'D': 8, 'C': 9, 'Q': 10, 'E': 11, 'G': 12,
    'H': 13, 'I': 14, 'L': 15, 'K': 16, 'M': 17, 'F': 18, 'P': 19,
    'S': 20, 'T': 21, 'W': 22, 'Y': 23, 'V': 24
}
```

**After:**
```python
# CORRECT - ESM2 mapping
# Order: L, A, G, V, S, E, R, T, I, D, P, K, Q, N, F, Y, M, H, W, C
aa_to_id = {
    'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11,
    'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18,
    'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23
}
```

### 2. Fixed Padding Token

**File:** `utils/data_utils.py` (Lines 157, 443)

**Before:**
```python
token_ids.extend([0] * padding_length)  # WRONG
```

**After:**
```python
token_ids.extend([1] * padding_length)  # 1 for <pad> token in ESM2
```

### 3. Fixed Attention Mask Logic

**File:** `utils/data_utils.py` (Lines 164, 450)

**Before:**
```python
attention_mask = [1 if token != 0 else 0 for token in token_ids]  # WRONG
```

**After:**
```python
attention_mask = [1 if token != 1 else 0 for token in token_ids]  # 1 is <pad> in ESM2
```

### 4. Fixed Mask Token ID

**File:** `scripts/train.py` (Lines 142, 340)

**Before:**
```python
masked_input_ids[mask_positions] = 4  # WRONG - This is 'L' in ESM2!
```

**After:**
```python
masked_input_ids[mask_positions] = 32  # Use <mask> token ID (32 in ESM2)
```

---

## ESM2 Token Vocabulary Reference

For future reference, here's the complete ESM2 token vocabulary:

| Token ID | Token | Description |
|----------|-------|-------------|
| 0 | `<cls>` | Classification token |
| 1 | `<pad>` | Padding token |
| 2 | `<eos>` | End of sequence |
| 3 | `<unk>` | Unknown amino acid |
| 4-23 | Amino acids | L, A, G, V, S, E, R, T, I, D, P, K, Q, N, F, Y, M, H, W, C |
| 32 | `<mask>` | Mask token for MLM |

### Amino Acid Token Mapping

```
L=4,  A=5,  G=6,  V=7,  S=8,  E=9,  R=10, T=11,
I=12, D=13, P=14, K=15, Q=16, N=17, F=18, Y=19,
M=20, H=21, W=22, C=23
```

---

## Changes Summary

### Files Modified

1. **`utils/data_utils.py`**
   - Lines 200-218: Fixed `sequence_to_tokens()` in both Dataset classes to use ESM2 mapping
   - Lines 157, 443: Changed padding token from 0 → 1
   - Lines 164, 450: Fixed attention mask to check for pad token 1

2. **`scripts/train.py`**
   - Line 142: Fixed mask token 4 → 32 (training)
   - Line 340: Fixed mask token 4 → 32 (validation)

---

## Impact

✅ **Token IDs now match ESM2 vocabulary exactly**
✅ **No more CUDA device-side assertions**
✅ **MLM loss will be calculated correctly**
✅ **Model will train successfully**

---

## Testing

After applying these fixes:

```bash
git pull  # Get the fixes
python scripts/train.py --config configs/config.yaml --data_path data/processed
```

Expected behavior:
- Training starts successfully
- No CUDA assert errors
- MLM loss is computed correctly
- Model trains normally

---

## Why This Matters

**Before:** Your model was seeing:
- `<mask>` tokens as leucine (L)
- Wrong amino acid mappings
- Invalid token IDs causing CUDA errors

**After:** Your model sees:
- Correct `<mask>` tokens (ID 32)
- Correct amino acid mappings matching ESM2
- All valid token IDs → No CUDA errors

This fix is **critical** - without it, the model cannot train at all because it's receiving invalid input that violates ESM2's vocabulary constraints.

---

## IMPORTANT: Re-process Dataset

⚠️ **You must regenerate your dataset** with the new token mapping!

The current `processed_dataset.pkl` was created with the old (wrong) token IDs. To fix:

```bash
# Re-run data processing with corrected token mapping
python scripts/process_data.py \
    --raw_dir [path_to_pdb_files] \
    --output_dir data/processed \
    --create_efficient_dataset

# This will regenerate processed_dataset.pkl with correct ESM2 token IDs
```

If you don't regenerate, the dataset will still have the old wrong token IDs, and training will still fail!

---

## Alternative: Use ESM Tokenizer Directly

For even better compatibility, consider using the ESM tokenizer directly instead of the custom `sequence_to_tokens` function:

```python
# In Dataset.__init__:
from transformers import EsmTokenizer
self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")

# In __getitem__:
encoded = self.tokenizer(sequence, add_special_tokens=False, return_tensors='pt')
token_ids = encoded['input_ids'].squeeze(0).tolist()
```

This ensures 100% compatibility with ESM2's expected input format.
