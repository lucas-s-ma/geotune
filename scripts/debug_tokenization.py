"""
Debug script to verify tokenization and position mapping for DMS evaluation.
"""
import torch
from transformers import EsmTokenizer

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

# Test sequence
test_seq = "ACDEFGHIKLMNPQRSTVWY"
print(f"Test sequence: {test_seq}")
print(f"Sequence length: {len(test_seq)}")

# Test tokenization
tokens = tokenizer(test_seq, return_tensors="pt")
input_ids = tokens['input_ids'][0]
attention_mask = tokens['attention_mask'][0]

print(f"\nToken IDs: {input_ids.tolist()}")
print(f"Num tokens: {len(input_ids)}")

# Decode each token
print("\nToken breakdown:")
for i, token_id in enumerate(input_ids):
    decoded = tokenizer.decode([token_id])
    print(f"  Position {i}: token_id={token_id.item()}, decoded='{decoded}'")

# Test masking at different positions
print("\n\n=== Testing masked positions ===")
for mask_pos in range(len(test_seq)):
    masked_seq = test_seq[:mask_pos] + '<mask>' + test_seq[mask_pos+1:]
    masked_tokens = tokenizer(masked_seq, return_tensors="pt")
    masked_ids = masked_tokens['input_ids'][0]
    
    # Find mask token position
    mask_token_id = tokenizer.mask_token_id
    mask_positions = (masked_ids == mask_token_id).nonzero(as_tuple=True)[0]
    
    print(f"Mask at seq pos {mask_pos}: token positions = {mask_positions.tolist()}")
    print(f"  Tokens: {[tokenizer.decode([tid]) for tid in masked_ids]}")

# Test amino acid encoding
print("\n\n=== Amino acid to token ID mapping ===")
aa_list = 'ACDEFGHIKLMNPQRSTVWY'
for aa in aa_list:
    token_id = tokenizer.encode(aa, add_special_tokens=False)[0]
    print(f"  {aa}: {token_id}")
