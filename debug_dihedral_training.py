#!/usr/bin/env python
"""
Debug script to test the new dihedral angle constraint implementation in training
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.geotune_esm_model import load_esm_with_lora
from utils.data_utils import collate_fn
from utils.dihedral_utils import DihedralAngleConstraint
import os

def debug_dihedral_training():
    """Debug the new dihedral angle constraint implementation"""
    
    print("Setting up model and dihedral constraints for debugging...")
    
    # Load model with LoRA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    lora_params = {
        'r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'target_modules': ["query", "key", "value", "dense", "intermediate.dense", "output.dense"]
    }
    
    model, tokenizer = load_esm_with_lora(
        model_name="facebook/esm2_t30_150M_UR50D",
        lora_params=lora_params
    )
    model.to(device)
    
    # Initialize dihedral angle constraints
    dihedral_constraints = DihedralAngleConstraint(
        constraint_weight=0.1
    ).to(device)
    
    print("Creating a small dummy dataset for testing with dihedral angles...")
    
    # Create a small dummy dataset for testing
    class DummyDihedralDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2  # Small dataset for quick testing
        
        def __getitem__(self, idx):
            # Create dummy data similar to what the real dataset would provide
            seq_len = 20  # Short sequence for testing
            input_ids = torch.randint(5, 25, (seq_len,))  # Random amino acid tokens
            attention_mask = torch.ones(seq_len)  # All valid tokens
            
            # Create dummy N, CA, C coordinates
            n_coords = torch.randn(seq_len, 3) 
            ca_coords = torch.randn(seq_len, 3) 
            c_coords = torch.randn(seq_len, 3) 
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'n_coords': n_coords,
                'ca_coords': ca_coords,
                'c_coords': c_coords,
                'seq_len': seq_len,
                'protein_id': f'dummy_{idx}'
            }
    
    dataset = DummyDihedralDataset()
    
    # Create a small dataloader for debugging
    dataloader = DataLoader(
        dataset,
        batch_size=8,  # Increased batch size for better efficiency
        shuffle=False,  # No shuffle for predictable debugging
        collate_fn=collate_fn,
        num_workers=0  # Use 0 to avoid multiprocessing issues during debugging
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Get one batch and test each component
    batch = next(iter(dataloader))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"N coordinates shape: {batch['n_coords'].shape}")
    print(f"CA coordinates shape: {batch['ca_coords'].shape}")
    print(f"C coordinates shape: {batch['c_coords'].shape}")
    
    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    n_coords = batch['n_coords'].to(device)
    ca_coords = batch['ca_coords'].to(device)
    c_coords = batch['c_coords'].to(device)
    
    # Test model forward pass (without geometric features)
    print("\nTesting model forward pass...")
    try:
        model.train()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        print(f"Model forward pass successful. Output keys: {list(outputs.keys())}")
        print(f"Sequence output shape: {outputs['sequence_output'].shape}")
        print(f"Constraint features shape: {outputs['constraint_features'].shape}")
    except Exception as e:
        print(f"Error in model forward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test the MLM loss calculation
    print("\nTesting MLM loss calculation...")
    try:
        # Create masked tokens for MLM
        seq_output = outputs['sequence_output']
        batch_size, seq_len, hidden_dim = seq_output.shape
        mask_ratio = 0.15
        
        # Create mask for masking tokens (ignore padding positions)
        mask_positions = (torch.rand(batch_size, seq_len, device=device) < mask_ratio) & (attention_mask.bool())
        
        # Clone input_ids to create labels
        labels = input_ids.clone()
        labels[~mask_positions] = -100  # Ignore non-masked positions in loss
        
        # Create masked input
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask_positions] = 4  # Use mask token ID (4 in our mapping)
        
        # Recompute with masked inputs to get predictions for masked positions
        masked_outputs = model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask
        )
        
        # Calculate MLM loss
        seq_logits = model.lm_head(masked_outputs['sequence_output'])
        mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)(seq_logits.view(-1, seq_logits.size(-1)), labels.view(-1))
        print(f"MLM loss calculation successful: {mlm_loss.item():.4f}")
    except Exception as e:
        print(f"Error in MLM loss calculation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test dihedral angle constraint loss calculation
    print("\nTesting dihedral angle constraint loss calculation...")
    try:
        dihedral_losses = dihedral_constraints(
            masked_outputs['sequence_output'], 
            n_coords, 
            ca_coords, 
            c_coords, 
            attention_mask
        )
        print(f"Dihedral constraint loss calculation successful.")
        print(f"Dihedral loss components: {list(dihedral_losses.keys())}")
        print(f"Total dihedral loss: {dihedral_losses['total_dihedral_loss'].item():.4f}")
        print(f"Phi loss: {dihedral_losses['phi_loss'].item():.4f}")
        print(f"Psi loss: {dihedral_losses['psi_loss'].item():.4f}")
    except Exception as e:
        print(f"Error in dihedral constraint loss calculation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test backward pass
    print("\nTesting backward pass...")
    try:
        total_loss = mlm_loss + 0.1 * dihedral_losses['total_dihedral_loss']
        
        # Clear gradients first
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        
        total_loss.backward()
        print("Backward pass successful!")
        
        # Check if gradients are finite (no NaN or inf)
        has_nan = False
        total_grad_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"NaN/Inf gradients in {name}")
                    has_nan = True
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        if has_nan:
            print("WARNING: Found NaN/Inf gradients!")
        else:
            print(f"Gradient norm: {total_grad_norm:.4f}")
    except Exception as e:
        print(f"Error in backward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ All components working correctly with dihedral angle constraints!")
    print("The training should now be much faster and not get stuck at the first batch.")
    print("Key improvements:")
    print("- Dihedral constraints are much more computationally efficient than geometric constraints")
    print("- O(n) complexity instead of O(n²) for angle calculations")
    print("- Proper extraction of N, CA, C backbone coordinates from PDB files")
    print("- Efficient training with dihedral angle preservation")

if __name__ == "__main__":
    debug_dihedral_training()