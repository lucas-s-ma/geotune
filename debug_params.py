#!/usr/bin/env python
"""
Debug script to check which parameters are trainable in the ESM model with LoRA
"""
import torch
from models.geotune_esm_model import load_esm_with_lora

def analyze_trainable_params():
    """Analyze which parameters are trainable"""
    print("Analyzing trainable parameters in ESM model with LoRA...")
    
    # Load model with LoRA
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
    
    # Count and analyze trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {trainable_params/total_params*100:.4f}%")
    
    print("\nTrainable parameters breakdown:")
    lora_param_count = 0
    lm_head_param_count = 0
    constraint_param_count = 0
    other_param_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            print(f"  {name}: {param_count:,} params")
            
            if 'lora' in name.lower():
                lora_param_count += param_count
            elif 'lm_head' in name.lower():
                lm_head_param_count += param_count
            elif 'constraint' in name.lower():
                constraint_param_count += param_count
            else:
                other_param_count += param_count
    
    print(f"\nBreakdown:")
    print(f"  LoRA parameters: {lora_param_count:,}")
    print(f"  LM head parameters: {lm_head_param_count:,}")
    print(f"  Constraint parameters: {constraint_param_count:,}")
    print(f"  Other parameters: {other_param_count:,}")
    print(f"  Total calculated: {lora_param_count + lm_head_param_count + constraint_param_count + other_param_count:,}")

if __name__ == "__main__":
    analyze_trainable_params()