#!/usr/bin/env python
"""
Test script to verify ESM2 model loading and LoRA integration
"""
import torch
from models.esm_model import load_esm_with_lora, ESMWithConstraints, create_lora_config

def test_base_model_loading():
    """Test if we can load the base ESM2 model correctly"""
    print("Testing base ESM2 model loading...")
    
    try:
        # Test loading the base ESM model directly
        from transformers import EsmModel, EsmTokenizer, EsmConfig
        
        model_name = "facebook/esm2_t30_150M_UR50D"
        print(f"Loading base ESM model: {model_name}")
        
        # Load config first
        config = EsmConfig.from_pretrained(model_name)
        print(f"Config loaded. Hidden size: {config.hidden_size}, vocab size: {config.vocab_size}")
        
        # Load tokenizer
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded successfully")
        
        # Load model
        base_model = EsmModel.from_pretrained(model_name)
        print(f"Base model loaded. Model type: {type(base_model)}")
        print(f"Base model parameters: {base_model.num_parameters():,}")
        
        # Test a simple forward pass
        sample_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        inputs = tokenizer(sample_sequence, return_tensors="pt", add_special_tokens=True)
        
        with torch.no_grad():
            outputs = base_model(**inputs)
            print(f"Forward pass successful. Last hidden state shape: {outputs.last_hidden_state.shape}")
        
        return True, base_model, tokenizer
        
    except Exception as e:
        print(f"Error loading base model: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_lora_integration():
    """Test if LoRA is properly integrated"""
    print("\nTesting LoRA integration...")
    
    try:
        # Create LoRA config
        lora_params = {
            'r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'target_modules': ["query", "key", "value", "dense", "intermediate.dense", "output.dense"]
        }
        
        lora_config = create_lora_config(**lora_params)
        print(f"LoRA config created: {lora_config}")
        
        # Load model with LoRA
        model, tokenizer = load_esm_with_lora(
            model_name="facebook/esm2_t30_150M_UR50D",
            lora_params=lora_params
        )
        
        print(f"Model with LoRA loaded. Model type: {type(model)}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {trainable_params/total_params*100:.4f}%")
        
        # Check if LoRA layers are actually trainable
        lora_param_count = 0
        base_param_count = 0
        
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                lora_param_count += param.numel()
                if param.requires_grad:
                    print(f"  LoRA parameter {name} is trainable: {param.requires_grad}")
            elif param.requires_grad and 'lora' not in name.lower():
                base_param_count += param.numel()
        
        print(f"LoRA parameters count: {lora_param_count:,}")
        print(f"Base model parameters that are still trainable: {base_param_count:,}")
        
        if base_param_count == 0:
            print("✓ Only LoRA parameters are trainable (as expected)")
        else:
            print(f"! Warning: {base_param_count:,} base model parameters are still trainable")
        
        # Test forward pass
        sample_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        inputs = tokenizer(sample_sequence, return_tensors="pt", add_special_tokens=True)
        
        # Move to device if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            print(f"Forward pass with LoRA successful. Output keys: {list(outputs.keys())}")
            if 'sequence_output' in outputs:
                print(f"Sequence output shape: {outputs['sequence_output'].shape}")
        
        return True, model, tokenizer
        
    except Exception as e:
        print(f"Error testing LoRA integration: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def main():
    print("Testing ESM2 model loading and LoRA integration...\n")
    
    # Test 1: Base model loading
    success1, base_model, base_tokenizer = test_base_model_loading()
    
    if success1:
        print("\n✓ Base model loading test PASSED")
    else:
        print("\n✗ Base model loading test FAILED")
        return
    
    # Test 2: LoRA integration
    success2, lora_model, lora_tokenizer = test_lora_integration()
    
    if success2:
        print("\n✓ LoRA integration test PASSED")
    else:
        print("\n✗ LoRA integration test FAILED")
        return
    
    print("\n🎉 All tests passed! ESM2 model and LoRA integration are working correctly.")

if __name__ == "__main__":
    main()