"""
Evaluation script for trained GeoTune models
"""
import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse

from models.esm_model import load_esm_with_lora
from utils.data_utils import ProteinStructureDataset, collate_fn
from utils.geom_utils import GeometricConstraints


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained GeoTune model")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to trained model (LoRA adapters)")
    parser.add_argument("--data_path", type=str, required=True, 
                       help="Path to evaluation data")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t30_150M_UR50D",
                       help="Base model name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--constraint_weight", type=float, default=0.1, 
                       help="Weight for constraint loss")
    parser.add_argument("--dist_threshold", type=float, default=15.0,
                       help="Distance threshold for constraints")
    
    return parser.parse_args()


def evaluate_model(model, dataloader, geometric_constraints, device):
    """Evaluate the model on test data"""
    model.eval()
    all_mlm_losses = []
    all_constraint_losses = []
    all_combined_losses = []
    
    distance_losses = []
    angle_losses = []
    neighborhood_losses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            coordinates = batch['coordinates'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                geometric_features=coordinates
            )
            
            # Create random masks for MLM task
            seq_output = outputs['sequence_output']
            batch_size, seq_len, hidden_dim = seq_output.shape
            mask_ratio = 0.15
            
            mask_positions = (torch.rand(batch_size, seq_len, device=device) < mask_ratio) & (attention_mask.bool())
            labels = input_ids.clone()
            labels[~mask_positions] = -100
            
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_positions] = 4  # Use mask token ID
            
            masked_outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                geometric_features=coordinates
            )
            
            # Calculate MLM loss
            seq_logits = model.lm_head(masked_outputs['sequence_output'])
            mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)(seq_logits.view(-1, seq_logits.size(-1)), labels.view(-1))
            
            # Calculate constraint loss
            constraint_losses = geometric_constraints(
                masked_outputs['sequence_output'], 
                coordinates, 
                attention_mask
            )
            
            # Combine losses
            combined_loss = mlm_loss + constraint_losses['total_constraint_loss']
            
            # Store losses
            all_mlm_losses.append(mlm_loss.item())
            all_constraint_losses.append(constraint_losses['total_constraint_loss'].item())
            all_combined_losses.append(combined_loss.item())
            
            # Store individual constraint losses
            distance_losses.append(constraint_losses['distance_loss'].item())
            angle_losses.append(constraint_losses['angle_loss'].item())
            neighborhood_losses.append(constraint_losses['neighborhood_loss'].item())
    
    # Calculate averages
    avg_mlm_loss = np.mean(all_mlm_losses)
    avg_constraint_loss = np.mean(all_constraint_losses)
    avg_combined_loss = np.mean(all_combined_losses)
    
    avg_distance_loss = np.mean(distance_losses)
    avg_angle_loss = np.mean(angle_losses)
    avg_neighborhood_loss = np.mean(neighborhood_losses)
    
    return {
        'mlm_loss': avg_mlm_loss,
        'constraint_loss': avg_constraint_loss,
        'combined_loss': avg_combined_loss,
        'distance_loss': avg_distance_loss,
        'angle_loss': avg_angle_loss,
        'neighborhood_loss': avg_neighborhood_loss
    }


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    lora_params = {
        'r': 8,  # Default LoRA params, adjust as needed
        'lora_alpha': 16,
        'target_modules': ["query", "key", "value", "dense", "intermediate.dense", "output.dense"]
    }
    model, tokenizer = load_esm_with_lora(args.model_name, lora_params)
    
    # Load LoRA adapters
    # Note: You'd need to adapt this part based on how you saved the model
    # This is a simplified version - actual loading would depend on your saving format
    print(f"Loading LoRA adapters from {args.model_path}")
    
    # For now, we'll just move the model to device
    # In a real scenario, you'd load the specific adapter weights
    model.to(device)
    
    # Initialize geometric constraints
    geometric_constraints = GeometricConstraints(
        constraint_weight=args.constraint_weight,
        dist_threshold=args.dist_threshold
    ).to(device)
    
    # Load test dataset
    print("Loading evaluation dataset...")
    eval_dataset = ProteinStructureDataset(args.data_path)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Evaluate
    print("Starting evaluation...")
    results = evaluate_model(model, eval_loader, geometric_constraints, device)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"MLM Loss: {results['mlm_loss']:.4f}")
    print(f"Constraint Loss: {results['constraint_loss']:.4f}")
    print(f"Combined Loss: {results['combined_loss']:.4f}")
    print(f"Distance Loss: {results['distance_loss']:.4f}")
    print(f"Angle Loss: {results['angle_loss']:.4f}")
    print(f"Neighborhood Loss: {results['neighborhood_loss']:.4f}")
    
    # Save results
    results_path = os.path.join(args.model_path, "evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("=" * 20 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value:.4f}\n")
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()