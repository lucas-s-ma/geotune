"""
Data pipeline package for protein structure processing

This package contains scripts for:
- Processing PDB files into training-ready datasets
- Generating foldseek structural tokens
- Generating GearNet structural embeddings
- Validating processed datasets
"""

from .process_dataset import create_efficient_dataset
from .generate_foldseek_tokens import generate_foldseek_tokens, check_foldseek_installation, convert_3di_to_ints
from .generate_gearnet_embeddings import generate_gearnet_embeddings_for_dataset, generate_gearnet_embeddings_for_protein
from .validate_dataset import validate_processed_dataset, validate_structural_tokens, validate_gearnet_embeddings

__all__ = [
    'create_efficient_dataset',
    'generate_foldseek_tokens', 
    'check_foldseek_installation',
    'convert_3di_to_ints',
    'generate_gearnet_embeddings_for_dataset',
    'generate_gearnet_embeddings_for_protein',
    'validate_processed_dataset',
    'validate_structural_tokens', 
    'validate_gearnet_embeddings'
]