# GeoTune New

A clean implementation of constraint learning methodology for fine-tuning baseline Protein Language Models (PLMs) with geometric information as constraints.

## Overview

GeoTune New implements a constraint learning approach that fine-tunes ESM2 models using geometric information from protein structures. The methodology incorporates geometric constraints during training to ensure the model learns meaningful structure-aware representations while maintaining sequence understanding. LoRA is used to make the training process memory efficient and faster.

## Key Features

1. **Constraint Learning**: Explicit geometric constraints during training
2. **LoRA Integration**: Memory efficient fine-tuning with Low-Rank Adaptation
3. **ESM2 Support**: Fine-tuning of ESM2 models of various sizes
4. **Geometric Information**: Uses 3D coordinates and structural features
5. **Modular Design**: Clean architecture for easy experimentation

## Architecture

```
geotune_new/
├── data/                   # Data processing and storage
│   ├── raw/               # Raw protein data
│   ├── processed/         # Processed datasets
│   └── pdb/               # PDB files and structures
├── models/                # Model definitions and LoRA components
│   ├── esm_model.py       # ESM2 model with LoRA integration
│   ├── constraint_model.py # Model with constraint layers
│   └── lora_utils.py      # LoRA utilities
├── utils/                 # Utility functions
│   ├── data_utils.py      # Data loading and preprocessing
│   ├── geom_utils.py      # Geometric constraint utilities
│   └── train_utils.py     # Training utilities
├── configs/               # Configuration files
│   ├── config.yaml        # Main configuration
│   └── models/            # Model-specific configs
├── scripts/               # Training and evaluation scripts
│   ├── train.py           # Main training script
│   ├── download_data.py   # Data download script
│   └── evaluate.py        # Evaluation script
├── notebooks/             # Analysis and exploration notebooks
└── requirements.txt       # Dependencies
```

## Constraint Learning Approach

The constraint learning methodology incorporates geometric information during training in the following ways:

1. **Distance Constraints**: Ensures predicted representations maintain proper inter-residue distances
2. **Angle Constraints**: Maintains geometric relationships between residues
3. **Secondary Structure Constraints**: Aligns predictions with known secondary structure elements
4. **Spatial Neighborhood Constraints**: Preserves local structural environments

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- PEFT (for LoRA)
- Biopython (for PDB parsing)
- NumPy, SciPy

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Download protein structure data:
```bash
python scripts/download_data.py --output_dir data/pdb
```

2. Process data:
```bash
python data_pipeline/process_dataset.py --raw_dir data/pdb --output_dir data/processed
```

3. Train model with geometric constraints:
```bash
python scripts/train.py --config configs/config.yaml
```