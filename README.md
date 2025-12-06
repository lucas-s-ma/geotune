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
geotune/
├── configs/              # Configuration files
├── data/                 # Raw and processed data
├── data_pipeline/        # Scripts for data processing and validation
├── models/               # Model definitions
├── scripts/              # Training and evaluation scripts
└── utils/                # Utility functions
```

## Data Pipeline

The data processing pipeline consists of the following steps:

1.  **PDB Download**: Raw PDB files are downloaded into the `data/pdb` directory.
2.  **Data Processing**: The `data_pipeline/process_dataset.py` script processes the PDB files into a single `processed_dataset.pkl` file. This file contains the protein sequence, 3D coordinates of backbone atoms (N, CA, C), and other relevant information.
3.  **Structural Token Generation (Optional)**: The `data_pipeline/generate_foldseek_tokens.py` script uses Foldseek to generate 3Di structural tokens for each protein. These tokens represent the local structure of the protein and are saved to `structural_tokens.pkl`.
4.  **GearNet Embedding Generation (Optional)**: The `data_pipeline/generate_gearnet_embeddings.py` script pre-computes GearNet embeddings for each protein. These embeddings provide a structural representation of the protein and are saved in the `data/processed/embeddings` directory.
5.  **Dataset Validation**: The `data_pipeline/validate_dataset.py` script can be used to validate the integrity and correctness of the processed data, structural tokens, and embeddings.

## Models

GeoTune supports fine-tuning of ESM2 models. The core model is `GeoTuneESMModel` in `models/geotune_esm_model.py`, which is a wrapper around a base ESM2 model that includes LoRA for efficient fine-tuning and a language modeling head.

### Loading a Pre-trained Model

You can load a pre-trained GeoTune model with LoRA adapters using the `load_esm_with_lora` function:

```python
from models.geotune_esm_model import load_esm_with_lora

model, tokenizer = load_esm_with_lora(
    model_name="facebook/esm2_t30_150M_UR50D",
    lora_weights_path="path/to/lora_adapters"
)
```

### Saving a Model

The LoRA adapters can be saved during or after training using the `save_lora_adapters` method:

```python
model.save_lora_adapters("path/to/save/lora_adapters")
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- PEFT (for LoRA)
- Biopython (for PDB parsing)
- NumPy, SciPy
- Matplotlib
- Scikit-learn
- Seaborn
- tqdm
- OmegaConf
- wandb

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Download Data
Download PDB files into the `data/pdb` directory. You can use the provided script or download them manually.
```bash
python scripts/download_data.py --output_dir data/pdb
```

### 2. Process Data
Process the raw PDB files into a format suitable for training. This will create a `processed_dataset.pkl` file in the `data/processed` directory.
```bash
python data_pipeline/process_dataset.py --raw_dir data/pdb --output_dir data/processed --create_efficient_dataset
```

### 3. Generate Structural Tokens (Optional)
Generate 3Di structural tokens using Foldseek.
```bash
python data_pipeline/generate_foldseek_tokens.py --pdb_file data/pdb/1a2z.pdb --output_file data/processed/structural_tokens.pkl
```

### 4. Generate GearNet Embeddings (Optional)
Pre-compute GearNet embeddings for the processed dataset.
```bash
python data_pipeline/generate_gearnet_embeddings.py --processed_dataset_path data/processed --output_dir data/processed/embeddings
```

### 5. Validate the Dataset
After processing the data and generating tokens/embeddings, it's recommended to validate the dataset.
```bash
python data_pipeline/validate_dataset.py --data_dir data/processed --embedding_dim 512
```

### 6. Train the Model
Train the GeoTune model using the processed data.

#### Unconstrained Training
```bash
python scripts/train.py --config configs/config.yaml --data_path data/processed
```

#### Constrained Training
```bash
python scripts/train_constrained.py --config configs/config.yaml --data_path data/processed
```

### 7. Evaluate the Model
Evaluate a trained model on a test set.
```bash
python scripts/evaluate.py --model_path outputs/final_model/lora_adapters --data_path data/processed
```