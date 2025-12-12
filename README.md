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

## Switching Between ESM2 Models

### Important: Embedding Dimension Compatibility

**When switching between different ESM2 models, you MUST be aware of dimension mismatches with pre-computed GearNet embeddings.**

Different ESM2 models have different hidden dimensions:
- **ESM2-8M** (`facebook/esm2_t6_8M_UR50D`): `hidden_size = 320`
- **ESM2-35M** (`facebook/esm2_t12_35M_UR50D`): `hidden_size = 480`
- **ESM2-150M** (`facebook/esm2_t30_150M_UR50D`): `hidden_size = 640`
- **ESM2-650M** (`facebook/esm2_t33_650M_UR50D`): `hidden_size = 1280`

The GearNet embeddings are initialized to match the ESM2 model's `hidden_size`. This means:

1. **Pre-computed embeddings are dimension-specific**: Embeddings computed for ESM2-150M (dim=640) cannot be used with ESM2-8M (dim=320)
2. **The code detects mismatches automatically**: The training script (train.py:485-516) checks embedding dimensions and falls back to on-the-fly generation if there's a mismatch
3. **You have two options when switching models**:
   - **Option A (Recommended)**: Regenerate GearNet embeddings with the new dimension
   - **Option B**: Let the code generate embeddings on-the-fly (slower but works)

### How to Switch to a Different ESM2 Model

#### Step 1: Update the config file

Edit `configs/config.yaml`:

```yaml
model:
  model_name: "facebook/esm2_t6_8M_UR50D"  # Change to desired model

data_pipeline:
  embedding_dim: 320  # Update to match: 8M=320, 35M=480, 150M=640, 650M=1280
```

#### Step 2: Regenerate GearNet embeddings (Recommended)

If you have pre-computed embeddings from a different ESM2 model, regenerate them:

```bash
python data_pipeline/generate_gearnet_embeddings.py \
    --processed_dataset_path data/processed \
    --output_dir data/processed/embeddings_esm2_8m \
    --hidden_dim 320
```

Then update your config to point to the new embeddings directory:

```yaml
data:
  data_path: "data/processed_esm2_8m"
```

#### Step 3: Train with the new model

```bash
python scripts/train.py --config configs/config.yaml --data_path data/processed
```

The training script will automatically:
- Detect if pre-computed embeddings match the current ESM2 model's dimension
- Use pre-computed embeddings if dimensions match
- Generate embeddings on-the-fly if dimensions don't match

## Primal and Dual Learning Rates

GeoTune supports separate learning rates for the **primal task** (ESM model + MLM) and **dual task** (structure alignment):

```yaml
training:
  primal_lr: 1e-3  # Learning rate for ESM model and MLM head
  dual_lr: 5e-4    # Learning rate for structure alignment loss module
```

If `primal_lr` and `dual_lr` are not specified, the single `learning_rate` parameter will be used for all parameters.

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