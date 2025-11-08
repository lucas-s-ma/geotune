# Constrained Learning Implementation in GeoTune

## Overview

The constrained learning process in GeoTune implements a multi-objective optimization approach that combines sequence modeling with structural constraints. This approach ensures that the learned protein representations respect both sequence patterns and structural properties, leading to more physically realistic protein embeddings.

## Architecture Components

### 1. Multi-Component Loss Function

The total loss function combines three key components:

#### A. Masked Language Model (MLM) Loss
- **Purpose**: Standard sequence reconstruction task to learn evolutionary patterns
- **Implementation**: Uses ESM2 or similar protein language model to predict masked amino acids
- **Role**: Maintains sequence-level learning while respecting structural constraints

#### B. Dihedral Constraint Loss
- **Purpose**: Enforces backbone dihedral angle constraints based on PDB reference structures
- **Implementation**: Compares predicted phi/psi angles to reference angles from PDB files
- **Method**: Calculates dihedral angles from backbone coordinates (N, CA, C atoms)

#### C. Structural Alignment Loss
- **Purpose**: Aligns protein language model embeddings with structural embeddings
- **Implementation**: Dual-level approach:
  - **Latent-level**: Contrastive learning between language model and GNN embeddings
  - **Physical-level**: Direct prediction of foldseek 3Di structural tokens

### 2. Lagrangian Optimization Framework

The core constrained learning mechanism uses Lagrangian optimization to balance primary learning objectives with structural constraints:

```
L_total = L_MLM + λ * L_constraint + α * L_structural_alignment
```

Where:
- `L_MLM`: Primary sequence learning loss
- `L_constraint`: Dihedral angle constraint violations
- `L_structural_alignment`: Structural token prediction loss
- `λ`: Adaptive Lagrange multiplier for constraint enforcement
- `α`: Weight for structural alignment (default 0.1)

#### Dynamic Lambda Updates
- Lambda (λ) is updated using dual ascent during training
- When constraint violations exceed thresholds, λ increases to enforce stricter adherence
- This adaptive mechanism balances sequence learning with structural constraints

### 3. Dihedral Angle Constraint Implementation

#### Data Requirements
- N, CA, C backbone coordinates from PDB structures
- Reference dihedral angles computed from native structures
- Proper coordinate parsing and validation

#### Constraint Calculation Process
1. **Coordinate Processing**: Extract N, CA, C coordinates for each residue
2. **Angle Calculation**: Compute phi (φ) and psi (ψ) angles using standard formulas:
   - φ: angle between (i-1)C, iN, iCA, iC atoms
   - ψ: angle between iN, iCA, iC, (i+1)N atoms
3. **Reference Comparison**: Compare predicted angles to PDB reference values
4. **Violation Scoring**: Calculate deviations exceeding acceptable thresholds
5. **Loss Aggregation**: Sum violations across all constrained angles

### 4. Structural Alignment Loss Details

#### Latent-Level Component
- **Purpose**: Align language model embeddings with structural embeddings (GearNet)
- **Method**: Contrastive learning in shared projection space
- **Implementation**:
  - Project both pLM and pGNN embeddings to shared space using linear layers
  - Compute similarity matrix with learnable temperature scaling
  - Apply cross-entropy loss for positive pair identification

#### Physical-Level Component
- **Purpose**: Direct structural token prediction (foldseek 3Di alphabet)
- **Method**: Cross-entropy loss for predicting 20-class structural tokens
- **Implementation**:
  - MLP head predicts structural token logits from pLM embeddings
  - 3Di alphabet: 20 structural states (A-Y), excluding B, J, O, U, X
  - Proper token mapping to [0, 19] range for CrossEntropyLoss

### 5. Training Workflow

#### Forward Pass Sequence
1. **Input Processing**: Process amino acid sequence and backbone coordinates
2. **Embedding Generation**: 
   - Generate language model embeddings from sequence
   - Generate GNN embeddings from coordinates (frozen during training)
3. **Dihedral Calculation**: Compute phi/psi angles from predicted coordinates
4. **Constraint Evaluation**: Calculate dihedral constraint violations
5. **Structural Alignment**: Compute structural alignment if tokens available
6. **Lagrangian Formation**: Combine losses using adaptive lambda
7. **Backward Pass**: Update parameters with combined loss

#### Optimization Strategy
- **Mixed Precision Training**: Uses autocast for memory efficiency
- **Gradient Accumulation**: Handles large sequences by accumulating gradients
- **Dynamic Lambda Updates**: Adjusts constraint emphasis based on violation levels
- **Validation Loop**: Separate validation phase with same multi-objective structure

### 6. Implementation Details

#### Data Pipeline Integration
- **Protein Structure Dataset**: Efficient loading of sequences and coordinates
- **Structural Token Processing**: Foldseek-based 3Di token generation
- **Coordinate Validation**: Ensures proper N, CA, C coordinate availability
- **Batch Processing**: Maintains attention masks for variable-length sequences

#### Loss Function Components
```python
# Pseudocode structure
def compute_total_loss(model_outputs, reference_data):
    mlm_loss = compute_mlm_loss(model_outputs, sequence_labels)
    constraint_loss = compute_dihedral_constraints(model_outputs, reference_coords)
    structural_loss = compute_structural_alignment(model_outputs, foldseek_tokens)
    
    lagrangian = mlm_loss + lambda_weight * constraint_loss
    total_loss = lagrangian + structural_weight * structural_loss
    
    return total_loss, mlm_loss, constraint_loss, structural_loss
```

#### Constraint Enforcement Mechanism
- **Threshold-based Violations**: Dihedral angles exceeding standard deviations trigger penalties
- **Adaptive Lambda**: Dual ascent updates based on running averages of constraint violations
- **Gradual Training**: Constraints may be phased in during early training epochs

### 7. Technical Considerations

#### Memory Management
- Mixed precision reduces memory usage while maintaining numerical stability
- Gradient accumulation allows larger effective batch sizes
- Efficient coordinate processing minimizes overhead

#### Convergence Challenges
- Multi-objective optimization can create conflicting gradients
- Adaptive lambda helps balance competing objectives
- Proper initialization and learning rate scheduling are critical

#### Validation Metrics
- Separate tracking of each loss component
- Dihedral angle accuracy metrics
- Structural alignment quality measures
- Sequence reconstruction quality

## Training Commands

### Basic Training with Constraints
```bash
python scripts/train_constrained.py --config config/training_config.yaml
```

### Key Configuration Parameters
- `model.constraint_weight`: Weight for dihedral constraint loss
- `training.gradient_accumulation_steps`: For handling large sequences
- `training.mixed_precision`: Enable mixed precision training
- `model.lambda_init`: Initial value for Lagrange multiplier

## Troubleshooting Common Issues

### Constraint Violation Spikes
- **Cause**: Lambda becoming too large
- **Solution**: Reduce initial lambda value or adjust update frequency

### Structural Alignment Loss Stagnation
- **Cause**: Improper token mapping or missing structural tokens
- **Solution**: Verify foldseek token generation and token range validation

### Memory Issues with Large Sequences
- **Cause**: Coordinate tensors for long proteins
- **Solution**: Reduce batch size or use gradient accumulation

## Performance Monitoring

### Metrics to Track
- Individual loss component values over time
- Lambda value evolution
- Constraint violation frequency
- Validation accuracy for each objective

### Expected Behavior
- Constraint violations should decrease over time
- Structural alignment loss should improve with sufficient training
- MLM performance should remain stable while satisfying constraints
- Lambda should stabilize at optimal values for balance

## Dependencies and Requirements

### External Tools
- **Foldseek**: For structural token generation
- **Biopython**: For PDB parsing and coordinate extraction
- **TorchDrug**: For GearNet implementation (optional stub available)

### Software Requirements
- PyTorch with CUDA support
- Mixed precision training capabilities
- Sufficient GPU memory for coordinate tensors