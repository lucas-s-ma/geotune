#!/bin/bash
# Debug script to run on SLURM cluster to diagnose CUDA assert error
# Usage: bash cluster_debug.sh

echo "=========================================="
echo "GEOTUNE CLUSTER DEBUG SCRIPT"
echo "=========================================="
echo ""

# Step 1: Check if code changes are synced
echo "Step 1: Checking code sync status..."
echo "------------------------------------------"
cd /work/sm996/co-amp

# Check if utils/data_utils.py has the correct token mapping
echo "Checking utils/data_utils.py for ESM2 token mapping..."
if grep -q "'L': 4, 'A': 5, 'G': 6" utils/data_utils.py; then
    echo "✅ data_utils.py has CORRECT ESM2 token mapping (L=4, A=5, ...)"
else
    echo "❌ data_utils.py has WRONG token mapping!"
    echo "   Expected: 'L': 4, 'A': 5, 'G': 6, ..."
    echo "   Need to git pull or sync files from local"
    exit 1
fi

# Check if train.py uses mask token 32
echo "Checking scripts/train.py for mask token..."
if grep -q "masked_input_ids\[mask_positions\] = 32" scripts/train.py; then
    echo "✅ train.py uses CORRECT mask token (32)"
else
    echo "❌ train.py uses WRONG mask token!"
    echo "   Expected: masked_input_ids[mask_positions] = 32"
    exit 1
fi

echo ""

# Step 2: Clear Python cache
echo "Step 2: Clearing Python cache..."
echo "------------------------------------------"
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "✅ Python cache cleared"
echo ""

# Step 3: Check dataset token IDs
echo "Step 3: Checking dataset token IDs..."
echo "------------------------------------------"
python debug_dataset_tokens.py data/processed/processed_dataset.pkl
echo ""

# Step 4: Check git status
echo "Step 4: Checking git status..."
echo "------------------------------------------"
git status --short
echo ""
git log --oneline -5
echo ""

# Step 5: Show Python environment
echo "Step 5: Python environment info..."
echo "------------------------------------------"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"
echo ""

echo "=========================================="
echo "DEBUG COMPLETE"
echo "=========================================="
echo ""
echo "If all checks passed, try running training with:"
echo "  CUDA_LAUNCH_BLOCKING=1 python scripts/train.py --config configs/config.yaml --data_path data/processed"
echo ""
echo "This will show the EXACT line where CUDA error occurs."
