#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:0:0
#SBATCH --nodes=1
#SBATCH --output=logs/finetune_vb_aggressive_lr1e5_ep5_%j.out
#SBATCH --job-name=motrv2_vb_ft_lr1e5_ep5

# =========================================================================
# MOTRv2 Fine-Tuning on Volleyball Dataset with D-FINE Detection Database
# =========================================================================
# This script trains MOTRv2 with:
# - Correct Ground Truth: volleyball/finetune/gt/
# - D-FINE Detection DB: det_db_volleyball_finetune_dfine.json
# - Config: volleyball_finetune_aggressive_lr1e5_ep5.args
#   * Learning rate: 1e-5
#   * Backbone LR: 1e-6
#   * Epochs: 5
#   * Training strategy: aggressive
# =========================================================================

# Load modules
module load devel/miniforge/24.11.0-python-3.12
module load devel/cuda/12.4

# Set environment variables for single-GPU training
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# CUDA optimization flags for H100
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate pytorch

# Print job information
echo "========================================="
echo "MOTRv2 Fine-Tuning: Volleyball Dataset"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Change to MOTRv2 directory
cd /home/es/es_es/es_lekamt00/BeachKI/MOTRv2

# Create logs directory if it doesn't exist
mkdir -p logs

# Configuration
CONFIG="configs/volleyball_finetune_aggressive_lr1e5_ep5.args"
OUTPUT_DIR="outputs/finetune_vb_aggressive_lr1e5_ep5"
PRETRAINED_WEIGHTS="./weights/motrv2_dancetrack.pth"

echo "========================================="
echo "Configuration"
echo "========================================="
echo "Config file: $CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Pretrained weights: $PRETRAINED_WEIGHTS"
echo ""
echo "Dataset:"
echo "  - Ground Truth: volleyball/finetune/gt/"
echo "  - Detection DB: det_db_volleyball_finetune_dfine.json"
echo "  - Data path: datasets/data_path/volleyball_finetune_train.txt"
echo ""
echo "Training parameters:"
echo "  - Learning rate: 1e-5"
echo "  - Backbone LR: 1e-6"
echo "  - Epochs: 5"
echo "  - Strategy: aggressive"
echo "  - Batch size: 1"
echo "  - Num queries: 10"
echo ""

# Check if Detection DB exists
DET_DB="data/Dataset/mot/det_db_volleyball_finetune_dfine.json"
if [ ! -f "$DET_DB" ]; then
    echo "ERROR: Detection database not found at $DET_DB"
    echo "Please run: cd D-FINE/tools/inference && python create_detection_db_for_finetune.py"
    exit 1
fi

# Check if pretrained weights exist
if [ ! -f "$PRETRAINED_WEIGHTS" ]; then
    echo "ERROR: Pretrained weights not found at $PRETRAINED_WEIGHTS"
    echo "Please download motrv2_dancetrack.pth to weights/"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Add current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check for existing checkpoint to resume
CHECKPOINT="$OUTPUT_DIR/checkpoint.pth"

echo "========================================="
echo "Starting Training"
echo "========================================="

if [ -f "$CHECKPOINT" ]; then
    echo "Found existing checkpoint at $CHECKPOINT"
    echo "Resuming training from checkpoint..."
    echo ""
    python main.py @"$CONFIG"
else
    echo "No checkpoint found, starting fresh training..."
    echo ""
    python main.py @"$CONFIG"
fi

EXIT_CODE=$?

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training Completed Successfully!"
else
    echo "Training Failed with exit code: $EXIT_CODE"
fi
echo "========================================="
echo "End time: $(date)"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Model checkpoints:"
ls -lh "$OUTPUT_DIR"/*.pth 2>/dev/null || echo "  No checkpoint files found"
echo ""
echo "To evaluate the model, use:"
echo "  python main.py --eval --resume $OUTPUT_DIR/checkpoint.pth ..."
echo "========================================="

exit $EXIT_CODE
