#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:0:0
#SBATCH --nodes=1
#SBATCH --output=logs/finetune_vb_minimal_lr1e5_ep5_%j.out
#SBATCH --job-name=motrv2_minimal_lr1e5_ep5

# =========================================================================
# MOTRv2 Fine-Tuning: MINIMAL Strategy, LR=1-5, 5 Epochs
# =========================================================================
# Training Strategy: minimal
#   - Trains ONLY: yolox_embed (~1-2% of parameters)
# Learning Rate: 1-5
# Dataset: volleyball/finetune/gt/ with D-FINE Detection DB
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

# CUDA optimization flags
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate pytorch

# Job information
echo "========================================="
echo "MOTRv2 Fine-Tuning: MINIMAL Strategy"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Change to MOTRv2 directory
cd /home/es/es_es/es_lekamt00/BeachKI/MOTRv2

# Create logs directory
mkdir -p logs

# Configuration
CONFIG="configs/volleyball_finetune_minimal_lr1e5_ep5.args"
OUTPUT_DIR="outputs/finetune_vb_minimal_lr1e5_ep5"
PRETRAINED_WEIGHTS="./weights/motrv2_dancetrack.pth"
DET_DB="data/Dataset/mot/det_db_volleyball_finetune_dfine.json"

echo "========================================="
echo "Configuration"
echo "========================================="
echo "Config file: $CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Strategy: minimal"
echo "Learning rate: 1-5"
echo "Epochs: 5"
echo ""

# Validation checks
if [ ! -f "$DET_DB" ]; then
    echo "ERROR: Detection database not found at $DET_DB"
    echo "Please run: cd D-FINE/tools/inference && python create_detection_db_for_finetune.py"
    exit 1
fi

if [ ! -f "$PRETRAINED_WEIGHTS" ]; then
    echo "ERROR: Pretrained weights not found at $PRETRAINED_WEIGHTS"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check for checkpoint
CHECKPOINT="$OUTPUT_DIR/checkpoint.pth"

echo "========================================="
echo "Starting Training"
echo "========================================="

if [ -f "$CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $CHECKPOINT"
    echo ""
fi

python main.py @"$CONFIG"

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
echo "Results: $OUTPUT_DIR"
echo ""
echo "Checkpoints:"
ls -lh "$OUTPUT_DIR"/*.pth 2>/dev/null || echo "  No checkpoints found"
echo "========================================="

exit $EXIT_CODE
