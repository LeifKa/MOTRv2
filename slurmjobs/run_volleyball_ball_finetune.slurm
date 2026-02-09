#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:0:0
#SBATCH --nodes=1
#SBATCH --output=logs/volleyball_ball_finetune_%j.out
#SBATCH --job-name=motrv2_ball_ft

# =========================================================================
# MOTRv2 Ball-Tracking Fine-Tuning
# =========================================================================
# Multi-class training: player (class 0) + ball (class 1)
# Uses LabelStudio-annotated SportsMOT data with ball bounding boxes
# num_classes=2, sample_interval=2, moderate training strategy
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
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate pytorch

# Change to MOTRv2 directory
cd /home/es/es_es/es_lekamt00/BeachKI/MOTRv2

# Create logs directory
mkdir -p logs

# Configuration
CONFIG="configs/volleyball_ball_finetune.args"
OUTPUT_DIR="outputs/volleyball_ball_finetune"
PRETRAINED_WEIGHTS="./weights/motrv2_dancetrack.pth"
DET_DB="data/Dataset/mot/det_db_volleyball_ball.json"
TRAIN_STRATEGY="moderate"

echo "========================================="
echo "MOTRv2 Ball-Tracking Fine-Tuning"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Config file: $CONFIG"
echo "  Output directory: $OUTPUT_DIR"
echo "  Strategy: $TRAIN_STRATEGY"
echo "  num_classes: 2 (player + ball)"
echo "  sample_interval: 2"
echo "  Detection DB: $DET_DB"
echo "========================================="
echo ""

# Validation checks
if [ ! -f "$DET_DB" ]; then
    echo "ERROR: Detection database not found at $DET_DB"
    echo "Run: python tools/fine_tuning/build_ball_det_db.py --help"
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

echo "========================================="
echo "GPU Diagnostics"
echo "========================================="
nvidia-smi || echo "WARNING: nvidia-smi failed!"
echo ""
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('NO CUDA!')" || echo "WARNING: CUDA check failed!"
echo "========================================="
echo ""

echo "========================================="
echo "Starting Fine-Tuning"
echo "========================================="

python tools/fine_tuning/finetune_for_dfine.py \
    @"$CONFIG" \
    --train_strategy "$TRAIN_STRATEGY"

EXIT_CODE=$?

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Fine-Tuning Completed Successfully!"
else
    echo "Fine-Tuning Failed with exit code: $EXIT_CODE"
fi
echo "========================================="
echo "End time: $(date)"
echo ""
echo "Results: $OUTPUT_DIR"
echo ""
echo "Checkpoints:"
ls -lh "$OUTPUT_DIR"/*.pth 2>/dev/null || echo "  No checkpoints found"
echo ""
echo "Next: Run inference with"
echo "  sbatch slurmjobs/run_volleyball_ball_inference.slurm"
echo "========================================="

exit $EXIT_CODE
