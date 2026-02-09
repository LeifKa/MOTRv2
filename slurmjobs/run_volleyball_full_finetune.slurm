#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:0:0
#SBATCH --nodes=1
#SBATCH --output=volleyball_full_finetune_%j.out
#SBATCH --job-name=motrv2_vb_ft

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
echo "MOTRv2 Full Finetuning on Volleyball Dataset"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Step 1: Create Detection Database
echo "========================================="
echo "Step 1: Creating Detection Database"
echo "========================================="
cd /home/es/es_es/es_lekamt00/BeachKI

DET_DB="MOTRv2/data/Dataset/mot/det_db_volleyball_full_dfine.json"
if [ -f "$DET_DB" ]; then
    echo "Detection database already exists: $DET_DB"
else
    echo "Creating detection database..."
    python create_det_db_volleyball_full.py

    if [ $? -eq 0 ]; then
        echo "✓ Detection database created successfully!"
    else
        echo "✗ Failed to create detection database"
        exit 1
    fi
fi
echo ""

# Step 2: Run MOTRv2 Training
echo "========================================="
echo "Step 2: Starting MOTRv2 Training"
echo "========================================="
cd /home/es/es_es/es_lekamt00/BeachKI/MOTRv2

# Check if checkpoint exists for resume
CHECKPOINT="outputs/finetune_volleyball_full/checkpoint.pth"
if [ -f "$CHECKPOINT" ]; then
    echo "Found existing checkpoint, resuming training..."
    python main.py $(cat configs/volleyball_full_finetune.args) --resume "$CHECKPOINT"
else
    echo "Starting fresh training..."
    python main.py $(cat configs/volleyball_full_finetune.args)
fi

echo ""
echo "========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "Output directory: outputs/finetune_volleyball_full"
echo "========================================="
