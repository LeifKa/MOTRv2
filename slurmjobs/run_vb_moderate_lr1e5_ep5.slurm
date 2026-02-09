#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:0:0
#SBATCH --nodes=1
#SBATCH --output=vb_moderate_lr1e5_ep5_%j.out
#SBATCH --job-name=motrv2_m1e5e5

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

# Print job information
echo "========================================="
echo "MOTRv2 Finetuning: Moderate LR=1e-5 Epochs=5"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Change to MOTRv2 directory
cd /home/es/es_es/es_lekamt00/BeachKI/MOTRv2

echo "========================================="
echo "Configuration"
echo "========================================="
echo "Strategy: moderate"
echo "Learning Rate: 1e-5"
echo "Epochs: 5"
echo "Output dir: outputs/finetune_vb_moderate_lr1e5_ep5"
echo ""

# Add current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if checkpoint exists for resume
CHECKPOINT="outputs/finetune_vb_moderate_lr1e5_ep5/checkpoint.pth"
if [ -f "$CHECKPOINT" ]; then
    echo "Found existing checkpoint, resuming training..."
    python tools/fine_tuning/finetune_for_dfine.py $(cat configs/volleyball_finetune_moderate_lr1e5_ep5.args) --resume "$CHECKPOINT"
else
    echo "Starting fresh training..."
    python tools/fine_tuning/finetune_for_dfine.py $(cat configs/volleyball_finetune_moderate_lr1e5_ep5.args)
fi

echo ""
echo "========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "Output directory: outputs/finetune_vb_moderate_lr1e5_ep5"
echo "========================================="
