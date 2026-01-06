#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:0:0
#SBATCH --nodes=1
#SBATCH --output=volleyball_combined_finetune_%j.out
#SBATCH --job-name=motrv2_vbc_ft

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
echo "MOTRv2 Finetuning on Volleyball Combined Dataset"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Change to MOTRv2 directory
cd /home/es/es_es/es_lekamt00/BeachKI/MOTRv2

echo "========================================="
echo "Dataset Information"
echo "========================================="
echo "Detection DB: det_db_volleyball_combined.json"
echo "Data path: datasets/data_path/volleyball_combined_train.txt"
echo "Output dir: outputs/finetune_volleyball_combined"
echo ""

# Check if checkpoint exists for resume
CHECKPOINT="outputs/finetune_volleyball_combined/checkpoint.pth"
if [ -f "$CHECKPOINT" ]; then
    echo "Found existing checkpoint, resuming training..."
    python tools/fine_tuning/finetune_for_dfine.py @configs/volleyball_combined_finetune.args --resume "$CHECKPOINT"
else
    echo "Starting fresh training..."
    python tools/fine_tuning/finetune_for_dfine.py @configs/volleyball_combined_finetune.args
fi

echo ""
echo "========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "Output directory: outputs/finetune_volleyball_combined"
echo "========================================="
