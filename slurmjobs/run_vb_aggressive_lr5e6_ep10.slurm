#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --exclude=gpu104
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:0:0
#SBATCH --nodes=1
#SBATCH --output=vb_aggressive_lr5e6_ep10_%j.out
#SBATCH --job-name=motrv2_a5e6e10

module load devel/miniforge/24.11.0-python-3.12
module load devel/cuda/12.4

export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29504

export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

eval "$(conda shell.bash hook)"
conda activate pytorch

echo "========================================="
echo "MOTRv2 Finetuning: Aggressive LR=5e-6 Epochs=10"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

cd /home/es/es_es/es_lekamt00/BeachKI/MOTRv2

echo "Configuration: aggressive, lr=5e-6, epochs=10"
echo "Output dir: outputs/finetune_vb_aggressive_lr5e6_ep10"
echo ""

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

CHECKPOINT="outputs/finetune_vb_aggressive_lr5e6_ep10/checkpoint.pth"
if [ -f "$CHECKPOINT" ]; then
    python tools/fine_tuning/finetune_for_dfine.py $(cat configs/volleyball_finetune_aggressive_lr5e6_ep10.args) --resume "$CHECKPOINT"
else
    python tools/fine_tuning/finetune_for_dfine.py $(cat configs/volleyball_finetune_aggressive_lr5e6_ep10.args)
fi

echo "Training completed at $(date)"
echo "Output: outputs/finetune_vb_aggressive_lr5e6_ep10"
