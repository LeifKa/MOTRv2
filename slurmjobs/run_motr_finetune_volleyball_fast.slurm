#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=4:0:0
#SBATCH --output=motr_finetune_volleyball_fast_%j.out
#SBATCH --job-name=motr_vb_fast

# Load modules
module load devel/miniforge/24.11.0-python-3.12
module load devel/cuda/12.4

# Activate conda
eval "$(conda shell.bash hook)"
conda activate pytorch

# Print job info
echo "========================================="
echo "MOTRv2 Fine-Tuning (Volleyball - OPTIMIZED)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: H100"
echo "Training strategy: moderate"
echo "Layers: yolox_embed + track_embed + class_embed"
echo "Dataset: e2e_dance (single-class tracking)"
echo ""
echo "Optimizations:"
echo "  - Batch size: 1 (MOTRv2 doesn't support >1)"
echo "  - Sample interval: 30 (was 10) -> 3x speedup"
echo "  - Sampler lengths: 2 (was 5) -> 2.5x speedup"
echo "  - Epochs: 3 (was 10)"
echo "  - Workers: 8 (was 4)"
echo "  => Expected: ~4-5h total (was 30h)"
echo ""
echo "Start time: $(date)"
echo ""

# Go to MOTRv2 directory
cd /home/es/es_es/es_lekamt00/BeachKI/MOTRv2

# Set PYTHONPATH for imports
export PYTHONPATH=/home/es/es_es/es_lekamt00/BeachKI/MOTRv2:$PYTHONPATH

# Run MOTRv2 fine-tuning with optimized settings
python tools/fine_tuning/finetune_for_dfine.py \
    --meta_arch motr \
    --dataset_file e2e_dance \
    --epochs 3 \
    --with_box_refine \
    --lr_drop 2 \
    --lr 1e-5 \
    --lr_backbone 1e-6 \
    --pretrained ./weights/motrv2_dancetrack.pth \
    --batch_size 1 \
    --save_period 1 \
    --sample_mode random_interval \
    --sample_interval 30 \
    --sampler_lengths 2 \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.4 \
    --fp_ratio 0.3 \
    --query_interaction_layer QIMv2 \
    --query_denoise 0.05 \
    --num_queries 10 \
    --det_db det_db_for_finetune_dfine.json \
    --use_checkpoint \
    --mot_path ./data/Dataset/mot \
    --data_txt_path_train ./datasets/data_path/volleyball_full_train.txt \
    --output_dir outputs/finetune_volleyball_full_fast \
    --exp_name volleyball_full_dfine_finetune_fast \
    --num_workers 8 \
    --train_strategy moderate \
    --resume ./weights/motrv2_dancetrack.pth

echo ""
echo "========================================="
echo "Fine-tuning completed!"
echo "End time: $(date)"
echo "========================================="