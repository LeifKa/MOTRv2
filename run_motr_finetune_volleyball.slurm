#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:0:0
#SBATCH --output=motr_finetune_volleyball_%j.out
#SBATCH --job-name=motr_vb_ft

# Load modules
module load devel/miniforge/24.11.0-python-3.12
module load devel/cuda/12.4

# Activate conda
eval "$(conda shell.bash hook)"
conda activate pytorch

# Print job info
echo "========================================="
echo "MOTRv2 Fine-Tuning (Volleyball)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Training strategy: moderate"
echo "Layers: yolox_embed + track_embed + class_embed"
echo "Dataset: e2e_dance (single-class tracking)"
echo "Start time: $(date)"
echo ""

# Go to MOTRv2 directory
cd /home/es/es_es/es_lekamt00/BeachKI/MOTRv2

# Set PYTHONPATH for imports
export PYTHONPATH=/home/es/es_es/es_lekamt00/BeachKI/MOTRv2:$PYTHONPATH

# Run MOTRv2 fine-tuning with moderate strategy
python tools/fine_tuning/finetune_for_dfine.py \
    --meta_arch motr \
    --dataset_file e2e_dance \
    --epochs 10 \
    --with_box_refine \
    --lr_drop 8 \
    --lr 1e-5 \
    --lr_backbone 1e-6 \
    --pretrained ./weights/motrv2_dancetrack.pth \
    --batch_size 1 \
    --save_period 1 \
    --sample_mode random_interval \
    --sample_interval 10 \
    --sampler_lengths 5 \
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
    --output_dir outputs/finetune_volleyball_full \
    --exp_name volleyball_full_dfine_finetune \
    --num_workers 4 \
    --train_strategy moderate \
    --resume ./weights/motrv2_dancetrack.pth

echo ""
echo "========================================="
echo "Fine-tuning completed!"
echo "End time: $(date)"
echo "========================================="