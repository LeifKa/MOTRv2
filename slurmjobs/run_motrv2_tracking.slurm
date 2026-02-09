#!/bin/bash
#SBATCH --partition=gpu8
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --time=2:0:0
#SBATCH --nodes=1
#SBATCH --output=motrv2_tracking_%j.out
#SBATCH --job-name=motrv2_track

# Load modules
module load devel/miniforge/24.11.0-py3.12
module load devel/cuda/12.4

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate pytorch

# Navigate to MOTRv2 directory
cd /home/es/es_es/es_lekamt00/BeachKI/MOTRv2

# Run MOTRv2 tracking with YOLOX detections to generate pseudo ground truth
echo "Starting MOTRv2 tracking with YOLOX detections..."
echo "Output directory: outputs/pseudo_labels"

python submit_dance.py \
    --meta_arch motr \
    --dataset_file e2e_dance \
    --with_box_refine \
    --query_interaction_layer QIMv2 \
    --num_queries 10 \
    --sample_mode random_interval \
    --sample_interval 10 \
    --sampler_lengths 5 \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_denoise 0.05 \
    --append_crowd \
    --resume weights/motrv2_dancetrack.pth \
    --det_db det_db_yolox_train.json \
    --mot_path ./data/Dataset/mot \
    --output_dir outputs/pseudo_labels \
    --exp_name train_pseudo_gt \
    --score_threshold 0.6 \
    --use_checkpoint

echo "MOTRv2 tracking complete!"
echo "Results saved to: outputs/pseudo_labels/"
