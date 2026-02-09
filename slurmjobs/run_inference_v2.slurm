#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:0:0
#SBATCH --nodes=1
#SBATCH --output=logs/inference_v2_all_%j.out
#SBATCH --job-name=motrv2_infer_v2

# =========================================================================
# MOTRv2 Inference for all v2 Fine-Tuned Models
# =========================================================================

# Load modules
module load devel/miniforge/24.11.0-python-3.12
module load devel/cuda/12.4

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate pytorch

# Change to MOTRv2 directory
cd /home/es/es_es/es_lekamt00/BeachKI/MOTRv2

# Create logs directory
mkdir -p logs

echo "========================================="
echo "MOTRv2 v2 Inference - All Models"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Test video/images path
TEST_IMAGES="./data/Dataset/mot/volleyball/test/test1/img1"
VIDEO_PATH="volleyball/test/test1"

# Detection database (must match test dataset!)
# det_db_beach_volleyball.json has keys for volleyball/test/test1/
DET_DB="det_db_beach_volleyball.json"

# Inference parameters
SCORE_THRESHOLD=0.3
MISS_TOLERANCE=20
NUM_QUERIES=10

# List of all 9 configurations
CONFIGS=(
    "minimal_lr5e6"
    "minimal_lr1e5"
    "minimal_lr2e5"
    "moderate_lr5e6"
    "moderate_lr1e5"
    "moderate_lr2e5"
    "aggressive_lr5e6"
    "aggressive_lr1e5"
    "aggressive_lr2e5"
)

# Run inference for each config
for config in "${CONFIGS[@]}"; do
    checkpoint="./outputs/finetune_vb_${config}_ep5/checkpoint0004.pth"
    output_dir="./outputs/inference_v2_${config}_ep5"

    echo ""
    echo "========================================="
    echo "Inference: $config"
    echo "========================================="
    echo "Checkpoint: $checkpoint"
    echo "Output: $output_dir"
    echo "Time: $(date)"
    echo ""

    if [ ! -f "$checkpoint" ]; then
        echo "WARNING: Checkpoint not found, skipping: $checkpoint"
        continue
    fi

    mkdir -p "$output_dir"

    python submit_dance.py \
        --meta_arch motr \
        --dataset_file e2e_dance \
        --with_box_refine \
        --query_interaction_layer QIMv2 \
        --num_queries $NUM_QUERIES \
        --sampler_lengths 2 \
        --sample_mode random_interval \
        --sample_interval 30 \
        --use_checkpoint \
        --resume "$checkpoint" \
        --det_db "$DET_DB" \
        --mot_path ./data/Dataset/mot \
        --output_dir "$output_dir" \
        --score_threshold $SCORE_THRESHOLD \
        --miss_tolerance $MISS_TOLERANCE

    echo "Completed: $config"
done

echo ""
echo "========================================="
echo "All Inferences Completed!"
echo "========================================="
echo "End time: $(date)"
echo ""
echo "Results:"
ls -la outputs/inference_v2_*/
