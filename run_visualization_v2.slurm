#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=4:0:0
#SBATCH --nodes=1
#SBATCH --output=logs/visualization_v2_all_%j.out
#SBATCH --job-name=motrv2_vis_v2

# =========================================================================
# MOTRv2 Visualization for all v2 Inference Results
# =========================================================================

# Load modules
module load devel/miniforge/24.11.0-python-3.12
module load devel/cuda/12.4

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate pytorch

# Change to MOTRv2 directory
cd /home/es/es_es/es_lekamt00/BeachKI/MOTRv2

echo "========================================="
echo "MOTRv2 v2 Visualization - All Models"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Test video/images path
TEST_IMAGES="./data/Dataset/mot/volleyball/test/test1/img1"
VIDEO_PATH="volleyball/test/test1"
FPS=30

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

# Run visualization for each config
for config in "${CONFIGS[@]}"; do
    tracking_file="./outputs/inference_v2_${config}_ep5/tracking_inference.txt"
    output_video="./outputs/inference_v2_${config}_ep5/tracking_visualization.mp4"

    echo ""
    echo "========================================="
    echo "Visualization: $config"
    echo "========================================="
    echo "Tracking file: $tracking_file"
    echo "Output video: $output_video"
    echo ""

    if [ ! -f "$tracking_file" ]; then
        echo "WARNING: Tracking file not found, skipping: $tracking_file"
        continue
    fi

    python tools/visualization/visualize_tracking.py \
        --images "$TEST_IMAGES" \
        --tracking "$tracking_file" \
        --output "$output_video" \
        --fps $FPS

    echo "Completed: $output_video"
done

echo ""
echo "========================================="
echo "All Visualizations Completed!"
echo "========================================="
echo "End time: $(date)"
echo ""
echo "Videos created:"
for config in "${CONFIGS[@]}"; do
    video="./outputs/inference_v2_${config}_ep5/tracking_visualization.mp4"
    if [ -f "$video" ]; then
        echo "  ✓ $video ($(du -h "$video" | cut -f1))"
    else
        echo "  ✗ $video (missing)"
    fi
done
