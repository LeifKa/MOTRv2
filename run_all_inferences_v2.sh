#!/bin/bash
# =========================================================================
# MOTRv2 Inference & Visualization for all v2 Fine-Tuned Models
# =========================================================================
# This script runs inference and visualization for all 9 fine-tuned models
# from the v2 training (correct GT + D-FINE Detection DB)
# =========================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Test video/images path
TEST_IMAGES="./data/Dataset/mot/volleyball/test/test1/img1"
VIDEO_PATH="volleyball/test/test1"

# Detection database for inference (must match test dataset path!)
# det_db_volleyball_finetune_dfine.json = training data (volleyball/finetune/gt/)
# det_db_beach_volleyball.json = test data (volleyball/test/test1/)
DET_DB="det_db_beach_volleyball.json"

# Inference parameters (can be adjusted)
SCORE_THRESHOLD=0.3
MISS_TOLERANCE=20
NUM_QUERIES=10
SAMPLER_LENGTHS=2
SAMPLE_INTERVAL=30

# Visualization parameters
FPS=30

echo "========================================="
echo "MOTRv2 v2 Inference & Visualization"
echo "========================================="
echo "Test images: $TEST_IMAGES"
echo "Detection DB: $DET_DB"
echo "Score threshold: $SCORE_THRESHOLD"
echo "Miss tolerance: $MISS_TOLERANCE"
echo ""

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

# Function to run inference for a single model
run_inference() {
    local config=$1
    local checkpoint="./outputs/finetune_vb_${config}_ep5/checkpoint0004.pth"
    local output_dir="./outputs/inference_v2_${config}_ep5"

    echo ""
    echo "========================================="
    echo "Running inference: $config"
    echo "========================================="
    echo "Checkpoint: $checkpoint"
    echo "Output: $output_dir"
    echo ""

    if [ ! -f "$checkpoint" ]; then
        echo "ERROR: Checkpoint not found: $checkpoint"
        return 1
    fi

    # Create output directory
    mkdir -p "$output_dir"

    # Run inference
    python submit_dance.py \
        --meta_arch motr \
        --dataset_file e2e_dance \
        --with_box_refine \
        --query_interaction_layer QIMv2 \
        --num_queries $NUM_QUERIES \
        --sampler_lengths $SAMPLER_LENGTHS \
        --sample_mode random_interval \
        --sample_interval $SAMPLE_INTERVAL \
        --use_checkpoint \
        --resume "$checkpoint" \
        --det_db "$DET_DB" \
        --mot_path ./data/Dataset/mot \
        --output_dir "$output_dir" \
        --score_threshold $SCORE_THRESHOLD \
        --miss_tolerance $MISS_TOLERANCE

    echo "Inference completed: $output_dir"
}

# Function to run visualization for a single model
run_visualization() {
    local config=$1
    local tracking_file="./outputs/inference_v2_${config}_ep5/tracking_inference.txt"
    local output_video="./outputs/inference_v2_${config}_ep5/tracking_visualization.mp4"

    echo ""
    echo "========================================="
    echo "Running visualization: $config"
    echo "========================================="
    echo "Tracking file: $tracking_file"
    echo "Output video: $output_video"
    echo ""

    if [ ! -f "$tracking_file" ]; then
        echo "ERROR: Tracking file not found: $tracking_file"
        return 1
    fi

    # Run visualization
    python tools/visualization/visualize_tracking.py \
        --images "$TEST_IMAGES" \
        --tracking "$tracking_file" \
        --output "$output_video" \
        --fps $FPS

    echo "Visualization completed: $output_video"
}

# Parse command line arguments
MODE="${1:-all}"  # all, inference, visualization, or specific config name

case "$MODE" in
    "all")
        echo "Running inference AND visualization for all 9 configs..."
        for config in "${CONFIGS[@]}"; do
            run_inference "$config"
            run_visualization "$config"
        done
        ;;
    "inference")
        echo "Running ONLY inference for all 9 configs..."
        for config in "${CONFIGS[@]}"; do
            run_inference "$config"
        done
        ;;
    "visualization")
        echo "Running ONLY visualization for all 9 configs..."
        for config in "${CONFIGS[@]}"; do
            run_visualization "$config"
        done
        ;;
    *)
        # Specific config
        if [[ " ${CONFIGS[*]} " =~ " ${MODE} " ]]; then
            echo "Running inference and visualization for: $MODE"
            run_inference "$MODE"
            run_visualization "$MODE"
        else
            echo "Usage: $0 [all|inference|visualization|<config_name>]"
            echo ""
            echo "Available configs:"
            for config in "${CONFIGS[@]}"; do
                echo "  - $config"
            done
            exit 1
        fi
        ;;
esac

echo ""
echo "========================================="
echo "All tasks completed!"
echo "========================================="
echo ""
echo "Results are in:"
for config in "${CONFIGS[@]}"; do
    echo "  - outputs/inference_v2_${config}_ep5/"
done
echo ""
echo "Videos:"
for config in "${CONFIGS[@]}"; do
    echo "  - outputs/inference_v2_${config}_ep5/tracking_visualization.mp4"
done
