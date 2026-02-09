#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2:0:0
#SBATCH --nodes=1
#SBATCH --output=logs/volleyball_ball_inference_%j.out
#SBATCH --job-name=motrv2_ball_inf

# =========================================================================
# MOTRv2 Ball-Tracking Inference
# =========================================================================
# Runs inference with ball-tracking fine-tuned model
# Uses lower score_threshold and area_threshold for ball detection
# =========================================================================

# Load modules
module load devel/miniforge/24.11.0-python-3.12
module load devel/cuda/12.4

# Set environment variables
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export OMP_NUM_THREADS=8

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate pytorch

# Change to MOTRv2 directory
cd /home/es/es_es/es_lekamt00/BeachKI/MOTRv2

mkdir -p logs

# Configuration
CHECKPOINT="outputs/volleyball_ball_finetune/checkpoint.pth"
DET_DB="det_db_beach_volleyball.json"
OUTPUT_DIR="outputs/volleyball_ball_inference"
SCORE_THRESHOLD=0.2
AREA_THRESHOLD=10
MISS_TOLERANCE=20

echo "========================================="
echo "MOTRv2 Ball-Tracking Inference"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Checkpoint: $CHECKPOINT"
echo "Score threshold: $SCORE_THRESHOLD"
echo "Area threshold: $AREA_THRESHOLD"
echo "Start time: $(date)"
echo "========================================="

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Run fine-tuning first: sbatch slurmjobs/run_volleyball_ball_finetune.slurm"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python submit_dance.py \
    --meta_arch motr \
    --dataset_file e2e_volleyball_ball \
    --with_box_refine \
    --query_interaction_layer QIMv2 \
    --num_queries 10 \
    --sampler_lengths 2 \
    --sample_mode random_interval \
    --sample_interval 2 \
    --use_checkpoint \
    --resume "$CHECKPOINT" \
    --det_db "$DET_DB" \
    --mot_path ./data/Dataset/mot \
    --output_dir "$OUTPUT_DIR" \
    --score_threshold $SCORE_THRESHOLD \
    --area_threshold $AREA_THRESHOLD \
    --miss_tolerance $MISS_TOLERANCE

EXIT_CODE=$?

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Inference Completed!"
    echo "Results: $OUTPUT_DIR/tracking_inference.txt"
    echo ""
    echo "Output format: frame,id,x,y,w,h,score,label,-1,-1"
    echo "  label=0: player"
    echo "  label=1: ball"
else
    echo "Inference Failed with exit code: $EXIT_CODE"
fi
echo "========================================="
echo "End time: $(date)"

exit $EXIT_CODE
