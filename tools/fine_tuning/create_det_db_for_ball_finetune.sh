#!/bin/bash
# =========================================================================
# Create D-FINE Detection Database for Ball-Finetuning Training Sequences
# =========================================================================
# Runs D-FINE on the 6 SportsMOT volleyball training sequences and
# merges the results into a single det_db JSON for MOTRv2 training.
#
# Usage:
#   bash tools/fine_tuning/create_det_db_for_ball_finetune.sh [SCORE_THRESHOLD]
#
# Default score threshold: 0.3
# =========================================================================

set -e

SCORE_THRESHOLD="${1:-0.3}"

# Paths
DFINE_DIR="/home/es/es_es/es_lekamt00/BeachKI/D-FINE"
MOTRV2_DIR="/home/es/es_es/es_lekamt00/BeachKI/MOTRv2"
MOT_PATH="$MOTRV2_DIR/data/Dataset/mot"
TRAIN_BASE="$MOT_PATH/volleyball/train_with_ball"
OUTPUT_DB="$MOT_PATH/det_db_volleyball_ball_dfine.json"
TEMP_DIR="$MOTRV2_DIR/outputs/temp_det_db_ball"

# D-FINE config
DFINE_CONFIG="$DFINE_DIR/configs/dfine/objects365/dfine_hgnetv2_l_obj365.yml"
DFINE_WEIGHTS="$DFINE_DIR/dfine_l_obj365.pth"
# Class IDs: 0=person, 1=bicycle, 36=sports ball, 156=tennis racket, 240=ball
ALLOWED_CLASSES="0,1,36,156,240"

echo "========================================="
echo "D-FINE Detection DB for Ball-Finetuning"
echo "========================================="
echo "Score threshold: $SCORE_THRESHOLD"
echo "Allowed classes: $ALLOWED_CLASSES"
echo "Output: $OUTPUT_DB"
echo ""

# Check prerequisites
if [ ! -f "$DFINE_WEIGHTS" ]; then
    echo "ERROR: D-FINE weights not found at $DFINE_WEIGHTS"
    exit 1
fi

if [ ! -d "$TRAIN_BASE" ]; then
    echo "ERROR: Training data not found at $TRAIN_BASE"
    exit 1
fi

# Check GPU
python3 -c "import torch; assert torch.cuda.is_available(), 'No GPU!'; print(f'GPU: {torch.cuda.get_device_name(0)}')" || {
    echo "ERROR: No GPU available. Run this on a GPU node."
    exit 1
}

# Create temp directory
mkdir -p "$TEMP_DIR"

# List of sequences
SEQUENCES=(
    "v_1LwtoLPw2TU_c006"
    "v_1LwtoLPw2TU_c012"
    "v_1LwtoLPw2TU_c014"
    "v_1LwtoLPw2TU_c016"
    "v_ApPxnw_Jffg_c001"
    "v_ApPxnw_Jffg_c002"
)

echo "Processing ${#SEQUENCES[@]} sequences..."
echo ""

# Process each sequence
for SEQ in "${SEQUENCES[@]}"; do
    IMG_DIR="$TRAIN_BASE/$SEQ/img1"
    SEQ_NAME="volleyball/train_with_ball/$SEQ"

    # Resolve symlink to actual image path
    REAL_IMG_DIR=$(readlink -f "$IMG_DIR")

    if [ ! -d "$REAL_IMG_DIR" ]; then
        echo "WARNING: Images not found for $SEQ at $REAL_IMG_DIR"
        continue
    fi

    NUM_IMAGES=$(ls "$REAL_IMG_DIR"/*.jpg 2>/dev/null | wc -l)
    echo "========================================="
    echo "Sequence: $SEQ ($NUM_IMAGES images)"
    echo "  Images: $REAL_IMG_DIR"
    echo "  MOTRv2 path: $SEQ_NAME"
    echo "========================================="

    # Run D-FINE inference
    cd "$DFINE_DIR"
    python tools/inference/torch_inf.py \
        -c "$DFINE_CONFIG" \
        -r "$DFINE_WEIGHTS" \
        -i "$REAL_IMG_DIR/*.jpg" \
        -d "cuda:0" \
        --motrv2 \
        --sequence-name "$SEQ_NAME" \
        --allowed-classes "$ALLOWED_CLASSES" \
        --motrv2-score-threshold "$SCORE_THRESHOLD"

    # Move the generated det_db to temp dir with sequence-specific name
    GENERATED_DB="$REAL_IMG_DIR/det_db_motrv2.json"
    if [ -f "$GENERATED_DB" ]; then
        mv "$GENERATED_DB" "$TEMP_DIR/det_db_${SEQ}.json"
        echo "  Saved to: $TEMP_DIR/det_db_${SEQ}.json"
    else
        echo "  WARNING: No det_db generated for $SEQ"
        # Also check in D-FINE directory
        if [ -f "$DFINE_DIR/det_db_motrv2.json" ]; then
            mv "$DFINE_DIR/det_db_motrv2.json" "$TEMP_DIR/det_db_${SEQ}.json"
            echo "  Saved from D-FINE dir to: $TEMP_DIR/det_db_${SEQ}.json"
        fi
    fi

    echo ""
done

# Merge all sequence det_dbs into one
echo "========================================="
echo "Merging detection databases..."
echo "========================================="

cd "$MOTRV2_DIR"
python3 << 'PYEOF'
import json
import os
import sys

temp_dir = os.environ.get('TEMP_DIR', 'outputs/temp_det_db_ball')
output_db = os.environ.get('OUTPUT_DB', 'data/Dataset/mot/det_db_volleyball_ball_dfine.json')

merged = {}
total_frames = 0
total_dets = 0

for fname in sorted(os.listdir(temp_dir)):
    if not fname.endswith('.json'):
        continue

    fpath = os.path.join(temp_dir, fname)
    print(f"Loading {fname}...")

    with open(fpath) as f:
        db = json.load(f)

    seq_frames = len(db)
    seq_dets = sum(len(v) for v in db.values())
    print(f"  {seq_frames} frames, {seq_dets} detections")

    merged.update(db)
    total_frames += seq_frames
    total_dets += seq_dets

# Save merged database
os.makedirs(os.path.dirname(output_db), exist_ok=True)
with open(output_db, 'w') as f:
    json.dump(merged, f)

print(f"\n{'='*50}")
print(f"Merged detection database:")
print(f"  Total frames: {total_frames}")
print(f"  Total detections: {total_dets}")
print(f"  Avg detections/frame: {total_dets/max(1,total_frames):.1f}")
print(f"  File: {output_db}")
print(f"  Size: {os.path.getsize(output_db)/1024:.1f} KB")
print(f"{'='*50}")

# Show sample keys
keys = sorted(merged.keys())[:3]
print(f"\nSample keys:")
for k in keys:
    print(f"  {k}: {len(merged[k])} detections")
PYEOF

echo ""
echo "========================================="
echo "Done!"
echo "========================================="
echo ""
echo "New detection database: $OUTPUT_DB"
echo ""
echo "To use for training, update volleyball_ball_finetune.args:"
echo "  --det_db det_db_volleyball_ball_dfine.json"
echo ""
echo "Cleanup temp files:"
echo "  rm -rf $TEMP_DIR"
echo "========================================="
