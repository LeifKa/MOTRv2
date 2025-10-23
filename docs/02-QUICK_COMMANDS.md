# Quick Command Reference: Fine-tuning yolox_embed

## One-Page Cheat Sheet

### Prerequisites
- D-FINE checkpoint: `dfine_l_obj365.pth`
- MOTRv2 checkpoint: `motrv2_checkpoint.pth`
- Training video(s): volleyball footage
- Test video: beach volleyball clip

---

## Step 1: Generate Training Data D-FINE Detections

```bash
cd D-FINE

# Create training config (lower threshold, persons only)
cat > motrv2_training_config.json << 'EOF'
{
  "allowed_classes": [0],
  "score_threshold": 0.3
}
EOF

# Run D-FINE on training video
python tools/inference/torch_inf.py \
    -c configs/dfine/dfine_hgnetv2_l_coco.yml \
    -r dfine_l_obj365.pth \
    --input /path/to/training_video.mp4 \
    --motrv2 \
    --motrv2_config motrv2_training_config.json \
    --sequence_name volleyball/train/game1

# Copy detections to MOTRv2
cp det_db_motrv2.json ../MOTRv2/det_db_volleyball_train.json
```

---

## Step 2: Generate Pseudo Ground Truth (Using YOLOX+MOTRv2)

```bash
cd ../MOTRv2

# Run MOTRv2 with YOLOX (which works well) to get pseudo-labels
python submit_dance.py \
    --resume motrv2_checkpoint.pth \
    --det_db det_db_yolox_train.json \
    --mot_path ./data/Dataset/mot \
    --score_threshold 0.6 \
    --output_dir outputs/pseudo_labels \
    --exp_name train_pseudo_gt

# Convert to ground truth format
python convert_to_gt.py \
    --input outputs/pseudo_labels/game1.txt \
    --output data/Dataset/mot/volleyball/train/game1/gt/gt.txt
```

---

## Step 3: Extract Training Frames

```bash
# Create directory structure
mkdir -p data/Dataset/mot/volleyball/train/game1/img1

# Extract frames
ffmpeg -i /path/to/training_video.mp4 \
    -qscale:v 2 \
    data/Dataset/mot/volleyball/train/game1/img1/%06d.jpg
```

---

## Step 4: Create Training List

```bash
# List sequences for training
mkdir -p datasets/data_path

echo "volleyball/train/game1" > datasets/data_path/volleyball_train.txt
# Add more sequences if you have them:
# echo "volleyball/train/game2" >> datasets/data_path/volleyball_train.txt
```

---

## Step 5: Fine-tune yolox_embed

```bash
# Run fine-tuning (1-2 hours)
python finetune_for_dfine.py \
    --meta_arch motr \
    --dataset_file e2e_dance \
    --epochs 2 \
    --with_box_refine \
    --lr 1e-5 \
    --lr_backbone 0 \
    --resume motrv2_checkpoint.pth \
    --batch_size 1 \
    --sample_mode random_interval \
    --sample_interval 10 \
    --sampler_lengths 5 \
    --query_interaction_layer QIMv2 \
    --num_queries 10 \
    --det_db det_db_volleyball_train.json \
    --mot_path ./data/Dataset/mot \
    --data_txt_path_train ./datasets/data_path/volleyball_train.txt \
    --output_dir outputs/finetune_dfine_embed \
    --exp_name dfine_volleyball \
    --embed_only True \
    --num_workers 2

# Output: outputs/finetune_dfine_embed/checkpoint0001.pth
```

---

## Step 6: Generate Test D-FINE Detections

```bash
cd ../D-FINE

# Create test config (higher threshold for inference)
cat > motrv2_test_config.json << 'EOF'
{
  "allowed_classes": [0],
  "score_threshold": 0.7
}
EOF

# Run D-FINE on test video
python tools/inference/torch_inf.py \
    -c configs/dfine/dfine_hgnetv2_l_coco.yml \
    -r dfine_l_obj365.pth \
    --input /path/to/beach_volleyball_test.mp4 \
    --motrv2 \
    --motrv2_config motrv2_test_config.json \
    --sequence_name volleyball/test/test1

# Copy to MOTRv2
cp det_db_motrv2.json ../MOTRv2/det_db_beach_test.json
```

---

## Step 7: Extract Test Frames

```bash
cd ../MOTRv2

mkdir -p data/Dataset/mot/volleyball/test/test1/img1

ffmpeg -i /path/to/beach_volleyball_test.mp4 \
    -qscale:v 2 \
    data/Dataset/mot/volleyball/test/test1/img1/%06d.jpg
```

---

## Step 8: Test Fine-tuned Model

```bash
# Test with FINE-TUNED model
python submit_dance.py \
    --resume outputs/finetune_dfine_embed/checkpoint0001.pth \
    --det_db det_db_beach_test.json \
    --mot_path ./data/Dataset/mot \
    --score_threshold 0.7 \
    --output_dir outputs/test_finetuned \
    --exp_name beach_finetuned

# Test with ORIGINAL model (for comparison)
python submit_dance.py \
    --resume motrv2_checkpoint.pth \
    --det_db det_db_beach_test.json \
    --mot_path ./data/Dataset/mot \
    --score_threshold 0.7 \
    --output_dir outputs/test_original \
    --exp_name beach_original
```

---

## Step 9: Visualize Results

```bash
# Visualize fine-tuned results
python visualize_tracking.py \
    --images data/Dataset/mot/volleyball/test/test1/img1 \
    --tracking outputs/test_finetuned/test1.txt \
    --output beach_finetuned_tracking.mp4

# Visualize original results
python visualize_tracking.py \
    --images data/Dataset/mot/volleyball/test/test1/img1 \
    --tracking outputs/test_original/test1.txt \
    --output beach_original_tracking.mp4

# Or create side-by-side comparison
python visualize_tracking.py \
    --images data/Dataset/mot/volleyball/test/test1/img1 \
    --tracking outputs/test_finetuned/test1.txt \
    --tracking2 outputs/test_original/test1.txt \
    --output beach_comparison.mp4 \
    --labels "Fine-tuned" "Original"
```

---

## Troubleshooting

### "No such file or directory"
```bash
# Check your data structure
tree data/Dataset/mot/volleyball/train/game1/
# Should show: img1/ and gt/ directories
```

### "Loss not decreasing"
```bash
# Try lower learning rate
python finetune_for_dfine.py ... --lr 5e-6
```

### "CUDA out of memory"
```bash
# Reduce batch size or sequence length
python finetune_for_dfine.py ... --batch_size 1 --sampler_lengths 3
```

### "FileNotFoundError: det_db_..."
```bash
# Verify detection file exists
ls -lh det_db_*.json

# Verify it's valid JSON
python merge_detections.py --verify det_db_volleyball_train.json
```

---

## Helper Scripts

### Merge multiple detection files
```bash
python merge_detections.py \
    --input D-FINE/det_db_*.json \
    --output MOTRv2/det_db_volleyball_train.json
```

### Batch convert tracking to ground truth
```bash
python convert_to_gt.py \
    --input outputs/pseudo_labels/ \
    --output data/Dataset/mot/volleyball/train/ \
    --batch \
    --min_track_length 10
```

### Verify detection database
```bash
python merge_detections.py --verify det_db_volleyball_train.json
```

---

## Expected Files After Each Step

**After Step 1:**
- ✅ `D-FINE/det_db_motrv2.json`
- ✅ `MOTRv2/det_db_volleyball_train.json`

**After Step 2:**
- ✅ `outputs/pseudo_labels/game1.txt`
- ✅ `data/Dataset/mot/volleyball/train/game1/gt/gt.txt`

**After Step 3:**
- ✅ `data/Dataset/mot/volleyball/train/game1/img1/000001.jpg` (and more)

**After Step 4:**
- ✅ `datasets/data_path/volleyball_train.txt`

**After Step 5:**
- ✅ `outputs/finetune_dfine_embed/checkpoint0001.pth`

**After Step 6:**
- ✅ `MOTRv2/det_db_beach_test.json`

**After Step 7:**
- ✅ `data/Dataset/mot/volleyball/test/test1/img1/000001.jpg` (and more)

**After Step 8:**
- ✅ `outputs/test_finetuned/test1.txt`
- ✅ `outputs/test_original/test1.txt`

**After Step 9:**
- ✅ `beach_finetuned_tracking.mp4`
- ✅ `beach_original_tracking.mp4` (or comparison video)

---

## Key Points to Remember

1. **Training detections** use lower threshold (0.3) and less filtering
2. **Test detections** use higher threshold (0.7) for cleaner inference
3. **Pseudo-labels** come from YOLOX+MOTRv2 (which works better currently)
4. **Only yolox_embed trains** - everything else is frozen (256 parameters)
5. **Training takes 1-2 hours** for 2 epochs on GPU
6. **Comparison** is key - always test both fine-tuned and original

---

## Quick Sanity Checks

```bash
# 1. Check training data structure
ls data/Dataset/mot/volleyball/train/game1/img1/ | wc -l
ls data/Dataset/mot/volleyball/train/game1/gt/

# 2. Check detection database
python -c "import json; print(len(json.load(open('det_db_volleyball_train.json'))))"

# 3. Check ground truth
head data/Dataset/mot/volleyball/train/game1/gt/gt.txt

# 4. Monitor training
tail -f outputs/finetune_dfine_embed/log.txt  # If logging enabled

# 5. Quick visualization test
python visualize_tracking.py \
    --images data/Dataset/mot/volleyball/test/test1/img1 \
    --tracking outputs/test_finetuned/test1.txt \
    --output test_quick.mp4
```

---

For detailed explanations, see: **FINE_TUNING_STEP_BY_STEP.md**
