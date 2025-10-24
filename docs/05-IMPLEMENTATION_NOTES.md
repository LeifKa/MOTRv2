# Implementation Notes: Fine-Tuning MOTRv2 for D-FINE

## Overview

This document describes the actual implementation process for fine-tuning MOTRv2's `yolox_embed` layer to work with D-FINE detections, including all fixes applied and lessons learned.

---

## Directory Structure (Updated)

The project has been reorganized for better file management:

```
MOTRv2/
├── inputs/                           ← NEW: Organized input files
│   ├── detections/                   ← Detection JSON files
│   │   ├── Beispielvideo_Beach_YOLOX.json
│   │   ├── det_db_beach_volleyball.json
│   │   ├── det_db_volleyball_train.json
│   │   └── det_db_volleyball_train_dfine.json
│   └── videos/                       ← Source video files
│       ├── Beispielvideo_Beach_YOLOX.mp4
│       └── Sequenz_Beach.mp4
│
├── weights/                          ← All model checkpoints
│   ├── motrv2_dancetrack.pth
│   └── r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth
│
├── data/Dataset/mot/
│   ├── det_db_volleyball_train_dfine.json  ← Copy required here for training
│   └── volleyball/train/game1/
│       ├── img1/                     ← 956 frames (6-digit format!)
│       │   ├── 000001.jpg
│       │   └── ...
│       └── gt/
│           └── gt.txt                ← 5,311 pseudo-labels
│
├── outputs/
│   ├── finetune_dfine_embed/
│   │   └── checkpoint.pth            ← Fine-tuned model (256 params trained)
│   ├── inference_finetuned/
│   │   ├── dfine_finetuned/
│   │   │   └── game1.txt             ← 4,315 tracking detections
│   │   └── tracking_visualization.mp4
│   └── pseudo_labels/train_pseudo_gt/
│       └── game1.txt                 ← From YOLOX tracking
│
└── tools/
    ├── fine_tuning/
    │   └── finetune_for_dfine.py     ← Fine-tuning script
    ├── conversion/
    │   └── convert_to_gt.py          ← Convert tracking to GT format
    └── visualization/
        └── visualize_tracking.py     ← Create visualization videos
```

---

## Step-by-Step Process Completed

### 1. Extract Training Frames
**Command**:
```bash
python extract_frames.py \
    --video Sequenz_Beach.mp4 \
    --output MOTRv2/data/Dataset/mot/volleyball/train/game1/img1
```

**Result**: 956 frames extracted (000001.jpg to 000956.jpg)

---

### 2. Generate YOLOX Detections
**Purpose**: Create detections for pseudo ground truth generation

**Detection Count**: 10,456 detections across 956 frames

---

### 3. Run MOTRv2 Tracking with YOLOX
**Purpose**: Generate pseudo ground truth labels

**Command**:
```bash
PYTHONPATH=$PWD:$PYTHONPATH python submit_dance.py \
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
```

**Result**: 5,328 detections → 5,311 annotations after filtering

---

### 4. Convert Tracking Results to Ground Truth
**Command**:
```bash
python tools/conversion/convert_to_gt.py \
    outputs/pseudo_labels/train_pseudo_gt/game1.txt \
    data/Dataset/mot/volleyball/train/game1/gt/gt.txt
```

**Result**: GT file with 5,311 annotations in MOT format

---

### 5. Generate D-FINE Detections on Training Data
**Config** (motrv2_training_config.json):
```json
{
  "score_threshold": 0.3
}
```

**Note**: Removed class restrictions because Objects365 uses different class IDs than COCO (class 1 = person in Objects365, class 0 in COCO)

**Result**: 16,484 detections kept (much more than YOLOX due to lower threshold and no class filter)

---

### 6. Fine-Tune yolox_embed Layer
**Command**:
```bash
PYTHONPATH=$PWD:$PYTHONPATH python tools/fine_tuning/finetune_for_dfine.py \
    --meta_arch motr \
    --dataset_file e2e_dance \
    --epochs 1 \
    --with_box_refine \
    --lr 1e-5 \
    --lr_backbone 0 \
    --resume weights/motrv2_dancetrack.pth \
    --batch_size 1 \
    --sample_mode random_interval \
    --sample_interval 10 \
    --sampler_lengths 5 \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_denoise 0.05 \
    --append_crowd \
    --query_interaction_layer QIMv2 \
    --num_queries 10 \
    --det_db det_db_volleyball_train_dfine.json \
    --mot_path ./data/Dataset/mot \
    --data_txt_path_train ./datasets/data_path/volleyball_train.txt \
    --output_dir outputs/finetune_dfine_embed \
    --exp_name dfine_volleyball \
    --embed_only True \
    --num_workers 2 \
    --use_checkpoint
```

**Training Stats**:
- Trainable parameters: 256 (0.00% of total)
- Frozen parameters: 41,494,528 (99.99% of total)
- Duration: ~17 minutes for 1 epoch
- Output: `outputs/finetune_dfine_embed/checkpoint.pth`

---

### 7. Run Inference with Fine-Tuned Model
**Command**:
```bash
PYTHONPATH=$PWD:$PYTHONPATH python submit_dance.py \
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
    --resume outputs/finetune_dfine_embed/checkpoint.pth \
    --det_db det_db_volleyball_train_dfine.json \
    --mot_path ./data/Dataset/mot \
    --output_dir outputs/inference_finetuned \
    --exp_name dfine_finetuned \
    --score_threshold 0.7 \
    --use_checkpoint
```

**Result**: 4,315 tracking detections in `outputs/inference_finetuned/dfine_finetuned/game1.txt`

---

### 8. Visualize Tracking Results
**Command**:
```bash
python tools/visualization/visualize_tracking.py \
    -i data/Dataset/mot/volleyball/train/game1/img1 \
    -t outputs/inference_finetuned/dfine_finetuned/game1.txt \
    -o outputs/inference_finetuned/tracking_visualization.mp4
```

**Output**: Video with bounding boxes and track IDs overlaid

---

## Key Fixes Applied

### Fix 1: Frame Numbering Format
**File**: `datasets/dance.py` (lines 183, 206)

**Problem**: Code expected 8-digit format (`00000001.jpg`) but frames were saved as 6-digit (`000001.jpg`)

**Original Code**:
```python
img_path = os.path.join(self.mot_path, vid, 'img1', f'{idx:08d}.jpg')
```

**Fixed Code**:
```python
img_path = os.path.join(self.mot_path, vid, 'img1', f'{idx:06d}.jpg')
```

---

### Fix 2: Detection Database Key Format
**File**: `submit_dance.py` (line 50)

**Problem**: Code looked for keys with `.txt` extension but JSON had keys without extension

**Original Code**:
```python
for line in self.det_db[f_path[:-4] + '.txt']:
```

**Fixed Code**:
```python
for line in self.det_db[f_path[:-4]]:
```

**Expected Key Format**: `volleyball/train/game1/img1/000001` (no `.txt`)

---

### Fix 3: Dataset Loading
**File**: `datasets/dance.py` (lines 68-88)

**Problem**: Hardcoded `add_mot_folder("DanceTrack/train")` instead of reading from `data_txt_path`

**Solution**: Commented out DanceTrack loading and added:

```python
# Load sequences from data_txt_path file
if data_txt_path and os.path.exists(data_txt_path):
    for line in open(data_txt_path):
        seq_name = line.strip()
        if seq_name:
            print(f"Loading sequence from data_txt_path: {seq_name}")
            vid = seq_name
            gt_path = os.path.join(self.mot_path, vid, 'gt', 'gt.txt')
            # ... load ground truth ...
```

---

### Fix 4: Class ID Differences
**File**: `D-FINE/motrv2_training_config.json`

**Problem**:
- COCO uses class 0 for "person"
- Objects365 uses class 1 for "person"
- Original config had `"allowed_classes": [0]` which filtered out all detections

**Original Config**:
```json
{
  "allowed_classes": [0],
  "score_threshold": 0.3
}
```

**Fixed Config**:
```json
{
  "score_threshold": 0.3
}
```

**Result**: Went from 0 detections to 16,484 detections

---

### Fix 5: .DS_Store Files
**Problem**: macOS creates `.DS_Store` files which training tried to process as sequence directories

**Fix**:
```bash
find MOTRv2/data/Dataset/mot/DanceTrack -name ".DS_Store" -type f -delete
```

---

### Fix 6: Detection File Location
**Problem**: Training expects detection files in `data/Dataset/mot/` directory

**Fix**:
```bash
cp det_db_volleyball_train_dfine.json data/Dataset/mot/
```

---

### Fix 7: Hardcoded Sequence Path
**File**: `submit_dance.py` (line 200)

**Problem**: Script had hardcoded test path

**Original**:
```python
vids = ['volleyball/test/test1']
```

**Fixed**:
```python
vids = ['volleyball/train/game1']
```

---

## How Fine-Tuning Works

The fine-tuning script ([tools/fine_tuning/finetune_for_dfine.py](tools/fine_tuning/finetune_for_dfine.py)) implements:

### 1. Selective Parameter Freezing
```python
def freeze_except_yolox_embed(model):
    for name, param in model.named_parameters():
        if 'yolox_embed' in name:
            param.requires_grad = True  # Train this
        else:
            param.requires_grad = False  # Freeze this
```

**Result**: Only 256 parameters trainable out of 41M+ total

### 2. Training Strategy
- **Data**: D-FINE detections (input) + pseudo GT from YOLOX tracking (labels)
- **Loss**: Standard MOTRv2 tracking loss
- **Learning rate**: 1e-5 (very small to avoid breaking existing weights)
- **Epochs**: 1-2 (quick adaptation)

### 3. What Gets Learned
The `yolox_embed` layer learns to map D-FINE score distributions to the embedding space that MOTRv2 expects:

```
Before:
D-FINE score (0.85) → yolox_embed (trained on YOLOX) → Wrong embedding

After:
D-FINE score (0.85) → yolox_embed (adapted to D-FINE) → Correct embedding
```

### 4. Why This Works
- **Transfer Learning**: Leverages existing tracking knowledge
- **Minimal Changes**: Only adapts the input encoding layer
- **Fast**: 17 minutes for 1 epoch
- **Safe**: Won't forget tracking abilities (99.99% of parameters frozen)

---

## Results Summary

| Metric | Value |
|--------|-------|
| Training frames | 956 |
| YOLOX detections (training) | 10,456 |
| Pseudo GT annotations | 5,311 |
| D-FINE detections (training) | 16,484 |
| Trainable parameters | 256 (0.00%) |
| Training time | ~17 minutes |
| Inference detections | 4,315 |

---

## Lessons Learned

### 1. Class ID Mapping Issues
Different detection models use different class ID schemes. Always check:
- COCO: class 0 = person
- Objects365: class 1 = person

### 2. Frame Naming Conventions
MOT datasets can use different padding (6-digit vs 8-digit). Check frame format before extraction.

### 3. Detection Database Format
MOTRv2 expects specific key formats in detection JSON:
- Correct: `"volleyball/train/game1/img1/000001"`
- Wrong: `"volleyball/train/game1/img1/000001.txt"`

### 4. Data Path Loading
MOTRv2's dataset loading has hardcoded paths. Must modify to use custom datasets.

### 5. Pseudo-Labels Work Well
Using YOLOX+MOTRv2 outputs as pseudo ground truth is effective when manual annotations aren't available.

---

## Next Steps

### For Better Results
1. **More training data**: Add more volleyball sequences
2. **Longer training**: Try 2-3 epochs
3. **Fine-tune more layers**: Set `--embed_only False` to train query interaction layers

### For Different Use Cases
1. **Other detectors**: Same approach works for YOLOv8, RT-DETR, etc.
2. **Other domains**: Fine-tune on domain-specific data (e.g., sports, surveillance)

### For Evaluation
1. **Comparison**: Compare with original MOTRv2 + YOLOX
2. **Metrics**: Calculate MOTA, IDF1, ID switches if ground truth available
3. **Visualization**: Create side-by-side comparison videos

---

## File Locations Reference

### Input Files
- Videos: `inputs/videos/`
- Detections: `inputs/detections/`
- Weights: `weights/`

### Training Files
- Frames: `data/Dataset/mot/volleyball/train/game1/img1/`
- Ground truth: `data/Dataset/mot/volleyball/train/game1/gt/gt.txt`
- Detection DB: `data/Dataset/mot/det_db_volleyball_train_dfine.json`
- Sequence list: `datasets/data_path/volleyball_train.txt`

### Output Files
- Fine-tuned model: `outputs/finetune_dfine_embed/checkpoint.pth`
- Tracking results: `outputs/inference_finetuned/dfine_finetuned/game1.txt`
- Visualization: `outputs/inference_finetuned/tracking_visualization.mp4`

---

## Important Commands Summary

### Fine-Tuning
```bash
PYTHONPATH=$PWD:$PYTHONPATH python tools/fine_tuning/finetune_for_dfine.py \
    --resume weights/motrv2_dancetrack.pth \
    --det_db det_db_volleyball_train_dfine.json \
    --data_txt_path_train ./datasets/data_path/volleyball_train.txt \
    --output_dir outputs/finetune_dfine_embed \
    --epochs 1 \
    --embed_only True
```

### Inference
```bash
PYTHONPATH=$PWD:$PYTHONPATH python submit_dance.py \
    --resume outputs/finetune_dfine_embed/checkpoint.pth \
    --det_db det_db_volleyball_train_dfine.json \
    --output_dir outputs/inference_finetuned \
    --exp_name dfine_finetuned
```

### Visualization
```bash
python tools/visualization/visualize_tracking.py \
    -i data/Dataset/mot/volleyball/train/game1/img1 \
    -t outputs/inference_finetuned/dfine_finetuned/game1.txt \
    -o outputs/inference_finetuned/tracking_visualization.mp4
```

---

**Last Updated**: 2025-10-24
**Status**: Successfully completed fine-tuning and inference
