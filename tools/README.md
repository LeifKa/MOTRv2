# MOTRv2 Tools

Organized helper scripts for MOTRv2 tracking and fine-tuning.

## 📁 Directory Structure

```
tools/
├── fine_tuning/          # Fine-tune MOTRv2 for D-FINE
│   └── finetune_for_dfine.py
├── data_prep/            # Prepare training data
│   └── convert_to_gt.py
├── visualization/        # Create tracking videos
│   └── visualize_tracking.py
├── legacy/               # Old/deprecated scripts
└── README.md (this file)
```

## 🔧 Scripts

### fine_tuning/finetune_for_dfine.py
**Purpose:** Fine-tune yolox_embed layer to work with D-FINE detections

**Usage:**
```bash
cd MOTRv2/tools/fine_tuning

# Using config file
python finetune_for_dfine.py @../../../configs/motrv2/finetune_dfine.args \
    --resume ../../motrv2_checkpoint.pth \
    --det_db ../../det_db_train.json

# Custom settings
python finetune_for_dfine.py \
    --meta_arch motr \
    --dataset_file e2e_dance \
    --epochs 2 \
    --lr 1e-5 \
    --resume ../../motrv2_checkpoint.pth \
    --det_db ../../det_db_train.json \
    --mot_path ../../data/Dataset/mot \
    --output_dir ../../outputs/finetune \
    --embed_only True
```

---

### data_prep/convert_to_gt.py
**Purpose:** Convert tracking results to ground truth format for training

**Usage:**
```bash
cd MOTRv2/tools/data_prep

# Single file
python convert_to_gt.py \
    --input ../../outputs/pseudo_labels/game1.txt \
    --output ../../data/Dataset/mot/volleyball/train/game1/gt/gt.txt

# Batch mode
python convert_to_gt.py \
    --input ../../outputs/pseudo_labels/ \
    --output ../../data/Dataset/mot/volleyball/train/ \
    --batch
```

---

### visualization/visualize_tracking.py
**Purpose:** Create videos with tracking visualizations

**Usage:**
```bash
cd MOTRv2/tools/visualization

# Single result
python visualize_tracking.py \
    --images ../../data/Dataset/mot/volleyball/test/test1/img1 \
    --tracking ../../outputs/results/test1.txt \
    --output tracking_viz.mp4

# Compare two results
python visualize_tracking.py \
    --images ../../data/Dataset/mot/volleyball/test/test1/img1 \
    --tracking ../../outputs/finetuned/test1.txt \
    --tracking2 ../../outputs/original/test1.txt \
    --output comparison.mp4 \
    --labels "Fine-tuned" "Original"
```

---

## 📖 Documentation

For detailed guides, see:
- [../docs/00-OVERVIEW.md](../docs/00-OVERVIEW.md) - Overview
- [../docs/01-STEP_BY_STEP.md](../docs/01-STEP_BY_STEP.md) - Complete workflow
- [../docs/02-QUICK_COMMANDS.md](../docs/02-QUICK_COMMANDS.md) - Quick reference

## ⚠️ Note on Paths

All scripts expect to be run from their own directory. Use relative paths from there:
```bash
# ✓ Correct
cd MOTRv2/tools/fine_tuning
python finetune_for_dfine.py --resume ../../checkpoint.pth

# ✗ Wrong
python MOTRv2/tools/fine_tuning/finetune_for_dfine.py --resume checkpoint.pth
```

---

**Navigate:** [Main README](../../README.md) | [MOTRv2 Docs](../docs/) | [Configs](../../../configs/)
