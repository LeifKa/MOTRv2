# MOTRv2 Directory Structure

## Organized File Layout

The project has been reorganized for better file management and clarity.

### 📁 Input Files
```
inputs/
├── detections/                 # Detection JSON files (all formats)
│   ├── Beispielvideo_Beach_YOLOX.json
│   ├── det_db_beach_volleyball.json
│   ├── det_db_volleyball_train.json
│   └── det_db_volleyball_train_dfine.json
└── videos/                     # Source video files
    ├── Beispielvideo_Beach_YOLOX.mp4
    └── Sequenz_Beach.mp4
```

### 🎯 Model Weights
```
weights/
├── motrv2_dancetrack.pth
└── r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth
```

### 🗂️ Training Data
```
data/Dataset/mot/
├── det_db_volleyball_train_dfine.json  # Copy of detection file (required here)
└── volleyball/train/game1/
    ├── img1/                            # 956 frames (6-digit: 000001.jpg)
    └── gt/                              # Ground truth
        └── gt.txt                       # 5,311 annotations
```

### 📊 Outputs
```
outputs/
├── finetune_dfine_embed/
│   └── checkpoint.pth                   # Fine-tuned model
├── inference_finetuned/
│   ├── dfine_finetuned/
│   │   └── game1.txt                    # Tracking results (4,315 detections)
│   └── tracking_visualization.mp4        # Visualization video
└── pseudo_labels/train_pseudo_gt/
    └── game1.txt                         # Pseudo GT from YOLOX
```

### 🛠️ Tools
```
tools/
├── fine_tuning/
│   └── finetune_for_dfine.py           # Fine-tuning script
├── conversion/
│   └── convert_to_gt.py                # Convert tracking to GT format
└── visualization/
    └── visualize_tracking.py           # Create visualization videos
```

### 📚 Documentation
```
docs/
├── 00-OVERVIEW.md                      # Complete overview
├── 01-STEP_BY_STEP.md                  # Detailed guide
├── 02-QUICK_COMMANDS.md                # Command reference
├── 03-ADVANCED.md                      # Advanced options
├── 04-THEORY.md                        # Technical explanation
└── 05-IMPLEMENTATION_NOTES.md          # Implementation details
```

---

## Key Configuration Files

### Training Sequence List
`datasets/data_path/volleyball_train.txt`:
```
volleyball/train/game1
```

### D-FINE Training Config
`D-FINE/motrv2_training_config.json`:
```json
{
  "score_threshold": 0.3
}
```

---

## Important Notes

### Frame Format
- **Correct**: `000001.jpg` (6 digits)
- **Wrong**: `00000001.jpg` (8 digits)

### Detection Database Keys
- **Correct**: `"volleyball/train/game1/img1/000001"`
- **Wrong**: `"volleyball/train/game1/img1/000001.txt"`

### Detection File Location
For training, detection JSON must be in **both** locations:
1. `inputs/detections/` (organized storage)
2. `data/Dataset/mot/` (required by training script)

---

## Quick Navigation

**Want to...**

- **Run fine-tuning?** → See [docs/02-QUICK_COMMANDS.md](docs/02-QUICK_COMMANDS.md)
- **Understand the process?** → See [docs/01-STEP_BY_STEP.md](docs/01-STEP_BY_STEP.md)
- **Check implementation details?** → See [docs/05-IMPLEMENTATION_NOTES.md](docs/05-IMPLEMENTATION_NOTES.md)
- **See complete summary?** → See [SUMMARY.md](SUMMARY.md)

---

## File Size Reference

| File | Size | Purpose |
|------|------|---------|
| motrv2_dancetrack.pth | 161M | Pretrained MOTRv2 model |
| r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth | 467M | DETR backbone weights |
| Sequenz_Beach.mp4 | 76M | Training video |
| det_db_volleyball_train_dfine.json | 763K | D-FINE detections (16,484) |
| checkpoint.pth (fine-tuned) | ~161M | Fine-tuned model |

---

## Directory Cleanup Checklist

After setup, you should have:

- [x] All JSON files in `inputs/detections/`
- [x] All videos in `inputs/videos/`
- [x] All weights in `weights/`
- [x] No loose files in root directory
- [x] Detection JSON copied to `data/Dataset/mot/`
- [x] Training frames in `data/Dataset/mot/volleyball/train/game1/img1/`
- [x] Ground truth in `data/Dataset/mot/volleyball/train/game1/gt/gt.txt`

---

**Last Updated**: 2025-10-24
**Status**: ✅ Organized and documented
