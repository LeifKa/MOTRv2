# MOTRv2 Fine-Tuning Configurations

## Overview

This document describes all fine-tuning configurations for the Volleyball dataset with correct Ground Truth and D-FINE Detection Database.

## Training Strategies

### Minimal (1-2% of parameters)
**Trains only:** `yolox_embed`
- **Safest approach** - prevents catastrophic forgetting
- Best for when you want to adapt the detector slightly
- Fastest training

### Moderate (5-10% of parameters)
**Trains:** `yolox_embed` + `track_embed` + `class_embed`
- **Recommended for most cases**
- Adapts both detection and basic tracking
- Good balance between adaptation and stability

### Aggressive (15-20% of parameters)
**Trains:** `yolox_embed` + `track_embed` + `class_embed` + `query_interaction`
- **Most adaptation** - trains query interaction layers
- Best tracking performance but higher risk of forgetting
- Slowest training

## Learning Rates

All configurations test 3 learning rates with 5 epochs:

| LR      | Backbone LR | Description |
|---------|-------------|-------------|
| 5e-6    | 5e-7        | Conservative - safest, slowest learning |
| 1e-5    | 1e-6        | Moderate - balanced approach |
| 2e-5    | 2e-6        | Aggressive - faster learning, higher risk |

## Configuration Matrix (9 total configs)

| Strategy    | LR    | Config File | SLURM Script | Output Dir |
|-------------|-------|-------------|--------------|------------|
| **Minimal** | 5e-6  | `volleyball_finetune_minimal_lr5e6_ep5.args` | `run_volleyball_finetune_minimal_lr5e6_ep5.slurm` | `outputs/finetune_vb_minimal_lr5e6_ep5/` |
| Minimal     | 1e-5  | `volleyball_finetune_minimal_lr1e5_ep5.args` | `run_volleyball_finetune_minimal_lr1e5_ep5.slurm` | `outputs/finetune_vb_minimal_lr1e5_ep5/` |
| Minimal     | 2e-5  | `volleyball_finetune_minimal_lr2e5_ep5.args` | `run_volleyball_finetune_minimal_lr2e5_ep5.slurm` | `outputs/finetune_vb_minimal_lr2e5_ep5/` |
| **Moderate** | 5e-6  | `volleyball_finetune_moderate_lr5e6_ep5.args` | `run_volleyball_finetune_moderate_lr5e6_ep5.slurm` | `outputs/finetune_vb_moderate_lr5e6_ep5/` |
| Moderate    | 1e-5  | `volleyball_finetune_moderate_lr1e5_ep5.args` | `run_volleyball_finetune_moderate_lr1e5_ep5.slurm` | `outputs/finetune_vb_moderate_lr1e5_ep5/` |
| Moderate    | 2e-5  | `volleyball_finetune_moderate_lr2e5_ep5.args` | `run_volleyball_finetune_moderate_lr2e5_ep5.slurm` | `outputs/finetune_vb_moderate_lr2e5_ep5/` |
| **Aggressive** | 5e-6  | `volleyball_finetune_aggressive_lr5e6_ep5.args` | `run_volleyball_finetune_aggressive_lr5e6_ep5.slurm` | `outputs/finetune_vb_aggressive_lr5e6_ep5/` |
| Aggressive  | 1e-5  | `volleyball_finetune_aggressive_lr1e5_ep5.args` | `run_volleyball_finetune_aggressive_lr1e5_ep5.slurm` | `outputs/finetune_vb_aggressive_lr1e5_ep5/` |
| Aggressive  | 2e-5  | `volleyball_finetune_aggressive_lr2e5_ep5.args` | `run_volleyball_finetune_aggressive_lr2e5_ep5.slurm` | `outputs/finetune_vb_aggressive_lr2e5_ep5/` |

## Usage

### Step 1: Create Detection Database (REQUIRED - Only once!)

```bash
cd D-FINE/tools/inference
python create_detection_db_for_finetune.py
```

This creates `det_db_volleyball_finetune_dfine.json` with real D-FINE detections.

### Step 2: Submit Training Jobs

**Submit individual job:**
```bash
cd MOTRv2
sbatch run_volleyball_finetune_moderate_lr1e5_ep5.slurm
```

**Submit all jobs (parallel training):**
```bash
cd MOTRv2
for script in run_volleyball_finetune_*_ep5.slurm; do
    sbatch "$script"
done
```

**Submit by strategy:**
```bash
# All minimal
sbatch run_volleyball_finetune_minimal_lr5e6_ep5.slurm
sbatch run_volleyball_finetune_minimal_lr1e5_ep5.slurm
sbatch run_volleyball_finetune_minimal_lr2e5_ep5.slurm

# All moderate
sbatch run_volleyball_finetune_moderate_lr5e6_ep5.slurm
sbatch run_volleyball_finetune_moderate_lr1e5_ep5.slurm
sbatch run_volleyball_finetune_moderate_lr2e5_ep5.slurm

# All aggressive
sbatch run_volleyball_finetune_aggressive_lr5e6_ep5.slurm
sbatch run_volleyball_finetune_aggressive_lr1e5_ep5.slurm
sbatch run_volleyball_finetune_aggressive_lr2e5_ep5.slurm
```

### Step 3: Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Watch specific log
tail -f logs/finetune_vb_moderate_lr1e5_ep5_<JOB_ID>.out

# Check trainable parameters (should differ by strategy!)
grep "Trainable params:" logs/finetune_vb_*_ep5_*.out
```

## Expected Results

### Trainable Parameters by Strategy

- **Minimal:** ~1-2% (only yolox_embed)
- **Moderate:** ~5-10% (+ track_embed, class_embed)
- **Aggressive:** ~15-20% (+ query_interaction)

The log output will show EXACTLY which layers are being trained.

### Model Checkpoints

After training, each configuration will have:
```
outputs/finetune_vb_<strategy>_lr<lr>_ep5/
├── checkpoint.pth          # Latest checkpoint
├── checkpoint0.pth         # Epoch 0
├── checkpoint1.pth         # Epoch 1
├── checkpoint2.pth         # Epoch 2
├── checkpoint3.pth         # Epoch 3
└── checkpoint4.pth         # Epoch 4
```

## Recommendations

### Start with these 3 configs:
1. **Moderate LR 1e-5** (balanced, safest bet)
2. **Aggressive LR 1e-5** (more tracking adaptation)
3. **Minimal LR 2e-5** (fast baseline)

### If you have time, run all 9:
This gives you a complete grid search over strategy × learning rate.

## Key Differences from Previous Training

✅ **Fixed:**
- Uses correct Ground Truth: `volleyball/finetune/gt/`
- Uses real D-FINE Detection DB: `det_db_volleyball_finetune_dfine.json`
- Proper parameter freezing with `--train_strategy`
- Clear naming for all outputs

❌ **Old (broken):**
- Used Detection DB with GT boxes (score=1.0)
- No proper parameter freezing
- Both strategies trained ~3% of parameters

## Troubleshooting

**Error: "Detection database not found"**
```bash
cd D-FINE/tools/inference
python create_detection_db_for_finetune.py
```

**Training uses wrong % of parameters:**
Check the log for "FINE-TUNING STRATEGY" section. It should show different percentages:
```
Trainable params: 1,234,567 (2.34%)   # minimal
Trainable params: 5,678,901 (8.67%)   # moderate
Trainable params: 9,876,543 (15.23%)  # aggressive
```

**Job fails immediately:**
Check: `tail -100 logs/finetune_vb_*_<JOB_ID>.out`

## Evaluation

After training, evaluate models:
```bash
python main.py \
    --eval \
    --resume outputs/finetune_vb_moderate_lr1e5_ep5/checkpoint4.pth \
    --mot_path ./data/Dataset/mot \
    --data_txt_path_val ./datasets/data_path/volleyball_test.txt
```

Compare tracking metrics (HOTA, MOTA, IDF1) across all 9 configurations to find the best combination.
