# Next Steps for Fine-Tuning

## What's Been Done ✅

1. **Training frames**: 956 frames extracted from Sequenz_Beach.mp4
   - Location: `data/Dataset/mot/volleyball/train/game1/img1/`

2. **YOLOX detections**: Generated on training data (10,456 detections)
   - Location: `det_db_yolox_train.json`

3. **Training list**: Created
   - Location: `datasets/data_path/volleyball_train.txt`

4. **SLURM Job Submitted**: Job #34702 - MOTRv2 tracking with YOLOX
   - Status: PENDING (waiting for GPU resources)
   - Purpose: Generate pseudo ground truth from YOLOX+MOTRv2

---

## What You Need to Do Next

### Step 1: Wait for SLURM Job to Complete

Check status:
```bash
squeue -u $USER
```

When job 34702 completes, verify output:
```bash
cd /home/es/es_es/es_lekamt00/BeachKI/MOTRv2
ls -lh outputs/pseudo_labels/game1.txt
```

### Step 2: Convert Tracking to Ground Truth

```bash
python tools/data_prep/convert_to_gt.py \
    --input outputs/pseudo_labels/game1.txt \
    --output data/Dataset/mot/volleyball/train/game1/gt/gt.txt
```

### Step 3: Generate D-FINE Detections on Training Data

```bash
cd ../D-FINE
sbatch run_dfine_training.slurm
```

Wait for completion, then verify:
```bash
ls -lh det_db_motrv2.json
ls -lh ../MOTRv2/det_db_volleyball_train_dfine.json
```

### Step 4: Run Fine-Tuning (MANUAL - YOU DO THIS)

**Refer to**: `docs/02-QUICK_COMMANDS.md` Step 5

Basic command:
```bash
cd /home/es/es_es/es_lekamt00/BeachKI/MOTRv2

python tools/fine_tuning/finetune_for_dfine.py \
    --meta_arch motr \
    --dataset_file e2e_dance \
    --epochs 2 \
    --with_box_refine \
    --lr 1e-5 \
    --lr_backbone 0 \
    --resume weights/motrv2_dancetrack.pth \
    --batch_size 1 \
    --sample_mode random_interval \
    --sample_interval 10 \
    --sampler_lengths 5 \
    --query_interaction_layer QIMv2 \
    --num_queries 10 \
    --det_db det_db_volleyball_train_dfine.json \
    --mot_path ./data/Dataset/mot \
    --data_txt_path_train ./datasets/data_path/volleyball_train.txt \
    --output_dir outputs/finetune_dfine_embed \
    --exp_name dfine_volleyball \
    --embed_only True \
    --num_workers 2
```

**Note**: This should be run on a GPU node. Consider creating a SLURM script.

---

## Full Documentation

For detailed explanations, see:
- **Quick Reference**: [docs/02-QUICK_COMMANDS.md](docs/02-QUICK_COMMANDS.md)
- **Step-by-Step Guide**: [docs/01-STEP_BY_STEP.md](docs/01-STEP_BY_STEP.md)
- **Overview**: [docs/00-OVERVIEW.md](docs/00-OVERVIEW.md)
- **Theory**: [docs/04-THEORY.md](docs/04-THEORY.md)

---

## File Locations

| Description | Path |
|-------------|------|
| Training frames | `data/Dataset/mot/volleyball/train/game1/img1/` (956 frames) |
| YOLOX detections | `det_db_yolox_train.json` (for pseudo GT) |
| D-FINE detections | `det_db_volleyball_train_dfine.json` (for training) |
| Ground truth | `data/Dataset/mot/volleyball/train/game1/gt/gt.txt` (Step 2) |
| Training list | `datasets/data_path/volleyball_train.txt` |
| Original weights | `weights/motrv2_dancetrack.pth` |
| Fine-tuned weights | `outputs/finetune_dfine_embed/checkpoint0001.pth` (Step 4 output) |
| Fine-tuning script | `tools/fine_tuning/finetune_for_dfine.py` |
| GT conversion script | `tools/data_prep/convert_to_gt.py` |

---

## Summary

**STOPPED AT**: Waiting for SLURM job #34702 (MOTRv2 tracking)

**YOU WILL RUN**: Step 4 (Fine-tuning) manually after Steps 1-3 complete

**CURRENT JOB STATUS**:
```bash
squeue -u $USER  # Check this periodically
```
