# Step-by-Step Guide: Fine-tuning yolox_embed for D-FINE

This guide will walk you through fine-tuning MOTRv2's `yolox_embed` layer to work better with D-FINE detections, then testing on your beach volleyball clip.

## Understanding What We Need

For fine-tuning MOTRv2, we need **TWO types of data**:

1. **Detection Proposals** (from D-FINE) - These are the INPUTS to MOTRv2
2. **Ground Truth Annotations** (tracking labels) - These are what we TRAIN against

### Important Clarification
- D-FINE creates DETECTIONS (bounding boxes), not tracking labels
- Ground truth needs TRACK IDs (which person is which across frames)
- We can either:
  - Use manual annotations (if you have them)
  - **Generate pseudo-labels** from YOLOX+MOTRv2 (recommended - much easier!)

---

## STEP 1: Generate Training D-FINE Detections (Unfiltered)

For training, we want MORE detections with LESS filtering, so the model learns from richer data.

### 1.1 Create a Training Config for D-FINE

Create `D-FINE/motrv2_training_config.json`:

```json
{
  "allowed_classes": [0],
  "score_threshold": 0.3,
  "comment": "Class 0 = person in COCO. Lower threshold for training."
}
```

**Why these settings:**
- `allowed_classes: [0]` - Only persons (class 0 in COCO), remove ball filtering
- `score_threshold: 0.3` - Lower than inference (0.7) to keep more proposals
- This gives the model more examples to learn from

### 1.2 Run D-FINE on Your Training Videos

```bash
cd D-FINE

# Generate detections for training
python tools/inference/torch_inf.py \
    -c configs/dfine/dfine_hgnetv2_l_coco.yml \
    -r dfine_l_obj365.pth \
    --input /path/to/your/training/video.mp4 \
    --motrv2 \
    --motrv2_config motrv2_training_config.json \
    --sequence_name volleyball/train/game1

# This will create: det_db_motrv2.json
```

**Important:** Run this on ALL your training videos, creating different sequence names:
- `volleyball/train/game1`
- `volleyball/train/game2`
- etc.

### 1.3 Verify the Detections

```bash
# Check what was generated
python -c "
import json
with open('det_db_motrv2.json') as f:
    data = json.load(f)
    print(f'Frames: {len(data)}')
    first_key = list(data.keys())[0]
    print(f'First frame: {first_key}')
    print(f'Detections: {len(data[first_key])}')
    print(f'Sample: {data[first_key][0] if data[first_key] else \"No detections\"}')
"
```

Expected output:
```
Frames: 300
First frame: volleyball/train/game1/img1/000001
Detections: 15
Sample: 771.57,604.20,73.07,229.56,0.523437
```

---

## STEP 2: Prepare Ground Truth Annotations

You have two options:

### Option A: Use Existing Ground Truth (If You Have It)

If you have manual annotations in MOT format:
```
volleyball/train/game1/gt/gt.txt
```

Format: `frame,id,x,y,w,h,conf,class,vis`
```
1,1,771,604,73,229,1,1,1
1,2,667,636,64,140,1,1,1
2,1,771,602,73,231,1,1,1
2,2,669,637,62,140,1,1,1
```

**If you have this, SKIP to Step 3.**

### Option B: Generate Pseudo-Labels from YOLOX+MOTRv2 (Recommended)

Since you said YOLOX+MOTRv2 works better, we'll use its outputs as "ground truth" for training with D-FINE!

#### 2.1 Generate YOLOX Detections

First, we need YOLOX detections on your training videos:

```bash
cd YOLOX

# Run YOLOX on training video
python tools/demo.py video \
    -n yolox-x \
    -c yolox_x.pth \
    --path /path/to/training/video.mp4 \
    --save_result \
    --output_format mot

# This creates tracking results
```

#### 2.2 Run MOTRv2 with YOLOX to Generate Pseudo-Labels

```bash
cd ../MOTRv2

# Run MOTRv2 with YOLOX detections
python submit_dance.py \
    --resume motrv2_checkpoint.pth \
    --det_db det_db_yolox_training.json \
    --mot_path ./data/Dataset/mot \
    --score_threshold 0.6 \
    --output_dir outputs/pseudo_labels

# This creates: outputs/pseudo_labels/volleyball_train_game1.txt
```

#### 2.3 Convert Tracking Results to Ground Truth Format

Create a script to convert MOTRv2 output to ground truth format:

```python
# convert_to_gt.py
import os
import shutil

def motrv2_to_gt(tracking_file, output_gt_file):
    """
    Convert MOTRv2 tracking output to MOT ground truth format.

    MOTRv2 output: frame,id,x1,y1,w,h,1,-1,-1,-1
    Ground truth: frame,id,x,y,w,h,1,1,1
    """
    with open(tracking_file, 'r') as f:
        lines = f.readlines()

    gt_lines = []
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 7:
            frame, track_id, x, y, w, h = parts[:6]
            # Convert to GT format: frame,id,x,y,w,h,conf=1,class=1,vis=1
            gt_line = f"{frame},{track_id},{x},{y},{w},{h},1,1,1\n"
            gt_lines.append(gt_line)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_gt_file), exist_ok=True)

    with open(output_gt_file, 'w') as f:
        f.writelines(gt_lines)

    print(f"✓ Converted {len(gt_lines)} annotations")
    print(f"  Output: {output_gt_file}")

if __name__ == "__main__":
    # Convert your tracking results to ground truth
    motrv2_to_gt(
        tracking_file="outputs/pseudo_labels/volleyball_train_game1.txt",
        output_gt_file="data/Dataset/mot/volleyball/train/game1/gt/gt.txt"
    )
```

Run it:
```bash
python convert_to_gt.py
```

---

## STEP 3: Set Up Training Data Structure

MOTRv2 expects a specific directory structure:

```
data/Dataset/mot/
└── volleyball/
    └── train/
        ├── game1/
        │   ├── img1/
        │   │   ├── 000001.jpg
        │   │   ├── 000002.jpg
        │   │   └── ...
        │   └── gt/
        │       └── gt.txt
        ├── game2/
        │   ├── img1/
        │   └── gt/
        └── game3/
            ├── img1/
            └── gt/
```

### 3.1 Extract Video Frames

```bash
# Extract frames from your training video
mkdir -p data/Dataset/mot/volleyball/train/game1/img1

ffmpeg -i /path/to/training/video.mp4 \
    -qscale:v 2 \
    data/Dataset/mot/volleyball/train/game1/img1/%06d.jpg
```

### 3.2 Verify Structure

```bash
# Check the structure
tree data/Dataset/mot/volleyball/train/game1/

# Should show:
# game1/
# ├── img1/
# │   ├── 000001.jpg
# │   ├── 000002.jpg
# │   └── ...
# └── gt/
#     └── gt.txt
```

---

## STEP 4: Create Training Data List

MOTRv2 needs a text file listing training sequences:

```bash
# Create datasets/data_path/volleyball_train.txt
mkdir -p datasets/data_path

cat > datasets/data_path/volleyball_train.txt << 'EOF'
volleyball/train/game1
volleyball/train/game2
volleyball/train/game3
EOF
```

---

## STEP 5: Merge Detection Databases

If you generated D-FINE detections for multiple videos, merge them:

```python
# merge_detections.py
import json
import glob

def merge_detection_dbs(input_files, output_file):
    """Merge multiple detection JSON files into one."""
    merged = {}

    for file in input_files:
        print(f"Loading {file}...")
        with open(file, 'r') as f:
            data = json.load(f)
            merged.update(data)

    print(f"\nTotal frames: {len(merged)}")

    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"✓ Saved to {output_file}")

if __name__ == "__main__":
    # Find all detection files
    det_files = glob.glob("D-FINE/det_db_*.json")

    merge_detection_dbs(
        input_files=det_files,
        output_file="MOTRv2/det_db_volleyball_train.json"
    )
```

```bash
python merge_detections.py
```

---

## STEP 6: Run Fine-Tuning!

Now we're ready to fine-tune!

### 6.1 Verify You Have Everything

Checklist:
- ✅ D-FINE detections: `det_db_volleyball_train.json`
- ✅ Training frames: `data/Dataset/mot/volleyball/train/game1/img1/*.jpg`
- ✅ Ground truth: `data/Dataset/mot/volleyball/train/game1/gt/gt.txt`
- ✅ Training list: `datasets/data_path/volleyball_train.txt`
- ✅ MOTRv2 checkpoint: `motrv2_checkpoint.pth`

### 6.2 Run Fine-Tuning Script

```bash
cd MOTRv2

# Fine-tune yolox_embed (1-2 hours on GPU)
python finetune_for_dfine.py \
    --meta_arch motr \
    --dataset_file e2e_dance \
    --epochs 2 \
    --with_box_refine \
    --lr 1e-5 \
    --lr_backbone 0 \
    --lr_drop 4 \
    --resume motrv2_checkpoint.pth \
    --batch_size 1 \
    --sample_mode random_interval \
    --sample_interval 10 \
    --sampler_lengths 5 \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer QIMv2 \
    --num_queries 10 \
    --det_db det_db_volleyball_train.json \
    --mot_path ./data/Dataset/mot \
    --data_txt_path_train ./datasets/data_path/volleyball_train.txt \
    --output_dir outputs/finetune_dfine_embed \
    --exp_name dfine_volleyball \
    --embed_only True \
    --num_workers 2
```

### 6.3 Monitor Training

You should see output like:
```
================================================================================
Setting up partial training: yolox_embed ONLY
================================================================================

✓ TRAINING: yolox_embed.weight                               shape=[1, 256]

================================================================================
Parameter Summary:
  Trainable:              256 ( 0.00%)
  Frozen:      41,494,528 (99.99%)
  Total:       41,494,784
================================================================================

🚀 Starting fine-tuning
...
Epoch 1/2
Train: Loss: 2.45 -> 2.12 -> 1.89 -> 1.67 (decreasing is good!)
...
```

**What to watch for:**
- ✅ Loss should decrease over time
- ✅ Should take ~30-60 minutes per epoch
- ❌ If loss increases or stays flat, learning rate may be too high

### 6.4 Checkpoints

Fine-tuning will save:
- `outputs/finetune_dfine_embed/checkpoint.pth` - Latest
- `outputs/finetune_dfine_embed/checkpoint0000.pth` - Epoch 1
- `outputs/finetune_dfine_embed/checkpoint0001.pth` - Epoch 2

---

## STEP 7: Test on Beach Volleyball Clip!

Now let's test the fine-tuned model on your test video.

### 7.1 Generate D-FINE Detections for Test Video

```bash
cd D-FINE

# Use INFERENCE config (higher threshold, filtered)
cat > motrv2_inference_config.json << 'EOF'
{
  "allowed_classes": [0],
  "score_threshold": 0.7
}
EOF

# Generate test detections
python tools/inference/torch_inf.py \
    -c configs/dfine/dfine_hgnetv2_l_coco.yml \
    -r dfine_l_obj365.pth \
    --input /path/to/beach_volleyball_test.mp4 \
    --motrv2 \
    --motrv2_config motrv2_inference_config.json \
    --sequence_name volleyball/test/test1

# Copy to MOTRv2
cp det_db_motrv2.json ../MOTRv2/det_db_beach_test.json
```

### 7.2 Extract Test Frames

```bash
cd ../MOTRv2

mkdir -p data/Dataset/mot/volleyball/test/test1/img1

ffmpeg -i /path/to/beach_volleyball_test.mp4 \
    -qscale:v 2 \
    data/Dataset/mot/volleyball/test/test1/img1/%06d.jpg
```

### 7.3 Run Inference with FINE-TUNED Model

```bash
python submit_dance.py \
    --resume outputs/finetune_dfine_embed/checkpoint0001.pth \
    --det_db det_db_beach_test.json \
    --mot_path ./data/Dataset/mot \
    --score_threshold 0.7 \
    --output_dir outputs/test_finetuned \
    --exp_name beach_test_finetuned

# Results saved to: outputs/test_finetuned/test1.txt
```

### 7.4 Run Inference with ORIGINAL Model (for comparison)

```bash
python submit_dance.py \
    --resume motrv2_checkpoint.pth \
    --det_db det_db_beach_test.json \
    --mot_path ./data/Dataset/mot \
    --score_threshold 0.7 \
    --output_dir outputs/test_original \
    --exp_name beach_test_original

# Results saved to: outputs/test_original/test1.txt
```

### 7.5 Visualize Results

Create a visualization script:

```python
# visualize_tracking.py
import cv2
import os
from collections import defaultdict

def parse_tracking_results(tracking_file):
    """Parse MOT format tracking results."""
    tracks = defaultdict(list)

    with open(tracking_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])

            tracks[frame].append({
                'id': track_id,
                'bbox': (x, y, w, h)
            })

    return tracks

def visualize_tracking(image_dir, tracking_file, output_video):
    """Create video with tracking visualizations."""
    tracks = parse_tracking_results(tracking_file)

    # Get image files
    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

    if not images:
        print(f"No images found in {image_dir}")
        return

    # Get video properties
    first_img = cv2.imread(os.path.join(image_dir, images[0]))
    height, width = first_img.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (width, height))

    # Generate colors for track IDs
    colors = {}

    for idx, img_name in enumerate(images, 1):
        img_path = os.path.join(image_dir, img_name)
        frame = cv2.imread(img_path)

        # Draw detections for this frame
        if idx in tracks:
            for detection in tracks[idx]:
                track_id = detection['id']
                x, y, w, h = detection['bbox']

                # Generate consistent color for each ID
                if track_id not in colors:
                    import random
                    colors[track_id] = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )

                color = colors[track_id]

                # Draw bounding box
                cv2.rectangle(frame,
                            (int(x), int(y)),
                            (int(x + w), int(y + h)),
                            color, 2)

                # Draw ID label
                label = f"ID: {track_id}"
                cv2.putText(frame, label,
                          (int(x), int(y) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.6, color, 2)

        # Add frame number
        cv2.putText(frame, f"Frame: {idx}",
                  (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  1, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"✓ Video saved to {output_video}")

if __name__ == "__main__":
    # Visualize fine-tuned results
    visualize_tracking(
        image_dir="data/Dataset/mot/volleyball/test/test1/img1",
        tracking_file="outputs/test_finetuned/test1.txt",
        output_video="outputs/beach_finetuned_tracking.mp4"
    )

    # Visualize original results
    visualize_tracking(
        image_dir="data/Dataset/mot/volleyball/test/test1/img1",
        tracking_file="outputs/test_original/test1.txt",
        output_video="outputs/beach_original_tracking.mp4"
    )

    print("\n✅ Done! Compare the two videos:")
    print("   - outputs/beach_finetuned_tracking.mp4")
    print("   - outputs/beach_original_tracking.mp4")
```

Run it:
```bash
python visualize_tracking.py
```

### 7.6 Compare Results

Watch both videos and look for:
- **Fewer ID switches** (same person keeps same ID)
- **More stable tracks** (boxes don't jump around)
- **Better occlusion handling**
- **Consistent tracking across frames**

---

## STEP 8: Evaluate Quantitatively (Optional)

If you have ground truth for your test video:

```bash
# Install MOT evaluation tools
pip install motmetrics

# Run evaluation
python -c "
import motmetrics as mm
import pandas as pd

# Load ground truth
gt = mm.io.loadtxt('data/Dataset/mot/volleyball/test/test1/gt/gt.txt')

# Load tracking results
finetuned = mm.io.loadtxt('outputs/test_finetuned/test1.txt')
original = mm.io.loadtxt('outputs/test_original/test1.txt')

# Create accumulator
acc_ft = mm.MOTAccumulator(auto_id=True)
acc_or = mm.MOTAccumulator(auto_id=True)

# Compute metrics
mh = mm.metrics.create()

# Compare
summary = mh.compute_many(
    [acc_ft, acc_or],
    names=['Fine-tuned', 'Original']
)

print(summary)
"
```

---

## Quick Start Commands (TL;DR)

If you just want to run everything quickly:

```bash
# 1. Generate training D-FINE detections
cd D-FINE
python tools/inference/torch_inf.py -c configs/dfine/dfine_hgnetv2_l_coco.yml \
    -r dfine_l_obj365.pth --input train_video.mp4 --motrv2 \
    --sequence_name volleyball/train/game1

# 2. Generate pseudo ground truth (using YOLOX+MOTRv2)
cd ../MOTRv2
python submit_dance.py --resume motrv2.pth --det_db det_db_yolox.json \
    --output_dir outputs/pseudo_gt

# 3. Convert to GT format
python convert_to_gt.py

# 4. Extract frames
ffmpeg -i train_video.mp4 data/Dataset/mot/volleyball/train/game1/img1/%06d.jpg

# 5. Create training list
echo "volleyball/train/game1" > datasets/data_path/volleyball_train.txt

# 6. Fine-tune
python finetune_for_dfine.py @configs/finetune_dfine.args \
    --resume motrv2.pth --epochs 2

# 7. Test on beach volleyball
cd ../D-FINE
python tools/inference/torch_inf.py -c configs/dfine/dfine_hgnetv2_l_coco.yml \
    -r dfine_l_obj365.pth --input beach_test.mp4 --motrv2 \
    --sequence_name volleyball/test/test1

cd ../MOTRv2
python submit_dance.py --resume outputs/finetune_dfine_embed/checkpoint0001.pth \
    --det_db det_db_beach_test.json --output_dir outputs/test_result
```

---

## Troubleshooting

### "FileNotFoundError: det_db_volleyball_train.json"
**Fix:** Make sure D-FINE output is copied/moved to MOTRv2 directory

### "No such file or directory: volleyball/train/game1"
**Fix:** Check your data structure matches the expected layout

### Loss is not decreasing
**Fix:** Try lower learning rate: `--lr 5e-6`

### Out of memory
**Fix:** Use `--batch_size 1` and `--sampler_lengths 3`

### "RuntimeError: CUDA out of memory"
**Fix:** Reduce `--num_workers` to 0 or use `--use_checkpoint` flag

---

## Expected Timeline

- **Setup (Steps 1-5):** 1-2 hours
- **Training (Step 6):** 1-2 hours
- **Testing (Step 7):** 30 minutes
- **Total:** ~3-5 hours

---

Good luck! Let me know if you hit any issues. 🚀
