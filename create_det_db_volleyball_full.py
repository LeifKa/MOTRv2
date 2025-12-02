#!/usr/bin/env python3
"""
Create detection database for volleyball_full dataset from ground truth annotations.
Converts MOT format ground truth to detection database JSON format.
"""
import json
import os
from pathlib import Path
from collections import defaultdict

def read_mot_gt(gt_file):
    """
    Read MOT format ground truth file.
    Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    """
    detections = defaultdict(list)

    with open(gt_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                frame_id = int(parts[0])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = float(parts[6]) if len(parts) > 6 else 1.0

                # Store detection as "x,y,w,h,conf\n"
                det_str = f"{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.6f}\n"
                detections[frame_id].append(det_str)

    return detections

def create_detection_db(base_path, sequences, split_name):
    """
    Create detection database from ground truth annotations.

    Args:
        base_path: Path to MOTRv2 data directory
        sequences: List of sequence names to process
        split_name: Name of the split (e.g., 'train', 'valid')

    Returns:
        Dictionary mapping image paths to detection lists
    """
    det_db = {}

    for seq_name in sequences:
        seq_path = base_path / split_name / seq_name
        gt_file = seq_path / "gt" / "gt.txt"

        if not gt_file.exists():
            print(f"Warning: Ground truth file not found: {gt_file}")
            continue

        print(f"Processing sequence: {split_name}/{seq_name}")

        # Read ground truth detections
        detections = read_mot_gt(gt_file)

        # Create entries for each frame
        for frame_id, dets in detections.items():
            # Image path format: volleyball_full/train/seq1/img1/000001
            img_key = f"volleyball_full/{split_name}/{seq_name}/img1/{frame_id:06d}"
            det_db[img_key] = dets

        print(f"  Added {len(detections)} frames with detections")

    return det_db

def main():
    # Paths
    base_path = Path("MOTRv2/data/Dataset/mot/volleyball_full")
    output_file = Path("MOTRv2/data/Dataset/mot/det_db_volleyball_full_dfine.json")

    # Process train and valid splits
    det_db = {}

    # Train sequences
    train_sequences = ["seq1"]  # Add more sequences if you have them
    print("\n=== Processing Training Sequences ===")
    train_db = create_detection_db(base_path, train_sequences, "train")
    det_db.update(train_db)

    # Valid sequences
    valid_sequences = ["seq1"]  # Add more sequences if you have them
    print("\n=== Processing Validation Sequences ===")
    valid_db = create_detection_db(base_path, valid_sequences, "valid")
    det_db.update(valid_db)

    # Save detection database
    print(f"\n=== Saving Detection Database ===")
    print(f"Total entries: {len(det_db)}")
    print(f"Output file: {output_file}")

    with open(output_file, 'w') as f:
        json.dump(det_db, f, indent=2)

    print("\nDetection database created successfully!")

    # Print statistics
    total_detections = sum(len(dets) for dets in det_db.values())
    print(f"\nStatistics:")
    print(f"  Total frames: {len(det_db)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Average detections per frame: {total_detections / len(det_db):.2f}")

if __name__ == "__main__":
    main()
