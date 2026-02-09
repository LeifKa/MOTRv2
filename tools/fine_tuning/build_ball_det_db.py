#!/usr/bin/env python3
"""
Build a detection database (det_db) JSON for MOTRv2 that includes ball detection proposals.

Converts ball_detections JSON files (from automated ball detection) into MOTRv2's
det_db format and optionally merges with an existing det_db (e.g. D-FINE player detections).

det_db format:
    {
        "sequence_path/img1/000001": ["x,y,w,h,confidence\n", ...],
        ...
    }

ball_detections format:
    {
        "1": [{"bbox": [x, y, w, h], "confidence": 0.24, "class_id": 32}],
        ...
    }

Usage:
    python build_ball_det_db.py \
        --ball_detections_dir /path/to/SportsMOT_Volleyball/ball_detections/train \
        --sequences_file /path/to/volleyball_train_with_ball_train.txt \
        --output /path/to/mot/det_db_volleyball_ball.json \
        [--existing_det_db /path/to/existing_det_db.json] \
        [--min_confidence 0.05]
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Build ball detection database for MOTRv2')
    parser.add_argument('--ball_detections_dir', required=True,
                        help='Directory containing *_ball_detections.json files')
    parser.add_argument('--sequences_file', required=True,
                        help='Data txt path file listing sequence paths (e.g. volleyball_train_with_ball_train.txt)')
    parser.add_argument('--output', required=True,
                        help='Output det_db JSON file path')
    parser.add_argument('--existing_det_db', default=None,
                        help='Optional existing det_db JSON to merge with (e.g. D-FINE player detections)')
    parser.add_argument('--min_confidence', type=float, default=0.05,
                        help='Minimum confidence threshold for ball detections (default: 0.05)')
    args = parser.parse_args()

    # Load existing det_db if provided
    if args.existing_det_db and os.path.exists(args.existing_det_db):
        print(f"Loading existing det_db from: {args.existing_det_db}")
        with open(args.existing_det_db) as f:
            det_db = json.load(f)
        print(f"  Existing entries: {len(det_db)} frames")
    else:
        det_db = defaultdict(list)

    # Read sequence list
    sequences = []
    with open(args.sequences_file) as f:
        for line in f:
            seq = line.strip()
            if seq:
                sequences.append(seq)
    print(f"Processing {len(sequences)} sequences")

    stats = {'total_detections': 0, 'frames_with_detections': 0, 'filtered_low_conf': 0}

    for seq_path in sequences:
        # Extract sequence name from path like "volleyball/train_with_ball/v_1LwtoLPw2TU_c006"
        seq_name = os.path.basename(seq_path)

        # Find corresponding ball_detections file
        ball_det_file = os.path.join(args.ball_detections_dir, f"{seq_name}_ball_detections.json")
        if not os.path.exists(ball_det_file):
            print(f"  WARNING: No ball detections file for {seq_name}")
            continue

        print(f"Processing {seq_name}...")
        with open(ball_det_file) as f:
            ball_dets = json.load(f)

        seq_detections = 0
        seq_frames = 0

        for frame_str, detections in ball_dets.items():
            if not detections:
                continue

            frame_num = int(frame_str)
            frame_key = f"{seq_path}/img1/{frame_num:06d}"

            if frame_key not in det_db:
                det_db[frame_key] = []

            for det in detections:
                conf = det['confidence']
                if conf < args.min_confidence:
                    stats['filtered_low_conf'] += 1
                    continue

                bbox = det['bbox']  # [x, y, w, h]
                det_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{conf}\n"
                det_db[frame_key].append(det_str)
                seq_detections += 1

            if seq_detections > 0:
                seq_frames += 1

        stats['total_detections'] += seq_detections
        stats['frames_with_detections'] += seq_frames
        print(f"  {seq_detections} ball detections across {seq_frames} frames")

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(det_db, f)

    print(f"\nDet DB written to: {args.output}")
    print(f"  Total frames with detections: {stats['frames_with_detections']}")
    print(f"  Total ball detections: {stats['total_detections']}")
    print(f"  Filtered (low confidence): {stats['filtered_low_conf']}")
    print(f"  File size: {os.path.getsize(args.output) / 1024:.1f} KB")


if __name__ == '__main__':
    main()
