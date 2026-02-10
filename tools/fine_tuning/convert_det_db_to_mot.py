#!/usr/bin/env python3
"""
Convert MOTRv2 Detection Database JSON to MOT format TXT.

det_db JSON format (from D-FINE/YOLOX):
  {
    "volleyball/test/test1/img1/000001": ["769.77,603.09,84.14,229.93,0.9938", ...],
    "volleyball/test/test1/img1/000002": [...],
    ...
  }

Output MOT format (same as inference output):
  frame, id, x, y, w, h, conf, -1, -1, -1

Usage:
    python convert_det_db_to_mot.py \
        --input data/Dataset/mot/det_db_sequenz_beach_yolox_conf08.json \
        --output analysis/yolox_detections_sequenz_beach.txt
"""

import argparse
import json
import re
from pathlib import Path


def extract_frame_number(key: str) -> int:
    """Extract frame number from det_db key like 'volleyball/test/test1/img1/000001'."""
    filename = key.rstrip('/').split('/')[-1]
    return int(filename)


def main():
    parser = argparse.ArgumentParser(description='Convert det_db JSON to MOT format TXT')
    parser.add_argument('--input', required=True, help='Path to det_db JSON file')
    parser.add_argument('--output', required=True, help='Output TXT file path')
    parser.add_argument('--score_threshold', type=float, default=0.0,
                        help='Minimum confidence score to include (default: 0.0 = all)')
    args = parser.parse_args()

    print(f"Loading det_db from {args.input}...")
    with open(args.input) as f:
        det_db = json.load(f)

    print(f"  Keys: {len(det_db)}")

    lines = []
    total_dets = 0
    filtered_dets = 0

    for key in sorted(det_db.keys(), key=extract_frame_number):
        frame_num = extract_frame_number(key)

        for det_id, det_str in enumerate(det_db[key]):
            parts = det_str.split(',')
            x = float(parts[0])
            y = float(parts[1])
            w = float(parts[2])
            h = float(parts[3])
            score = float(parts[4])

            total_dets += 1

            if score < args.score_threshold:
                filtered_dets += 1
                continue

            # MOT format: frame, id, x, y, w, h, conf, -1, -1, -1
            lines.append(f"{frame_num},{det_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"  Total detections: {total_dets}")
    if args.score_threshold > 0:
        print(f"  Filtered (score < {args.score_threshold}): {filtered_dets}")
    print(f"  Written: {len(lines)} detections over {len(det_db)} frames")
    print(f"  Output: {args.output}")


if __name__ == '__main__':
    main()
