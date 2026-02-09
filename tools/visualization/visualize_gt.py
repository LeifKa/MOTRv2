#!/usr/bin/env python3
"""
Visualize Ground Truth annotations from MOT format.

MOT Format: frame, id, x, y, w, h, conf, class
"""

import cv2
import os
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np


def parse_gt_file(gt_file):
    """Parse MOT format GT file."""
    detections = defaultdict(list)

    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                frame = int(float(parts[0]))
                track_id = int(float(parts[1]))
                x, y, w, h = map(float, parts[2:6])

                detections[frame].append({
                    'bbox': (x, y, w, h),
                    'track_id': track_id
                })

    return detections


def get_color_for_id(track_id):
    """Generate consistent color for track ID."""
    np.random.seed(track_id * 100)
    return tuple(map(int, np.random.randint(50, 255, 3)))


def visualize_gt(image_dir, gt_file, output_video, fps=30, line_thickness=2):
    """Create video with GT visualizations."""

    print(f"\n{'='*60}")
    print("Ground Truth Visualization")
    print(f"{'='*60}\n")

    print(f"Images: {image_dir}")
    print(f"GT file: {gt_file}")
    print(f"Output: {output_video}\n")

    # Parse GT
    detections = parse_gt_file(gt_file)
    print(f"Loaded {len(detections)} frames with annotations")
    total_boxes = sum(len(v) for v in detections.values())
    print(f"Total bounding boxes: {total_boxes}\n")

    # Get image files
    image_path = Path(image_dir)
    images = sorted(list(image_path.glob('*.jpg')) + list(image_path.glob('*.png')))

    if not images:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(images)} images")

    # Get video properties
    first_img = cv2.imread(str(images[0]))
    height, width = first_img.shape[:2]
    print(f"Resolution: {width}x{height}")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print(f"\nGenerating video...")

    for idx, img_path in enumerate(images, 1):
        frame = cv2.imread(str(img_path))

        if frame is None:
            continue

        # Draw GT boxes
        if idx in detections:
            for det in detections[idx]:
                x, y, w, h = det['bbox']
                track_id = det['track_id']
                color = get_color_for_id(track_id)

                # Draw box
                cv2.rectangle(frame,
                            (int(x), int(y)),
                            (int(x + w), int(y + h)),
                            color, line_thickness)

                # Draw ID label
                label = f"ID:{track_id}"
                cv2.rectangle(frame,
                            (int(x), int(y) - 20),
                            (int(x) + 60, int(y)),
                            color, -1)
                cv2.putText(frame, label,
                          (int(x) + 2, int(y) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, (255, 255, 255), 1)

        # Frame info
        num_dets = len(detections.get(idx, []))
        cv2.rectangle(frame, (0, 0), (300, 35), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame: {idx} | Objects: {num_dets}",
                  (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)

        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(images)}", end='\r')

    out.release()
    print(f"\n\nDone! Video saved to: {output_video}")
    print(f"Duration: {len(images)/fps:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MOT GT annotations")
    parser.add_argument('--images', '-i', required=True, help='Image directory')
    parser.add_argument('--gt', '-g', required=True, help='GT file (MOT format)')
    parser.add_argument('--output', '-o', required=True, help='Output video path')
    parser.add_argument('--fps', type=int, default=30, help='FPS (default: 30)')

    args = parser.parse_args()
    visualize_gt(args.images, args.gt, args.output, args.fps)
