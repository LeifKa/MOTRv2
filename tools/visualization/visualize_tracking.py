#!/usr/bin/env python3
"""
Visualize MOTRv2 tracking results as a video.

This script creates a video with bounding boxes and track IDs overlaid.

Usage:
    python visualize_tracking.py \
        --images data/Dataset/mot/volleyball/test/test1/img1 \
        --tracking outputs/test_finetuned/test1.txt \
        --output tracking_viz.mp4
"""

import cv2
import os
import argparse
from collections import defaultdict
from pathlib import Path
import random


def parse_tracking_results(tracking_file):
    """
    Parse MOT format tracking results.

    Format: frame,id,x,y,w,h,conf,class,vis
    """
    tracks = defaultdict(list)

    print(f"📂 Loading tracking results from: {tracking_file}")

    with open(tracking_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue

            frame = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])

            tracks[frame].append({
                'id': track_id,
                'bbox': (x, y, w, h)
            })

    print(f"   ✓ Loaded {len(tracks)} frames")
    total_detections = sum(len(v) for v in tracks.values())
    print(f"   ✓ Total detections: {total_detections}\n")

    return tracks


def generate_colors(num_colors=100):
    """Generate distinct colors for track IDs."""
    random.seed(42)  # Fixed seed for consistency
    colors = {}

    for i in range(num_colors):
        # Generate bright, distinct colors
        hue = int(180 * i / num_colors)
        color_hsv = [[[hue, 255, 255]]]

        import numpy as np
        color_bgr = cv2.cvtColor(np.uint8(color_hsv), cv2.COLOR_HSV2BGR)[0][0]
        colors[i] = tuple(map(int, color_bgr))

    return colors


def visualize_tracking(image_dir, tracking_file, output_video,
                       fps=30, show_id=True, show_box=True,
                       line_thickness=2, font_scale=0.6):
    """
    Create video with tracking visualizations.

    Args:
        image_dir: Directory containing image frames
        tracking_file: Path to tracking results file
        output_video: Path to save output video
        fps: Frames per second for output video
        show_id: Show track ID labels
        show_box: Show bounding boxes
        line_thickness: Thickness of bounding box lines
        font_scale: Scale of text labels
    """
    print(f"\n{'='*80}")
    print("Creating Tracking Visualization")
    print(f"{'='*80}\n")

    print(f"Input images: {image_dir}")
    print(f"Tracking file: {tracking_file}")
    print(f"Output video: {output_video}\n")

    # Parse tracking results
    tracks = parse_tracking_results(tracking_file)

    # Get image files
    image_path = Path(image_dir)
    images = sorted([f for f in image_path.glob('*.jpg')] +
                   [f for f in image_path.glob('*.png')])

    if not images:
        print(f"❌ No images found in {image_dir}")
        return

    print(f"📷 Found {len(images)} images")

    # Get video properties from first image
    first_img = cv2.imread(str(images[0]))
    if first_img is None:
        print(f"❌ Error reading first image: {images[0]}")
        return

    height, width = first_img.shape[:2]
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}\n")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"❌ Error: Could not create video writer")
        return

    # Generate colors for track IDs
    colors = generate_colors()

    # Process each frame
    print(f"🎬 Generating video...")

    track_ids_seen = set()

    for idx, img_path in enumerate(images, 1):
        frame = cv2.imread(str(img_path))

        if frame is None:
            print(f"⚠️  Warning: Could not read {img_path}")
            continue

        # Draw detections for this frame
        if idx in tracks:
            frame_detections = tracks[idx]

            for detection in frame_detections:
                track_id = detection['id']
                x, y, w, h = detection['bbox']

                track_ids_seen.add(track_id)

                # Get color for this track ID
                color = colors.get(track_id % 100, (0, 255, 0))

                # Draw bounding box
                if show_box:
                    cv2.rectangle(frame,
                                (int(x), int(y)),
                                (int(x + w), int(y + h)),
                                color, line_thickness)

                # Draw ID label
                if show_id:
                    label = f"ID: {track_id}"
                    label_size, baseline = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        line_thickness
                    )

                    # Draw background rectangle for text
                    cv2.rectangle(frame,
                                (int(x), int(y) - label_size[1] - 10),
                                (int(x) + label_size[0], int(y)),
                                color, -1)

                    # Draw text
                    cv2.putText(frame, label,
                              (int(x), int(y) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              font_scale, (255, 255, 255), line_thickness)

            # Add detection count
            info_text = f"Frame: {idx} | Tracks: {len(frame_detections)}"
        else:
            info_text = f"Frame: {idx} | Tracks: 0"

        # Add frame info at top
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, info_text,
                  (10, 25),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.7, (255, 255, 255), 2)

        out.write(frame)

        # Progress indicator
        if idx % 30 == 0 or idx == len(images):
            print(f"   Progress: {idx}/{len(images)} frames", end='\r')

    out.release()

    print(f"\n\n{'='*80}")
    print("✅ Visualization Complete!")
    print(f"{'='*80}")
    print(f"  Output video: {output_video}")
    print(f"  Total frames: {len(images)}")
    print(f"  Unique track IDs: {len(track_ids_seen)}")
    print(f"  Video duration: {len(images)/fps:.1f}s")
    print(f"{'='*80}\n")


def compare_tracking_results(image_dir, tracking_file1, tracking_file2,
                            output_video, labels=None, fps=30):
    """
    Create side-by-side comparison of two tracking results.

    Args:
        image_dir: Directory containing image frames
        tracking_file1: First tracking results file
        tracking_file2: Second tracking results file
        output_video: Path to save comparison video
        labels: List of two labels for the tracking results
        fps: Frames per second
    """
    if labels is None:
        labels = ["Tracking 1", "Tracking 2"]

    print(f"\n{'='*80}")
    print("Creating Side-by-Side Comparison")
    print(f"{'='*80}\n")

    # Parse both tracking results
    tracks1 = parse_tracking_results(tracking_file1)
    tracks2 = parse_tracking_results(tracking_file2)

    # Get images
    image_path = Path(image_dir)
    images = sorted([f for f in image_path.glob('*.jpg')] +
                   [f for f in image_path.glob('*.png')])

    if not images:
        print(f"❌ No images found in {image_dir}")
        return

    # Get video properties
    first_img = cv2.imread(str(images[0]))
    height, width = first_img.shape[:2]

    # Create side-by-side video (double width)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width * 2, height))

    # Generate colors
    colors = generate_colors()

    print(f"🎬 Generating comparison video...")

    for idx, img_path in enumerate(images, 1):
        frame = cv2.imread(str(img_path))

        if frame is None:
            continue

        # Create two copies of the frame
        frame1 = frame.copy()
        frame2 = frame.copy()

        # Draw tracking 1
        if idx in tracks1:
            for detection in tracks1[idx]:
                track_id = detection['id']
                x, y, w, h = detection['bbox']
                color = colors.get(track_id % 100, (0, 255, 0))

                cv2.rectangle(frame1, (int(x), int(y)),
                            (int(x + w), int(y + h)), color, 2)
                cv2.putText(frame1, f"ID:{track_id}",
                          (int(x), int(y) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw tracking 2
        if idx in tracks2:
            for detection in tracks2[idx]:
                track_id = detection['id']
                x, y, w, h = detection['bbox']
                color = colors.get(track_id % 100, (0, 255, 0))

                cv2.rectangle(frame2, (int(x), int(y)),
                            (int(x + w), int(y + h)), color, 2)
                cv2.putText(frame2, f"ID:{track_id}",
                          (int(x), int(y) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add labels
        cv2.putText(frame1, labels[0], (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame2, labels[1], (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Combine side by side
        combined = cv2.hconcat([frame1, frame2])
        out.write(combined)

        if idx % 30 == 0:
            print(f"   Progress: {idx}/{len(images)} frames", end='\r')

    out.release()

    print(f"\n\n✅ Comparison video saved to: {output_video}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize MOTRv2 tracking results"
    )

    parser.add_argument(
        '--images', '-i',
        type=str,
        required=True,
        help='Directory containing image frames'
    )

    parser.add_argument(
        '--tracking', '-t',
        type=str,
        required=True,
        help='Path to tracking results file'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to save output video'
    )

    parser.add_argument(
        '--tracking2',
        type=str,
        help='Second tracking file for comparison (optional)'
    )

    parser.add_argument(
        '--labels',
        nargs=2,
        default=['Fine-tuned', 'Original'],
        help='Labels for comparison mode'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Output video FPS (default: 30)'
    )

    parser.add_argument(
        '--no_box',
        action='store_true',
        help='Hide bounding boxes'
    )

    parser.add_argument(
        '--no_id',
        action='store_true',
        help='Hide track ID labels'
    )

    args = parser.parse_args()

    # Comparison mode
    if args.tracking2:
        compare_tracking_results(
            args.images,
            args.tracking,
            args.tracking2,
            args.output,
            args.labels,
            args.fps
        )
    # Single tracking visualization
    else:
        visualize_tracking(
            args.images,
            args.tracking,
            args.output,
            fps=args.fps,
            show_id=not args.no_id,
            show_box=not args.no_box
        )
