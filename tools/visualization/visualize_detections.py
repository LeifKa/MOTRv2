#!/usr/bin/env python3
"""
Visualize D-FINE detections as a video.

This script creates a video with bounding boxes and confidence scores overlaid.

Usage:
    python visualize_detections.py \
        --images data/Dataset/mot/volleyball/test/test1/img1 \
        --det_db det_db_beach_volleyball.json \
        --output detections_viz.mp4
"""

import cv2
import os
import json
import argparse
from collections import defaultdict
from pathlib import Path


def parse_detection_db(det_db_file, video_path):
    """
    Parse detection database JSON file.

    Args:
        det_db_file: Path to detection database JSON
        video_path: Video path prefix (e.g., "volleyball/test/test1")

    Returns:
        Dictionary mapping frame number to list of detections
    """
    detections = defaultdict(list)

    print(f"📂 Loading detections from: {det_db_file}")

    with open(det_db_file, 'r') as f:
        det_db = json.load(f)

    # Parse detections for the specified video
    for frame_key, det_list in det_db.items():
        # frame_key format: "volleyball/test/test1/img1/000001"
        if video_path in frame_key:
            # Extract frame number from path
            frame_num_str = frame_key.split('/')[-1]
            frame_num = int(frame_num_str)

            # Parse each detection string: "x,y,w,h,score\n"
            for det_str in det_list:
                parts = det_str.strip().split(',')
                if len(parts) >= 5:
                    x, y, w, h, score = map(float, parts[:5])
                    detections[frame_num].append({
                        'bbox': (x, y, w, h),
                        'score': score
                    })

    print(f"   ✓ Loaded {len(detections)} frames")
    total_detections = sum(len(v) for v in detections.values())
    print(f"   ✓ Total detections: {total_detections}\n")

    return detections


def visualize_detections(image_dir, det_db_file, output_video, video_path,
                        fps=30, score_threshold=0.0,
                        line_thickness=2, font_scale=0.5):
    """
    Create video with detection visualizations.

    Args:
        image_dir: Directory containing image frames
        det_db_file: Path to detection database JSON
        output_video: Path to save output video
        video_path: Video path in detection database (e.g., "volleyball/test/test1")
        fps: Frames per second for output video
        score_threshold: Minimum confidence score to display
        line_thickness: Thickness of bounding box lines
        font_scale: Scale of text labels
    """
    print(f"\n{'='*80}")
    print("Creating Detection Visualization")
    print(f"{'='*80}\n")

    print(f"Input images: {image_dir}")
    print(f"Detection DB: {det_db_file}")
    print(f"Video path: {video_path}")
    print(f"Output video: {output_video}")
    print(f"Score threshold: {score_threshold}\n")

    # Parse detections
    detections = parse_detection_db(det_db_file, video_path)

    if not detections:
        print(f"❌ ERROR: No detections found for video path '{video_path}'")
        print(f"   Make sure the video_path matches the keys in the detection database.")
        return

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

    # Process each frame
    print(f"🎬 Generating video...")
    print(f"   First 5 frames have detections: ", end="")
    for i in range(1, min(6, len(images)+1)):
        if i in detections:
            print(f"✓", end=" ")
        else:
            print(f"✗", end=" ")
    print()

    total_dets_shown = 0
    total_dets_filtered = 0

    for idx, img_path in enumerate(images, 1):
        frame = cv2.imread(str(img_path))

        if frame is None:
            print(f"⚠️  Warning: Could not read {img_path}")
            continue

        # Draw detections for this frame
        num_dets = 0
        num_filtered = 0

        if idx in detections:
            frame_detections = detections[idx]

            for detection in frame_detections:
                x, y, w, h = detection['bbox']
                score = detection['score']

                # Filter by score threshold
                if score < score_threshold:
                    num_filtered += 1
                    continue

                num_dets += 1
                total_dets_shown += 1

                # Color based on confidence (green = high, yellow = medium, red = low)
                if score >= 0.7:
                    color = (0, 255, 0)  # Green
                elif score >= 0.5:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 165, 255)  # Orange

                # Draw bounding box
                cv2.rectangle(frame,
                            (int(x), int(y)),
                            (int(x + w), int(y + h)),
                            color, line_thickness)

                # Draw confidence score
                label = f"{score:.2f}"
                label_size, baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    1
                )

                # Draw background rectangle for text
                cv2.rectangle(frame,
                            (int(x), int(y) - label_size[1] - 8),
                            (int(x) + label_size[0], int(y)),
                            color, -1)

                # Draw text
                cv2.putText(frame, label,
                          (int(x), int(y) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          font_scale, (0, 0, 0), 1)

            total_dets_filtered += num_filtered
            info_text = f"Frame: {idx} | Detections: {num_dets}"
            if num_filtered > 0:
                info_text += f" (filtered: {num_filtered})"
        else:
            info_text = f"Frame: {idx} | Detections: 0"

        # Add frame info at top
        cv2.rectangle(frame, (0, 0), (500, 40), (0, 0, 0), -1)
        cv2.putText(frame, info_text,
                  (10, 25),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.7, (255, 255, 255), 2)

        # Add color legend
        legend_y = 50
        cv2.putText(frame, "Score: ", (10, legend_y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (80, legend_y - 12), (110, legend_y), (0, 255, 0), -1)
        cv2.putText(frame, ">0.7", (115, legend_y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (170, legend_y - 12), (200, legend_y), (0, 255, 255), -1)
        cv2.putText(frame, ">0.5", (205, legend_y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (260, legend_y - 12), (290, legend_y), (0, 165, 255), -1)
        cv2.putText(frame, "<0.5", (295, legend_y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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
    print(f"  Detections shown: {total_dets_shown}")
    if total_dets_filtered > 0:
        print(f"  Detections filtered: {total_dets_filtered}")
    print(f"  Video duration: {len(images)/fps:.1f}s")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize D-FINE detections"
    )

    parser.add_argument(
        '--images', '-i',
        type=str,
        required=True,
        help='Directory containing image frames'
    )

    parser.add_argument(
        '--det_db', '-d',
        type=str,
        required=True,
        help='Path to detection database JSON file'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to save output video'
    )

    parser.add_argument(
        '--video_path', '-v',
        type=str,
        default='volleyball/test/test1',
        help='Video path in detection database (default: volleyball/test/test1)'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Output video FPS (default: 30)'
    )

    parser.add_argument(
        '--score_threshold',
        type=float,
        default=0.0,
        help='Minimum confidence score to display (default: 0.0)'
    )

    args = parser.parse_args()

    visualize_detections(
        args.images,
        args.det_db,
        args.output,
        args.video_path,
        fps=args.fps,
        score_threshold=args.score_threshold
    )
