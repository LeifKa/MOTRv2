#!/usr/bin/env python3
"""
Convert MOTRv2 tracking output to MOT ground truth format.

This script converts tracking results from MOTRv2 (which uses YOLOX)
into ground truth format that can be used for training with D-FINE.

Usage:
    python convert_to_gt.py --input tracking_result.txt --output gt/gt.txt
"""

import os
import argparse
from pathlib import Path


def motrv2_to_gt(tracking_file, output_gt_file, min_track_length=10, verbose=True):
    """
    Convert MOTRv2 tracking output to MOT ground truth format.

    MOTRv2 output: frame,id,x1,y1,w,h,1,-1,-1,-1
    Ground truth:  frame,id,x,y,w,h,1,1,1

    Args:
        tracking_file: Path to MOTRv2 tracking output
        output_gt_file: Path to save ground truth file
        min_track_length: Minimum number of frames for a track to be included
        verbose: Print statistics
    """
    if verbose:
        print(f"\n{'='*80}")
        print("Converting Tracking Results to Ground Truth Format")
        print(f"{'='*80}\n")
        print(f"Input:  {tracking_file}")
        print(f"Output: {output_gt_file}\n")

    # Read tracking results
    with open(tracking_file, 'r') as f:
        lines = f.readlines()

    # Parse and filter tracks
    from collections import defaultdict
    tracks = defaultdict(list)

    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 7:
            frame = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])

            tracks[track_id].append({
                'frame': frame,
                'x': x,
                'y': y,
                'w': w,
                'h': h
            })

    # Filter short tracks
    if verbose:
        print(f"📊 Track Statistics:")
        print(f"  Total tracks: {len(tracks)}")

    filtered_tracks = {}
    for track_id, detections in tracks.items():
        if len(detections) >= min_track_length:
            filtered_tracks[track_id] = detections

    if verbose:
        print(f"  Tracks >= {min_track_length} frames: {len(filtered_tracks)}")
        removed = len(tracks) - len(filtered_tracks)
        if removed > 0:
            print(f"  Removed {removed} short tracks\n")

    # Convert to ground truth format
    gt_lines = []
    total_annotations = 0

    for track_id, detections in filtered_tracks.items():
        for det in detections:
            # GT format: frame,id,x,y,w,h,conf=1,class=1,vis=1
            gt_line = (f"{det['frame']},{track_id},"
                      f"{det['x']:.2f},{det['y']:.2f},"
                      f"{det['w']:.2f},{det['h']:.2f},"
                      f"1,1,1\n")
            gt_lines.append(gt_line)
            total_annotations += 1

    # Sort by frame number
    gt_lines.sort(key=lambda x: int(x.split(',')[0]))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_gt_file), exist_ok=True)

    # Write ground truth file
    with open(output_gt_file, 'w') as f:
        f.writelines(gt_lines)

    if verbose:
        print(f"✅ Conversion Complete!")
        print(f"  Total annotations: {total_annotations:,}")
        print(f"  Output file: {output_gt_file}")
        print(f"\n{'='*80}\n")

    return len(filtered_tracks), total_annotations


def batch_convert(input_dir, output_base_dir, min_track_length=10):
    """
    Convert multiple tracking result files to ground truth format.

    Args:
        input_dir: Directory containing tracking result files
        output_base_dir: Base directory for ground truth outputs
        min_track_length: Minimum track length to include
    """
    print(f"\n{'='*80}")
    print("Batch Conversion: Tracking Results → Ground Truth")
    print(f"{'='*80}\n")

    tracking_files = list(Path(input_dir).glob("*.txt"))

    if not tracking_files:
        print(f"❌ No .txt files found in {input_dir}")
        return

    print(f"Found {len(tracking_files)} tracking result files\n")

    total_tracks = 0
    total_annotations = 0

    for tracking_file in tracking_files:
        # Generate output path
        # Example: game1.txt -> volleyball/train/game1/gt/gt.txt
        sequence_name = tracking_file.stem
        output_gt_file = os.path.join(
            output_base_dir,
            f"{sequence_name}/gt/gt.txt"
        )

        print(f"Processing: {tracking_file.name}")

        try:
            tracks, annotations = motrv2_to_gt(
                str(tracking_file),
                output_gt_file,
                min_track_length,
                verbose=False
            )
            total_tracks += tracks
            total_annotations += annotations
            print(f"  ✓ {tracks} tracks, {annotations} annotations\n")

        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            continue

    print(f"{'='*80}")
    print(f"Batch Conversion Complete!")
    print(f"  Total tracks: {total_tracks}")
    print(f"  Total annotations: {total_annotations:,}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MOTRv2 tracking results to ground truth format"
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to tracking result file or directory'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to output ground truth file or directory'
    )

    parser.add_argument(
        '--min_track_length',
        type=int,
        default=10,
        help='Minimum number of frames for a track to be included (default: 10)'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch mode: convert all .txt files in input directory'
    )

    args = parser.parse_args()

    # Interactive mode if no arguments
    if not args.input:
        print("\n" + "="*80)
        print("MOTRv2 Tracking → Ground Truth Converter")
        print("="*80 + "\n")

        print("Example usage:")
        print("  Single file:")
        print("    python convert_to_gt.py -i tracking.txt -o gt/gt.txt\n")
        print("  Batch mode:")
        print("    python convert_to_gt.py -i outputs/pseudo_labels/ -o data/Dataset/mot/volleyball/train/ --batch\n")

        # Provide example conversion
        print("Running example conversion...\n")

        example_input = "outputs/pseudo_labels/game1.txt"
        example_output = "data/Dataset/mot/volleyball/train/game1/gt/gt.txt"

        if os.path.exists(example_input):
            motrv2_to_gt(example_input, example_output, args.min_track_length)
        else:
            print(f"Example file not found: {example_input}")
            print("Please specify --input and --output paths.\n")

        exit(0)

    # Batch mode
    if args.batch:
        if not args.output:
            print("❌ Error: --output required in batch mode")
            exit(1)

        batch_convert(args.input, args.output, args.min_track_length)

    # Single file mode
    else:
        if not args.output:
            print("❌ Error: --output required")
            exit(1)

        motrv2_to_gt(args.input, args.output, args.min_track_length)
