#!/usr/bin/env python3
"""
Convert LabelStudio JSON export to MOT ground truth format for MOTRv2 fine-tuning.

Takes a LabelStudio JSON export containing player and ball bounding box annotations
on SportsMOT volleyball frames, and produces:
  - MOT gt.txt files with both player (class_id=1) and ball (class_id=2) annotations
  - Symlinks to original SportsMOT image directories
  - seqinfo.ini files for each sequence
  - A data_txt_path file listing all sequences for training

Usage:
    python convert_labelstudio_to_mot.py \
        --input /path/to/labelstudio_export.json \
        --sportsmot_root /path/to/SportsMOT_Volleyball \
        --mot_path /path/to/MOTRv2/data/Dataset/mot \
        --output_name train_with_ball
"""

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path


def parse_image_path(image_path: str):
    """Extract sequence name and frame number from LabelStudio image path.

    LabelStudio paths look like:
        /data/local-files/?d=train/v_1LwtoLPw2TU_c006/img1/000001.jpg
    """
    # Extract the part after ?d=
    match = re.search(r'\?d=(.+)$', image_path)
    if not match:
        raise ValueError(f"Cannot parse image path: {image_path}")

    rel_path = match.group(1)  # e.g. train/v_1LwtoLPw2TU_c006/img1/000001.jpg
    parts = rel_path.split('/')

    # Find sequence name (starts with v_)
    seq_name = None
    split_name = None
    for i, p in enumerate(parts):
        if p.startswith('v_'):
            seq_name = p
            split_name = parts[i - 1] if i > 0 else 'train'
            break

    if seq_name is None:
        raise ValueError(f"Cannot find sequence name in path: {image_path}")

    # Extract frame number from filename
    filename = parts[-1]  # e.g. 000001.jpg
    frame_num = int(filename.replace('.jpg', ''))

    return split_name, seq_name, frame_num


def convert_bbox_to_pixels(result: dict):
    """Convert LabelStudio percentage-based bbox to pixel coordinates.

    LabelStudio format: x, y, width, height in percentage (0-100)
    original_width/height are on the result level, not inside value.
    MOT format: x, y, w, h in pixels (top-left corner)
    """
    value = result['value']
    orig_w = result.get('original_width', value.get('original_width', 1280))
    orig_h = result.get('original_height', value.get('original_height', 720))

    x_px = value['x'] / 100.0 * orig_w
    y_px = value['y'] / 100.0 * orig_h
    w_px = value['width'] / 100.0 * orig_w
    h_px = value['height'] / 100.0 * orig_h

    return x_px, y_px, w_px, h_px


def extract_player_track_id(annotation_id: str) -> int:
    """Extract numeric track ID from LabelStudio annotation ID.

    Player annotations have IDs like 'track_0', 'track_1', etc.
    Returns the numeric part.
    """
    match = re.match(r'track_(\d+)', annotation_id)
    if match:
        return int(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(description='Convert LabelStudio JSON to MOT format')
    parser.add_argument('--input', required=True,
                        help='Path to LabelStudio JSON export file')
    parser.add_argument('--sportsmot_root', required=True,
                        help='Path to SportsMOT_Volleyball dataset root')
    parser.add_argument('--mot_path', required=True,
                        help='Path to MOTRv2 mot directory (data/Dataset/mot)')
    parser.add_argument('--output_name', default='train_with_ball',
                        help='Name for output directory under volleyball/ (default: train_with_ball)')
    parser.add_argument('--ball_class_id', type=int, default=2,
                        help='Class ID for ball in gt.txt (default: 2, players are 1)')
    parser.add_argument('--player_class_id', type=int, default=1,
                        help='Class ID for players in gt.txt (default: 1)')
    args = parser.parse_args()

    # Load LabelStudio export
    print(f"Loading LabelStudio export from: {args.input}")
    with open(args.input) as f:
        data = json.load(f)
    print(f"  Found {len(data)} annotated frames")

    # Parse all annotations, grouped by sequence
    # Structure: sequences[seq_name][frame_num] = {'players': [...], 'balls': [...]}
    sequences = defaultdict(lambda: defaultdict(lambda: {'players': [], 'balls': []}))
    seq_splits = {}  # seq_name -> split_name (e.g. 'train')

    stats = {'total_frames': 0, 'ball_frames': 0, 'player_annotations': 0, 'ball_annotations': 0}

    for task in data:
        image_path = task['data']['image']
        split_name, seq_name, frame_num = parse_image_path(image_path)
        seq_splits[seq_name] = split_name
        stats['total_frames'] += 1

        # Process annotations (take last/most recent annotation set)
        annotations = task.get('annotations', [])
        if not annotations:
            continue

        # Use the most recent annotation
        ann = annotations[-1]
        has_ball = False

        for result in ann.get('result', []):
            if result.get('type') != 'rectanglelabels':
                continue

            value = result['value']
            labels = value.get('rectanglelabels', [])
            ann_id = result.get('id', '')
            x, y, w, h = convert_bbox_to_pixels(result)

            if 'ball' in labels:
                sequences[seq_name][frame_num]['balls'].append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'id': ann_id,
                })
                stats['ball_annotations'] += 1
                has_ball = True

            elif 'player' in labels:
                track_id = extract_player_track_id(ann_id)
                if track_id is None:
                    # Fallback: assign incremental ID if track_N pattern doesn't match
                    track_id = len(sequences[seq_name][frame_num]['players'])

                sequences[seq_name][frame_num]['players'].append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'track_id': track_id,
                })
                stats['player_annotations'] += 1

        if has_ball:
            stats['ball_frames'] += 1

    print(f"\nParsed annotations:")
    print(f"  Sequences: {len(sequences)}")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Frames with ball: {stats['ball_frames']} ({100*stats['ball_frames']/max(1,stats['total_frames']):.1f}%)")
    print(f"  Player annotations: {stats['player_annotations']}")
    print(f"  Ball annotations: {stats['ball_annotations']}")

    # Create output directory structure
    output_base = os.path.join(args.mot_path, 'volleyball', args.output_name)
    os.makedirs(output_base, exist_ok=True)

    data_txt_lines = []

    for seq_name in sorted(sequences.keys()):
        split_name = seq_splits[seq_name]
        frames = sequences[seq_name]
        print(f"\nProcessing {seq_name} ({len(frames)} frames)...")

        # Determine ball track ID (one higher than max player ID in this sequence)
        max_player_id = 0
        for frame_data in frames.values():
            for player in frame_data['players']:
                max_player_id = max(max_player_id, player['track_id'])
        ball_track_id = max_player_id + 1

        # Create sequence directory
        seq_dir = os.path.join(output_base, seq_name)
        gt_dir = os.path.join(seq_dir, 'gt')
        os.makedirs(gt_dir, exist_ok=True)

        # Symlink images from original SportsMOT
        img_link = os.path.join(seq_dir, 'img1')
        img_source = os.path.join(args.sportsmot_root, split_name, seq_name, 'img1')

        if not os.path.exists(img_source):
            print(f"  WARNING: Source images not found at {img_source}")
            continue

        if os.path.islink(img_link):
            os.remove(img_link)
        elif os.path.exists(img_link):
            shutil.rmtree(img_link)

        os.symlink(os.path.abspath(img_source), img_link)
        print(f"  Linked images: {img_link} -> {img_source}")

        # Copy seqinfo.ini from original
        seqinfo_source = os.path.join(args.sportsmot_root, split_name, seq_name, 'seqinfo.ini')
        seqinfo_dest = os.path.join(seq_dir, 'seqinfo.ini')
        if os.path.exists(seqinfo_source):
            shutil.copy2(seqinfo_source, seqinfo_dest)
        else:
            # Create seqinfo.ini from what we know
            num_frames = max(frames.keys())
            with open(seqinfo_dest, 'w') as f:
                f.write(f"[Sequence]\n")
                f.write(f"name={seq_name}\n")
                f.write(f"imDir=img1\n")
                f.write(f"frameRate=25\n")
                f.write(f"seqLength={num_frames}\n")
                f.write(f"imWidth=1280\n")
                f.write(f"imHeight=720\n")
                f.write(f"imExt=.jpg\n")

        # Write gt.txt
        gt_path = os.path.join(gt_dir, 'gt.txt')
        gt_lines = []

        for frame_num in sorted(frames.keys()):
            frame_data = frames[frame_num]

            # Write player annotations
            for player in frame_data['players']:
                gt_lines.append(
                    f"{frame_num},{player['track_id']},"
                    f"{player['x']:.2f},{player['y']:.2f},"
                    f"{player['w']:.2f},{player['h']:.2f},"
                    f"1,{args.player_class_id},1\n"
                )

            # Write ball annotations (use consistent track ID)
            for ball in frame_data['balls']:
                gt_lines.append(
                    f"{frame_num},{ball_track_id},"
                    f"{ball['x']:.2f},{ball['y']:.2f},"
                    f"{ball['w']:.2f},{ball['h']:.2f},"
                    f"1,{args.ball_class_id},1\n"
                )

        with open(gt_path, 'w') as f:
            f.writelines(gt_lines)

        n_player_lines = sum(1 for l in gt_lines if l.rstrip().endswith(f',{args.player_class_id},1'))
        n_ball_lines = sum(1 for l in gt_lines if l.rstrip().endswith(f',{args.ball_class_id},1'))
        print(f"  Wrote gt.txt: {len(gt_lines)} lines ({n_player_lines} player, {n_ball_lines} ball)")
        print(f"  Ball track_id: {ball_track_id}")

        # Add to data_txt_path
        rel_seq_path = f"volleyball/{args.output_name}/{seq_name}"
        data_txt_lines.append(rel_seq_path)

    # Write data_txt_path file
    data_txt_path = os.path.join(
        args.mot_path, '..', '..', '..', 'datasets', 'data_path',
        f'volleyball_{args.output_name}_train.txt'
    )
    data_txt_path = os.path.normpath(data_txt_path)
    os.makedirs(os.path.dirname(data_txt_path), exist_ok=True)

    with open(data_txt_path, 'w') as f:
        for line in data_txt_lines:
            f.write(line + '\n')

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"{'='*60}")
    print(f"  Output directory: {output_base}")
    print(f"  Sequences: {len(sequences)}")
    print(f"  Data txt path: {data_txt_path}")
    print(f"\ngt.txt format:")
    print(f"  frame_id, track_id, x, y, w, h, 1, class_id, 1")
    print(f"  Players: class_id={args.player_class_id}")
    print(f"  Ball:    class_id={args.ball_class_id}")
    print(f"\nTo use for fine-tuning, update your training config:")
    print(f"  --data_txt_path_train {data_txt_path}")
    print(f"  --mot_path {args.mot_path}")

    # Show sample gt.txt output
    sample_seq = sorted(sequences.keys())[0]
    sample_gt = os.path.join(output_base, sample_seq, 'gt', 'gt.txt')
    print(f"\nSample output ({sample_gt}):")
    with open(sample_gt) as f:
        for i, line in enumerate(f):
            if i >= 10:
                print("  ...")
                break
            print(f"  {line.rstrip()}")


if __name__ == '__main__':
    main()
