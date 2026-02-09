#!/usr/bin/env python3
"""
Demo-Skript das mit synthetischen Daten zeigt, wie die Evaluation funktioniert
"""

import json
import numpy as np
from pathlib import Path
from mot_evaluation import MOTEvaluator


def create_synthetic_gt(num_frames=50, num_objects=6, output_file="analysis/demo_data/demo_gt.json"):
    """Erstellt synthetische Ground Truth Daten im Label Studio Format"""

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    data = []

    for frame_id in range(1, num_frames + 1):
        entry = {
            "id": frame_id,
            "data": {
                "image": f"/demo/frames/frame_{frame_id:06d}.jpg"
            },
            "annotations": [
                {
                    "id": 1,
                    "completed_by": 1,
                    "result": []
                }
            ]
        }

        # Generate bounding boxes for each object
        for obj_id in range(num_objects):
            # Simulate object movement across frames
            base_x = 10 + obj_id * 15
            base_y = 20 + (obj_id % 3) * 20

            # Add some random movement
            x = base_x + (frame_id % 10) * 0.5
            y = base_y + np.sin(frame_id * 0.2 + obj_id) * 2
            w = 8 + np.random.uniform(-0.5, 0.5)
            h = 12 + np.random.uniform(-0.5, 0.5)

            # Ensure within bounds
            x = max(0, min(90, x))
            y = max(0, min(80, y))

            bbox = {
                "id": f"track_{obj_id}",
                "type": "rectanglelabels",
                "value": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "rotation": 0,
                    "rectanglelabels": ["player"]
                },
                "origin": "manual",
                "to_name": "image",
                "from_name": "label",
                "image_rotation": 0,
                "original_width": 1280,
                "original_height": 720
            }

            entry["annotations"][0]["result"].append(bbox)

        data.append(entry)

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ Created synthetic GT with {num_frames} frames, {num_objects} objects")
    print(f"   Saved to: {output_file}")


def create_synthetic_predictions(num_frames=50, num_objects=6,
                                 noise_level=0.1, missing_rate=0.05,
                                 fp_rate=0.1, id_switch_rate=0.02,
                                 output_file="analysis/demo_data/demo_predictions.txt"):
    """
    Erstellt synthetische Predictions im MOT Format
    Simuliert realistische Tracking-Fehler
    """

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    predictions = []
    current_ids = list(range(num_objects))
    next_id = num_objects

    for frame_id in range(1, num_frames + 1):
        for obj_id in range(num_objects):
            # Simulate missing detections
            if np.random.random() < missing_rate:
                continue

            # Simulate ID switches
            if np.random.random() < id_switch_rate:
                current_ids[obj_id] = next_id
                next_id += 1

            track_id = current_ids[obj_id]

            # Calculate position (similar to GT but with noise)
            base_x = 10 + obj_id * 15
            base_y = 20 + (obj_id % 3) * 20

            x = base_x + (frame_id % 10) * 0.5
            y = base_y + np.sin(frame_id * 0.2 + obj_id) * 2
            w = 8
            h = 12

            # Convert from percentage to pixels (GT is in %, predictions in pixels)
            x_px = (x / 100.0) * 1280 + np.random.normal(0, noise_level * 1280)
            y_px = (y / 100.0) * 720 + np.random.normal(0, noise_level * 720)
            w_px = (w / 100.0) * 1280 + np.random.normal(0, noise_level * 50)
            h_px = (h / 100.0) * 720 + np.random.normal(0, noise_level * 50)

            # Ensure positive dimensions
            w_px = max(10, w_px)
            h_px = max(10, h_px)

            conf = np.random.uniform(0.7, 0.99)

            # MOT format: <frame>, <id>, <x>, <y>, <w>, <h>, <conf>, <x>, <y>, <z>
            predictions.append(f"{frame_id},{track_id},{x_px:.2f},{y_px:.2f},{w_px:.2f},{h_px:.2f},{conf:.4f},-1,-1,-1")

        # Add false positives
        if np.random.random() < fp_rate:
            fp_x = np.random.uniform(0, 1280)
            fp_y = np.random.uniform(0, 720)
            fp_w = np.random.uniform(50, 150)
            fp_h = np.random.uniform(80, 180)
            fp_conf = np.random.uniform(0.5, 0.8)

            predictions.append(f"{frame_id},{next_id},{fp_x:.2f},{fp_y:.2f},{fp_w:.2f},{fp_h:.2f},{fp_conf:.4f},-1,-1,-1")
            next_id += 1

    with open(output_file, 'w') as f:
        f.write('\n'.join(predictions))

    print(f"✅ Created synthetic predictions with:")
    print(f"   - Noise level: {noise_level}")
    print(f"   - Missing rate: {missing_rate}")
    print(f"   - FP rate: {fp_rate}")
    print(f"   - ID switch rate: {id_switch_rate}")
    print(f"   Saved to: {output_file}")


def run_demo_evaluation():
    """Führt Demo-Evaluation mit synthetischen Daten aus"""

    print("\n" + "="*80)
    print("DEMO: MOTRv2 EVALUATION WITH SYNTHETIC DATA")
    print("="*80 + "\n")

    # Erstelle synthetische Daten
    print("📝 Creating synthetic data...\n")
    create_synthetic_gt(num_frames=100, num_objects=8,
                       output_file="analysis/demo_data/demo_gt.json")
    create_synthetic_predictions(num_frames=100, num_objects=8,
                                 noise_level=0.05, missing_rate=0.08,
                                 fp_rate=0.12, id_switch_rate=0.03,
                                 output_file="analysis/demo_data/demo_predictions.txt")

    print("\n" + "="*80)
    print("Running Evaluation...")
    print("="*80 + "\n")

    # Führe Evaluation aus
    evaluator = MOTEvaluator(iou_threshold=0.5)
    evaluator.load_ground_truth_json("analysis/demo_data/demo_gt.json")
    evaluator.load_predictions_mot_format("analysis/demo_data/demo_predictions.txt")

    evaluator.compute_metrics()
    evaluator.print_summary()

    # Speichere Ergebnisse
    print("\n📊 Saving results...")
    evaluator.save_metrics_json("analysis/demo_data/demo_metrics.json")
    evaluator.plot_results("analysis/demo_data/plots")

    print("\n" + "="*80)
    print("✅ DEMO COMPLETE!")
    print("="*80)
    print("\nCheck the following directories:")
    print("  - analysis/demo_data/plots/           - All visualizations")
    print("  - analysis/demo_data/demo_metrics.json - Metrics JSON")
    print("\nThis shows exactly how the evaluation will work with your real data!")


if __name__ == "__main__":
    run_demo_evaluation()
