#!/usr/bin/env python3
"""
MOTRv2 Inference Evaluation Script
Vergleicht Inferenz-Ergebnisse mit Ground Truth und berechnet quantitative Metriken
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import seaborn as sns

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


class MOTEvaluator:
    """Evaluiert MOTRv2 Tracking Ergebnisse gegen Ground Truth"""

    def __init__(self, iou_threshold: float = 0.5):
        """
        Args:
            iou_threshold: IoU Schwellenwert für Detection Matching (default: 0.5)
        """
        self.iou_threshold = iou_threshold
        self.gt_data = {}  # {frame_id: [(track_id, bbox), ...]}
        self.pred_data = {}  # {frame_id: [(track_id, bbox, conf), ...]}
        self.metrics = {}
        self.track_id_mapping = {}  # Maps string IDs to integer IDs

    def load_ground_truth_json(self, json_path: str):
        """
        Lädt Ground Truth aus Label Studio JSON Format

        Args:
            json_path: Pfad zur JSON Datei
        """
        print(f"Loading Ground Truth from {json_path}...")
        with open(json_path, 'r') as f:
            data = json.load(f)

        for entry in data:
            # Extract frame number from filename
            image_path = entry['data'].get('image', '')
            # Assume format: .../frame_XXXX.jpg or similar
            frame_id = self._extract_frame_id(image_path)

            if frame_id is None:
                continue

            frame_annotations = []

            if 'annotations' in entry and len(entry['annotations']) > 0:
                for annotation in entry['annotations']:
                    if 'result' in annotation:
                        for bbox_data in annotation['result']:
                            if bbox_data['type'] == 'rectanglelabels':
                                # Extract track ID from 'id' field
                                track_id_str = bbox_data.get('id', 'track_0')

                                # Try to extract numeric ID, otherwise create mapping
                                try:
                                    if track_id_str.startswith('track_'):
                                        track_id = int(track_id_str.replace('track_', ''))
                                    else:
                                        # Use hash-based mapping for non-numeric IDs
                                        if track_id_str not in self.track_id_mapping:
                                            self.track_id_mapping[track_id_str] = len(self.track_id_mapping)
                                        track_id = self.track_id_mapping[track_id_str]
                                except (ValueError, AttributeError):
                                    # Fallback: use hash
                                    if track_id_str not in self.track_id_mapping:
                                        self.track_id_mapping[track_id_str] = len(self.track_id_mapping)
                                    track_id = self.track_id_mapping[track_id_str]

                                # Get original dimensions (they're outside of 'value')
                                orig_w = bbox_data.get('original_width', 1280)
                                orig_h = bbox_data.get('original_height', 720)

                                # Convert percentage coordinates to absolute pixels
                                value = bbox_data['value']

                                # Convert from (x%, y%, w%, h%) to (x1, y1, x2, y2) in pixels
                                x1 = value['x'] / 100.0 * orig_w
                                y1 = value['y'] / 100.0 * orig_h
                                w = value['width'] / 100.0 * orig_w
                                h = value['height'] / 100.0 * orig_h
                                x2 = x1 + w
                                y2 = y1 + h

                                bbox = (x1, y1, x2, y2)
                                frame_annotations.append((track_id, bbox))

            if frame_annotations:
                self.gt_data[frame_id] = frame_annotations

        print(f"Loaded {len(self.gt_data)} frames with ground truth annotations")

    def load_ground_truth_txt(self, txt_path: str):
        """
        Lädt Ground Truth aus MOT Format TXT Datei
        Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, ...

        Args:
            txt_path: Pfad zur TXT Datei
        """
        print(f"Loading Ground Truth from {txt_path}...")

        if not Path(txt_path).exists():
            print(f"Warning: GT file {txt_path} not found!")
            return

        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue

                frame_id = int(parts[0])
                track_id = int(parts[1])
                x1 = float(parts[2])
                y1 = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])

                # Convert from (x, y, w, h) to (x1, y1, x2, y2)
                x2 = x1 + w
                y2 = y1 + h
                bbox = (x1, y1, x2, y2)

                if frame_id not in self.gt_data:
                    self.gt_data[frame_id] = []
                self.gt_data[frame_id].append((track_id, bbox))

        print(f"Loaded {len(self.gt_data)} frames with ground truth annotations")

    def load_predictions_mot_format(self, mot_file: str):
        """
        Lädt Predictions aus MOT Challenge Format (.txt)
        Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        Args:
            mot_file: Pfad zur MOT Format .txt Datei
        """
        print(f"Loading Predictions from {mot_file}...")

        if not Path(mot_file).exists():
            print(f"Warning: Prediction file {mot_file} not found!")
            return

        with open(mot_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue

                frame_id = int(parts[0])
                track_id = int(parts[1])
                x1 = float(parts[2])
                y1 = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = float(parts[6])

                # Convert from (x, y, w, h) to (x1, y1, x2, y2)
                x2 = x1 + w
                y2 = y1 + h
                bbox = (x1, y1, x2, y2)

                if frame_id not in self.pred_data:
                    self.pred_data[frame_id] = []
                self.pred_data[frame_id].append((track_id, bbox, conf))

        print(f"Loaded {len(self.pred_data)} frames with predictions")

    @staticmethod
    def _extract_frame_id(image_path: str) -> Optional[int]:
        """Extrahiert Frame ID aus Bildpfad"""
        import re
        # Try to find frame number in various formats
        match = re.search(r'frame[_-]?(\d+)', image_path.lower())
        if match:
            return int(match.group(1))

        # Try to find just numbers in filename
        match = re.search(r'(\d+)\.(jpg|jpeg|png)', image_path.lower())
        if match:
            return int(match.group(1))

        return None

    @staticmethod
    def compute_iou(bbox1: Tuple[float, float, float, float],
                     bbox2: Tuple[float, float, float, float]) -> float:
        """
        Berechnet Intersection over Union (IoU) zwischen zwei Bounding Boxes

        Args:
            bbox1, bbox2: (x1, y1, x2, y2) format

        Returns:
            IoU Wert zwischen 0 und 1
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        # Union
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def compute_metrics(self) -> Dict:
        """
        Berechnet umfassende Tracking-Metriken

        Returns:
            Dictionary mit allen berechneten Metriken
        """
        print("\nComputing metrics...")

        # Frame-level statistics
        total_gt_detections = 0
        total_pred_detections = 0
        total_tp = 0  # True Positives (correct detections)
        total_fp = 0  # False Positives
        total_fn = 0  # False Negatives

        # Tracking-specific statistics
        id_switches = 0
        fragmentations = 0

        # IoU statistics
        iou_scores = []

        # Track history for computing ID switches
        prev_gt_to_pred_mapping = {}  # {gt_id: pred_id} mapping in previous frame

        # Per-frame analysis
        frame_metrics = []

        # Get common frames
        common_frames = sorted(set(self.gt_data.keys()) & set(self.pred_data.keys()))

        if not common_frames:
            print("Warning: No common frames between GT and predictions!")
            return {}

        print(f"Evaluating {len(common_frames)} common frames...")

        for frame_id in common_frames:
            gt_boxes = self.gt_data[frame_id]
            pred_boxes = self.pred_data[frame_id]

            total_gt_detections += len(gt_boxes)
            total_pred_detections += len(pred_boxes)

            # Create IoU matrix
            iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i, (_, gt_bbox) in enumerate(gt_boxes):
                for j, (_, pred_bbox, _) in enumerate(pred_boxes):
                    iou_matrix[i, j] = self.compute_iou(gt_bbox, pred_bbox)

            # Match GT to predictions using Hungarian algorithm (greedy approximation)
            matched_gt = set()
            matched_pred = set()
            gt_to_pred_mapping = {}

            # Greedy matching: match highest IoU first
            matches = []
            for i in range(len(gt_boxes)):
                for j in range(len(pred_boxes)):
                    if iou_matrix[i, j] >= self.iou_threshold:
                        matches.append((iou_matrix[i, j], i, j))

            matches.sort(reverse=True)

            for iou_val, i, j in matches:
                if i not in matched_gt and j not in matched_pred:
                    matched_gt.add(i)
                    matched_pred.add(j)
                    gt_id, _ = gt_boxes[i]
                    pred_id, _, _ = pred_boxes[j]
                    gt_to_pred_mapping[gt_id] = pred_id
                    iou_scores.append(iou_val)

            # Count TP, FP, FN
            tp = len(matched_gt)
            fp = len(pred_boxes) - len(matched_pred)
            fn = len(gt_boxes) - len(matched_gt)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            # Detect ID switches
            for gt_id, pred_id in gt_to_pred_mapping.items():
                if gt_id in prev_gt_to_pred_mapping:
                    if prev_gt_to_pred_mapping[gt_id] != pred_id:
                        id_switches += 1

            prev_gt_to_pred_mapping = gt_to_pred_mapping

            # Frame metrics
            frame_precision = tp / len(pred_boxes) if len(pred_boxes) > 0 else 0
            frame_recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0
            frame_f1 = 2 * frame_precision * frame_recall / (frame_precision + frame_recall) if (frame_precision + frame_recall) > 0 else 0

            frame_metrics.append({
                'frame_id': frame_id,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': frame_precision,
                'recall': frame_recall,
                'f1': frame_f1,
                'gt_count': len(gt_boxes),
                'pred_count': len(pred_boxes)
            })

        # Compute overall metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # MOTA (Multiple Object Tracking Accuracy)
        mota = 1 - (total_fn + total_fp + id_switches) / total_gt_detections if total_gt_detections > 0 else 0

        # Average IoU
        avg_iou = np.mean(iou_scores) if iou_scores else 0

        self.metrics = {
            'total_frames': len(common_frames),
            'total_gt_detections': total_gt_detections,
            'total_pred_detections': total_pred_detections,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'mota': mota,
            'id_switches': id_switches,
            'avg_iou': avg_iou,
            'iou_threshold': self.iou_threshold,
            'frame_metrics': frame_metrics
        }

        return self.metrics

    def compute_loss_functions(self) -> Dict:
        """
        Berechnet verschiedene Loss-Funktionen für die Evaluation

        Returns:
            Dictionary mit verschiedenen Loss-Werten
        """
        if not self.metrics:
            self.compute_metrics()

        # Return empty if no valid metrics
        if not self.metrics or 'precision' not in self.metrics:
            return {
                'detection_loss': 0.0,
                'iou_loss': 0.0,
                'tracking_loss': 0.0,
                'mota_loss': 0.0,
                'combined_loss': 0.0
            }

        # Detection Loss: Kombination aus Precision und Recall Fehler
        detection_loss = (1 - self.metrics['precision']) + (1 - self.metrics['recall'])

        # IoU Loss: 1 - average IoU
        iou_loss = 1 - self.metrics['avg_iou']

        # Tracking Loss: basiert auf ID switches
        # Normalisiert durch Anzahl der GT Detections
        tracking_loss = self.metrics['id_switches'] / self.metrics['total_gt_detections'] \
                       if self.metrics['total_gt_detections'] > 0 else 0

        # MOTA Loss: 1 - MOTA (höher ist schlechter)
        mota_loss = 1 - self.metrics['mota']

        # Combined Loss: gewichtete Summe
        # Gewichtung: Detection (40%), IoU (30%), Tracking (30%)
        combined_loss = 0.4 * detection_loss + 0.3 * iou_loss + 0.3 * tracking_loss

        losses = {
            'detection_loss': detection_loss,
            'iou_loss': iou_loss,
            'tracking_loss': tracking_loss,
            'mota_loss': mota_loss,
            'combined_loss': combined_loss
        }

        return losses

    def print_summary(self):
        """Gibt eine Zusammenfassung der Metriken aus"""
        if not self.metrics:
            print("No metrics computed yet. Run compute_metrics() first.")
            return

        print("\n" + "="*80)
        print("MOTRv2 EVALUATION SUMMARY")
        print("="*80)

        print(f"\n📊 Dataset Statistics:")
        print(f"  Total Frames:           {self.metrics['total_frames']}")
        print(f"  GT Detections:          {self.metrics['total_gt_detections']}")
        print(f"  Predicted Detections:   {self.metrics['total_pred_detections']}")

        print(f"\n🎯 Detection Metrics:")
        print(f"  True Positives:         {self.metrics['true_positives']}")
        print(f"  False Positives:        {self.metrics['false_positives']}")
        print(f"  False Negatives:        {self.metrics['false_negatives']}")
        print(f"  Precision:              {self.metrics['precision']:.4f}")
        print(f"  Recall:                 {self.metrics['recall']:.4f}")
        print(f"  F1-Score:               {self.metrics['f1_score']:.4f}")
        print(f"  Average IoU:            {self.metrics['avg_iou']:.4f}")

        print(f"\n🔄 Tracking Metrics:")
        print(f"  MOTA:                   {self.metrics['mota']:.4f}")
        print(f"  ID Switches:            {self.metrics['id_switches']}")

        # Compute and display losses
        losses = self.compute_loss_functions()
        print(f"\n📉 Loss Functions:")
        print(f"  Detection Loss:         {losses['detection_loss']:.4f}")
        print(f"  IoU Loss:               {losses['iou_loss']:.4f}")
        print(f"  Tracking Loss:          {losses['tracking_loss']:.4f}")
        print(f"  MOTA Loss:              {losses['mota_loss']:.4f}")
        print(f"  Combined Loss:          {losses['combined_loss']:.4f}")

        print("\n" + "="*80)

    def plot_results(self, output_dir: str = "analysis/plots"):
        """
        Erstellt umfassende Visualisierungen der Evaluationsergebnisse

        Args:
            output_dir: Verzeichnis für die Ausgabe der Plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not self.metrics:
            print("No metrics to plot. Run compute_metrics() first.")
            return

        # 1. Metrics Overview Bar Chart
        self._plot_metrics_overview(output_path)

        # 2. Loss Functions Comparison
        self._plot_loss_functions(output_path)

        # 3. Per-Frame Metrics
        self._plot_frame_metrics(output_path)

        # 4. Confusion Matrix Style Plot
        self._plot_detection_confusion(output_path)

        # 5. Performance Summary Dashboard
        self._plot_summary_dashboard(output_path)

        print(f"\n✅ All plots saved to {output_path}")

    def _plot_metrics_overview(self, output_path: Path):
        """Plot overview of main metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Detection Metrics
        metrics_names = ['Precision', 'Recall', 'F1-Score', 'Avg IoU', 'MOTA']
        metrics_values = [
            self.metrics['precision'],
            self.metrics['recall'],
            self.metrics['f1_score'],
            self.metrics['avg_iou'],
            self.metrics['mota']
        ]

        colors = ['#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#e74c3c']
        bars = axes[0].bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_ylim([0, 1])
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].set_title('MOTRv2 Performance Metrics', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=10)

        # Detection Counts
        count_names = ['True\nPositives', 'False\nPositives', 'False\nNegatives']
        count_values = [
            self.metrics['true_positives'],
            self.metrics['false_positives'],
            self.metrics['false_negatives']
        ]

        count_colors = ['#2ecc71', '#e74c3c', '#f39c12']
        bars = axes[1].bar(count_names, count_values, color=count_colors, alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Detection Counts', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path / 'metrics_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved metrics_overview.png")

    def _plot_loss_functions(self, output_path: Path):
        """Plot comparison of different loss functions"""
        losses = self.compute_loss_functions()

        fig, ax = plt.subplots(figsize=(12, 6))

        loss_names = ['Detection\nLoss', 'IoU\nLoss', 'Tracking\nLoss', 'MOTA\nLoss', 'Combined\nLoss']
        loss_values = [
            losses['detection_loss'],
            losses['iou_loss'],
            losses['tracking_loss'],
            losses['mota_loss'],
            losses['combined_loss']
        ]

        # Color gradient from green (low loss) to red (high loss)
        colors_map = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(loss_values)))

        bars = ax.bar(loss_names, loss_values, color=colors_map, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.set_title('MOTRv2 Loss Functions Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path / 'loss_functions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved loss_functions.png")

    def _plot_frame_metrics(self, output_path: Path):
        """Plot per-frame metrics over time"""
        frame_metrics = self.metrics['frame_metrics']
        frames = [m['frame_id'] for m in frame_metrics]
        precisions = [m['precision'] for m in frame_metrics]
        recalls = [m['recall'] for m in frame_metrics]
        f1_scores = [m['f1'] for m in frame_metrics]

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Plot 1: Precision, Recall, F1 over frames
        axes[0].plot(frames, precisions, label='Precision', marker='o', markersize=3, linewidth=2, alpha=0.8)
        axes[0].plot(frames, recalls, label='Recall', marker='s', markersize=3, linewidth=2, alpha=0.8)
        axes[0].plot(frames, f1_scores, label='F1-Score', marker='^', markersize=3, linewidth=2, alpha=0.8)
        axes[0].set_xlabel('Frame ID', fontsize=12)
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].set_title('Per-Frame Detection Performance', fontsize=14, fontweight='bold')
        axes[0].legend(loc='best', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.1])

        # Plot 2: GT vs Predicted detections count
        gt_counts = [m['gt_count'] for m in frame_metrics]
        pred_counts = [m['pred_count'] for m in frame_metrics]

        axes[1].plot(frames, gt_counts, label='Ground Truth', marker='o', markersize=3, linewidth=2, alpha=0.8, color='green')
        axes[1].plot(frames, pred_counts, label='Predictions', marker='s', markersize=3, linewidth=2, alpha=0.8, color='blue')
        axes[1].fill_between(frames, gt_counts, pred_counts, alpha=0.2)
        axes[1].set_xlabel('Frame ID', fontsize=12)
        axes[1].set_ylabel('Number of Detections', fontsize=12)
        axes[1].set_title('Ground Truth vs Predicted Detections per Frame', fontsize=14, fontweight='bold')
        axes[1].legend(loc='best', fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'frame_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved frame_metrics.png")

    def _plot_detection_confusion(self, output_path: Path):
        """Plot detection results as confusion-style visualization"""
        fig, ax = plt.subplots(figsize=(8, 6))

        tp = self.metrics['true_positives']
        fp = self.metrics['false_positives']
        fn = self.metrics['false_negatives']

        # Create a 2x2 style visualization
        data = np.array([[tp, fn], [fp, 0]])

        # Custom colormap
        cmap = plt.cm.RdYlGn

        im = ax.imshow([[tp, fn], [fp, 0]], cmap=cmap, alpha=0.6, vmin=0, vmax=max(tp, fp, fn))

        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Detected', 'Missed'], fontsize=12)
        ax.set_yticklabels(['Actual Objects', 'False Alarms'], fontsize=12)

        # Add text annotations
        for i in range(2):
            for j in range(2):
                if not (i == 1 and j == 1):  # Skip bottom-right
                    label = ''
                    value = data[i, j]
                    if i == 0 and j == 0:
                        label = f'True Positives\n{int(value)}'
                    elif i == 0 and j == 1:
                        label = f'False Negatives\n{int(value)}'
                    elif i == 1 and j == 0:
                        label = f'False Positives\n{int(value)}'

                    ax.text(j, i, label, ha='center', va='center',
                           fontsize=14, fontweight='bold', color='black')

        ax.set_title('Detection Confusion Matrix', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Count')

        plt.tight_layout()
        plt.savefig(output_path / 'detection_confusion.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved detection_confusion.png")

    def _plot_summary_dashboard(self, output_path: Path):
        """Create a comprehensive summary dashboard"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle('MOTRv2 Evaluation Dashboard', fontsize=18, fontweight='bold', y=0.98)

        # 1. Key Metrics (top-left, large)
        ax1 = fig.add_subplot(gs[0, :2])
        metrics_data = {
            'Precision': self.metrics['precision'],
            'Recall': self.metrics['recall'],
            'F1-Score': self.metrics['f1_score'],
            'MOTA': self.metrics['mota'],
            'Avg IoU': self.metrics['avg_iou']
        }
        bars = ax1.barh(list(metrics_data.keys()), list(metrics_data.values()),
                       color=plt.cm.viridis(np.linspace(0.3, 0.9, len(metrics_data))),
                       alpha=0.8, edgecolor='black')
        ax1.set_xlim([0, 1])
        ax1.set_xlabel('Score', fontsize=11)
        ax1.set_title('Key Performance Metrics', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', va='center', fontsize=10, fontweight='bold')

        # 2. Statistics Box (top-right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        stats_text = f"""
        DATASET STATISTICS
        ═══════════════════

        Total Frames: {self.metrics['total_frames']}

        GT Detections: {self.metrics['total_gt_detections']}

        Predictions: {self.metrics['total_pred_detections']}

        ───────────────────

        ID Switches: {self.metrics['id_switches']}

        IoU Threshold: {self.metrics['iou_threshold']}
        """
        ax2.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # 3. Losses (middle-left)
        ax3 = fig.add_subplot(gs[1, 0])
        losses = self.compute_loss_functions()
        loss_data = {
            'Detection': losses['detection_loss'],
            'IoU': losses['iou_loss'],
            'Tracking': losses['tracking_loss'],
            'Combined': losses['combined_loss']
        }
        wedges, texts, autotexts = ax3.pie(list(loss_data.values()), labels=list(loss_data.keys()),
                                            autopct='%1.1f%%', startangle=90,
                                            colors=plt.cm.Set3(range(len(loss_data))))
        ax3.set_title('Loss Distribution', fontsize=12, fontweight='bold')

        # 4. TP/FP/FN Pie Chart (middle-center)
        ax4 = fig.add_subplot(gs[1, 1])
        detection_data = {
            'True Positives': self.metrics['true_positives'],
            'False Positives': self.metrics['false_positives'],
            'False Negatives': self.metrics['false_negatives']
        }
        colors_det = ['#2ecc71', '#e74c3c', '#f39c12']
        wedges, texts, autotexts = ax4.pie(list(detection_data.values()), labels=list(detection_data.keys()),
                                            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(detection_data.values()))})',
                                            startangle=90, colors=colors_det, textprops={'fontsize': 9})
        ax4.set_title('Detection Breakdown', fontsize=12, fontweight='bold')

        # 5. Score Radar Chart (middle-right)
        ax5 = fig.add_subplot(gs[1, 2], projection='polar')
        categories = ['Precision', 'Recall', 'F1-Score', 'MOTA', 'Avg IoU']
        values = [
            self.metrics['precision'],
            self.metrics['recall'],
            self.metrics['f1_score'],
            self.metrics['mota'],
            self.metrics['avg_iou']
        ]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        ax5.plot(angles, values, 'o-', linewidth=2, color='#3498db')
        ax5.fill(angles, values, alpha=0.25, color='#3498db')
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(categories, fontsize=9)
        ax5.set_ylim(0, 1)
        ax5.set_title('Performance Radar', fontsize=12, fontweight='bold', pad=20)
        ax5.grid(True)

        # 6. Frame-wise Performance (bottom, spanning all columns)
        ax6 = fig.add_subplot(gs[2, :])
        frame_metrics = self.metrics['frame_metrics']
        frames = [m['frame_id'] for m in frame_metrics]
        f1_scores = [m['f1'] for m in frame_metrics]

        ax6.plot(frames, f1_scores, linewidth=2, color='#9b59b6', alpha=0.8)
        ax6.fill_between(frames, 0, f1_scores, alpha=0.3, color='#9b59b6')
        ax6.set_xlabel('Frame ID', fontsize=11)
        ax6.set_ylabel('F1-Score', fontsize=11)
        ax6.set_title('Per-Frame F1-Score', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([0, 1.1])

        # Add mean line
        mean_f1 = np.mean(f1_scores)
        ax6.axhline(y=mean_f1, color='red', linestyle='--', linewidth=2,
                   label=f'Mean F1: {mean_f1:.3f}', alpha=0.7)
        ax6.legend(loc='best')

        plt.savefig(output_path / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved summary_dashboard.png")

    def save_metrics_json(self, output_path: str = "analysis/metrics_results.json"):
        """Save all metrics and losses to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'metrics': {k: v for k, v in self.metrics.items() if k != 'frame_metrics'},
            'losses': self.compute_loss_functions(),
            'frame_metrics': self.metrics.get('frame_metrics', [])
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Metrics saved to {output_file}")


def evaluate(gt_path: str, pred_path: str, output_dir: str, label: str,
             iou_threshold: float = 0.5, gt_format: str = "txt"):
    """
    Fuehrt eine einzelne Evaluation durch.

    Args:
        gt_path: Pfad zur Ground Truth Datei (TXT oder JSON)
        pred_path: Pfad zur Predictions Datei (TXT)
        output_dir: Ausgabeverzeichnis fuer Plots und Metriken
        label: Name/Label fuer diese Evaluation
        iou_threshold: IoU Schwellenwert
        gt_format: "txt" oder "json"

    Returns:
        MOTEvaluator instance mit berechneten Metriken
    """
    print(f"\n{'='*80}")
    print(f"EVALUATION: {label}")
    print(f"{'='*80}\n")

    evaluator = MOTEvaluator(iou_threshold=iou_threshold)

    # Lade GT
    if gt_format == "json":
        evaluator.load_ground_truth_json(gt_path)
    else:
        evaluator.load_ground_truth_txt(gt_path)

    # Lade Predictions
    evaluator.load_predictions_mot_format(pred_path)

    # Berechne Metriken
    metrics = evaluator.compute_metrics()
    if not metrics:
        print(f"Skipping {label}: no common frames.")
        return None

    evaluator.print_summary()
    evaluator.save_metrics_json(f"{output_dir}/metrics.json")

    print(f"\nCreating visualizations...")
    evaluator.plot_results(f"{output_dir}/plots")

    return evaluator


def main():
    """Main evaluation pipeline - Beach Volleyball Validation"""
    import argparse

    parser = argparse.ArgumentParser(description='MOTRv2 Evaluation')
    parser.add_argument('--gt', default='../Datasets/Sequenz_Beach/sequenz_beach_valid_gt.txt',
                        help='Ground Truth TXT file')
    parser.add_argument('--pred', default=None,
                        help='Single prediction file to evaluate')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IoU threshold (default: 0.5)')
    parser.add_argument('--output', default='analysis/results',
                        help='Output directory')
    args = parser.parse_args()

    gt_path = args.gt

    if args.pred:
        # Einzelne Evaluation
        name = Path(args.pred).stem
        evaluate(gt_path, args.pred, f"{args.output}/{name}", name, args.iou)
    else:
        # Alle verfuegbaren Inference-Ergebnisse evaluieren
        predictions = {
            'MOTRv2 + YOLOX (Vanilla)': 'outputs/inference_motrv2_yolox_vanilla/inference_motrv2_yolox_vanilla_th0.4_mt30.txt',
            'YOLOX Detections only': 'analysis/yolox_detections_sequenz_beach.txt',
        }

        # Finde alle finetune inference Ergebnisse
        for p in sorted(Path('outputs').glob('inference_v2_*/tracking_inference.txt')):
            name = p.parent.name.replace('inference_v2_', 'Finetuned: ')
            predictions[name] = str(p)

        results = {}
        for label, pred_path in predictions.items():
            if not Path(pred_path).exists():
                print(f"\nSkipping {label}: {pred_path} not found")
                continue

            safe_name = label.replace(' ', '_').replace('+', '').replace(':', '_').replace('(', '').replace(')', '')
            evaluator = evaluate(gt_path, pred_path, f"{args.output}/{safe_name}", label, args.iou)
            if evaluator and evaluator.metrics:
                results[label] = {
                    'metrics': {k: v for k, v in evaluator.metrics.items() if k != 'frame_metrics'},
                    'losses': evaluator.compute_loss_functions()
                }

        # Vergleichstabelle ausgeben
        if results:
            print(f"\n\n{'='*100}")
            print("COMPARISON TABLE")
            print(f"{'='*100}")
            print(f"{'Model':<40} {'MOTA':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AvgIoU':>8} {'FP':>8} {'FN':>8}")
            print("-" * 100)
            for label, data in results.items():
                m = data['metrics']
                print(f"{label:<40} {m['mota']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} "
                      f"{m['f1_score']:>8.4f} {m['avg_iou']:>8.4f} {m['false_positives']:>8d} {m['false_negatives']:>8d}")
            print(f"{'='*100}")

            # Speichere Vergleich als JSON
            comparison_path = Path(args.output) / 'comparison.json'
            with open(comparison_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nComparison saved to {comparison_path}")


if __name__ == "__main__":
    main()
