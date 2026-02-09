#!/usr/bin/env python3
"""
Beispiel-Script zur Nutzung des MOT Evaluators
Zeigt verschiedene Anwendungsfälle und Konfigurationen
"""

from mot_evaluation import MOTEvaluator
from pathlib import Path


def evaluate_single_run(gt_path: str, pred_path: str, output_name: str, iou_threshold: float = 0.5):
    """
    Evaluiert einen einzelnen Inference-Lauf

    Args:
        gt_path: Pfad zur Ground Truth JSON
        pred_path: Pfad zur Prediction TXT (MOT Format)
        output_name: Name für die Ausgabe-Dateien
        iou_threshold: IoU Schwellenwert für Matching
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {output_name}")
    print(f"{'='*80}")

    evaluator = MOTEvaluator(iou_threshold=iou_threshold)

    # Lade Daten
    evaluator.load_ground_truth_json(gt_path)
    evaluator.load_predictions_mot_format(pred_path)

    # Berechne Metriken
    evaluator.compute_metrics()

    # Zeige Zusammenfassung
    evaluator.print_summary()

    # Speichere Ergebnisse
    output_dir = f"analysis/results/{output_name}"
    evaluator.save_metrics_json(f"{output_dir}/metrics.json")
    evaluator.plot_results(f"{output_dir}/plots")


def compare_multiple_models():
    """
    Vergleicht mehrere Modell-Versionen oder Konfigurationen
    """
    # Ground Truth (immer gleich)
    gt_path = "data/Dataset/mot/sportsmot_detections_gt_train_onethird.json"

    # Verschiedene Modell-Outputs
    models_to_compare = [
        {
            'name': 'baseline',
            'pred_path': 'outputs/finetune_v1/inference_test/tracking_inference.txt',
            'iou_threshold': 0.5
        },
        {
            'name': 'finetuned_moderate',
            'pred_path': 'outputs/finetune_v1/inference_moderate_lr1e5_ep5/tracking_inference.txt',
            'iou_threshold': 0.5
        },
        {
            'name': 'finetuned_aggressive',
            'pred_path': 'outputs/finetune_v1/inference_aggressive_lr1e5_ep5/tracking_inference.txt',
            'iou_threshold': 0.5
        },
    ]

    results = {}

    for model_config in models_to_compare:
        name = model_config['name']
        pred_path = model_config['pred_path']
        iou_threshold = model_config['iou_threshold']

        # Überspringe, wenn Datei nicht existiert
        if not Path(pred_path).exists():
            print(f"⚠️  Skipping {name}: File {pred_path} not found")
            continue

        evaluator = MOTEvaluator(iou_threshold=iou_threshold)

        try:
            evaluator.load_ground_truth_json(gt_path)
            evaluator.load_predictions_mot_format(pred_path)
            metrics = evaluator.compute_metrics()
            losses = evaluator.compute_loss_functions()

            results[name] = {
                'metrics': metrics,
                'losses': losses
            }

            # Speichere individuelle Plots
            output_dir = f"analysis/comparison/{name}"
            evaluator.save_metrics_json(f"{output_dir}/metrics.json")
            evaluator.plot_results(f"{output_dir}/plots")

        except Exception as e:
            print(f"❌ Error evaluating {name}: {e}")
            continue

    # Vergleichs-Visualisierung
    if results:
        create_comparison_plot(results)


def create_comparison_plot(results: dict):
    """
    Erstellt einen Vergleichsplot für mehrere Modelle
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')

    models = list(results.keys())
    x_pos = np.arange(len(models))

    # 1. Precision, Recall, F1
    ax = axes[0, 0]
    width = 0.25
    precisions = [results[m]['metrics']['precision'] for m in models]
    recalls = [results[m]['metrics']['recall'] for m in models]
    f1s = [results[m]['metrics']['f1_score'] for m in models]

    ax.bar(x_pos - width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x_pos, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x_pos + width, f1s, width, label='F1-Score', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Detection Metrics')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])

    # 2. MOTA und Avg IoU
    ax = axes[0, 1]
    motas = [results[m]['metrics']['mota'] for m in models]
    ious = [results[m]['metrics']['avg_iou'] for m in models]

    ax.bar(x_pos - width/2, motas, width, label='MOTA', alpha=0.8)
    ax.bar(x_pos + width/2, ious, width, label='Avg IoU', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Tracking Metrics')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])

    # 3. Loss Functions
    ax = axes[1, 0]
    detection_losses = [results[m]['losses']['detection_loss'] for m in models]
    tracking_losses = [results[m]['losses']['tracking_loss'] for m in models]
    combined_losses = [results[m]['losses']['combined_loss'] for m in models]

    ax.bar(x_pos - width, detection_losses, width, label='Detection Loss', alpha=0.8)
    ax.bar(x_pos, tracking_losses, width, label='Tracking Loss', alpha=0.8)
    ax.bar(x_pos + width, combined_losses, width, label='Combined Loss', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Comparison (Lower is Better)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 4. ID Switches
    ax = axes[1, 1]
    id_switches = [results[m]['metrics']['id_switches'] for m in models]

    bars = ax.bar(models, id_switches, alpha=0.8, color='coral', edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('ID Switches (Lower is Better)')
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom')

    plt.tight_layout()
    output_path = Path("analysis/comparison/model_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Comparison plot saved to {output_path}")


def evaluate_with_different_thresholds():
    """
    Evaluiert dasselbe Modell mit verschiedenen IoU-Schwellenwerten
    """
    gt_path = "data/Dataset/mot/sportsmot_detections_gt_train_onethird.json"
    pred_path = "outputs/finetune_v1/inference_test/tracking_inference.txt"

    thresholds = [0.3, 0.5, 0.7, 0.9]
    results = {}

    for threshold in thresholds:
        print(f"\n{'='*80}")
        print(f"Evaluating with IoU threshold: {threshold}")
        print(f"{'='*80}")

        evaluator = MOTEvaluator(iou_threshold=threshold)
        evaluator.load_ground_truth_json(gt_path)
        evaluator.load_predictions_mot_format(pred_path)

        metrics = evaluator.compute_metrics()
        losses = evaluator.compute_loss_functions()

        results[threshold] = {
            'metrics': metrics,
            'losses': losses
        }

        # Speichere Ergebnisse
        output_dir = f"analysis/threshold_study/iou_{threshold}"
        evaluator.save_metrics_json(f"{output_dir}/metrics.json")

    # Plot threshold sensitivity
    plot_threshold_sensitivity(results)


def plot_threshold_sensitivity(results: dict):
    """Plot wie Metriken sich mit IoU-Threshold ändern"""
    import matplotlib.pyplot as plt

    thresholds = sorted(results.keys())

    metrics_to_plot = ['precision', 'recall', 'f1_score', 'mota']
    metric_labels = ['Precision', 'Recall', 'F1-Score', 'MOTA']

    fig, ax = plt.subplots(figsize=(12, 7))

    for metric, label in zip(metrics_to_plot, metric_labels):
        values = [results[t]['metrics'][metric] for t in thresholds]
        ax.plot(thresholds, values, marker='o', label=label, linewidth=2, markersize=8)

    ax.set_xlabel('IoU Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Metric Sensitivity to IoU Threshold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    output_path = Path("analysis/threshold_study/threshold_sensitivity.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Threshold sensitivity plot saved to {output_path}")


if __name__ == "__main__":
    import sys

    print("\n" + "="*80)
    print("MOTRv2 EVALUATION - EXAMPLE USAGE")
    print("="*80)

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("\nAvailable modes:")
        print("  1. single       - Evaluate single model run")
        print("  2. compare      - Compare multiple models")
        print("  3. thresholds   - Test different IoU thresholds")
        print("\nUsage: python example_usage.py [mode]")
        mode = input("\nSelect mode (1-3): ").strip()

    if mode in ['1', 'single']:
        # Beispiel für einzelne Evaluation
        evaluate_single_run(
            gt_path="data/Dataset/mot/sportsmot_detections_gt_train_onethird.json",
            pred_path="outputs/finetune_v1/inference_test/tracking_inference.txt",
            output_name="baseline_evaluation",
            iou_threshold=0.5
        )

    elif mode in ['2', 'compare']:
        # Vergleiche mehrere Modelle
        compare_multiple_models()

    elif mode in ['3', 'thresholds']:
        # Teste verschiedene Thresholds
        evaluate_with_different_thresholds()

    else:
        print(f"❌ Unknown mode: {mode}")
        sys.exit(1)

    print("\n" + "="*80)
    print("✅ DONE!")
    print("="*80 + "\n")
