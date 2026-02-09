# MOTRv2 Evaluation Framework

Umfassendes Framework zur quantitativen Analyse und Visualisierung von MOTRv2 Tracking-Ergebnissen.

## Features

- 📊 **Umfassende Metriken**: Precision, Recall, F1-Score, MOTA, IoU
- 📉 **Loss-Funktionen**: Detection Loss, IoU Loss, Tracking Loss, Combined Loss
- 📈 **Visualisierungen**: 6 verschiedene Plot-Typen für detaillierte Analyse
- 🔄 **Modell-Vergleich**: Vergleiche mehrere Modell-Versionen
- 🎯 **Threshold-Analyse**: Teste verschiedene IoU-Schwellenwerte
- 💾 **Export**: JSON-Export aller Metriken

## Installation

```bash
cd analysis
pip install -r requirements.txt
```

## Schnellstart

### 1. Einzelne Evaluation

```python
from mot_evaluation import MOTEvaluator

evaluator = MOTEvaluator(iou_threshold=0.5)

# Lade Daten
evaluator.load_ground_truth_json("path/to/ground_truth.json")
evaluator.load_predictions_mot_format("path/to/predictions.txt")

# Berechne Metriken
evaluator.compute_metrics()

# Zeige Zusammenfassung
evaluator.print_summary()

# Erstelle Visualisierungen
evaluator.plot_results("output/plots")

# Speichere Metriken
evaluator.save_metrics_json("output/metrics.json")
```

### 2. Automatische Evaluation

```bash
# Einzelne Evaluation
python mot_evaluation.py

# Mit eigenen Pfaden
python example_usage.py 1

# Mehrere Modelle vergleichen
python example_usage.py 2

# Threshold-Analyse
python example_usage.py 3
```

## Datenformat

### Ground Truth (JSON - Label Studio Format)

```json
[
  {
    "id": 1,
    "data": {
      "image": "/path/to/frame_0001.jpg"
    },
    "annotations": [
      {
        "result": [
          {
            "id": "track_0",
            "type": "rectanglelabels",
            "value": {
              "x": 50.5,
              "y": 30.2,
              "width": 10.0,
              "height": 15.0,
              "original_width": 1280,
              "original_height": 720,
              "rectanglelabels": ["player"]
            }
          }
        ]
      }
    ]
  }
]
```

### Predictions (TXT - MOT Challenge Format)

```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
1, 1, 100.5, 200.3, 50.2, 80.1, 0.95, -1, -1, -1
1, 2, 300.1, 150.7, 45.8, 75.3, 0.87, -1, -1, -1
2, 1, 102.3, 205.1, 51.0, 81.2, 0.93, -1, -1, -1
```

## Berechnete Metriken

### Detection Metriken
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Average IoU**: Durchschnittliche Intersection over Union für alle Matches

### Tracking Metriken
- **MOTA** (Multiple Object Tracking Accuracy): 1 - (FN + FP + ID_SW) / GT
- **ID Switches**: Anzahl der Tracking-ID Wechsel

### Loss-Funktionen
- **Detection Loss**: (1 - Precision) + (1 - Recall)
- **IoU Loss**: 1 - Average IoU
- **Tracking Loss**: ID_Switches / Total_GT_Detections
- **MOTA Loss**: 1 - MOTA
- **Combined Loss**: 0.4 × Detection + 0.3 × IoU + 0.3 × Tracking

## Generierte Visualisierungen

1. **metrics_overview.png**: Übersicht aller Hauptmetriken
2. **loss_functions.png**: Vergleich aller Loss-Funktionen
3. **frame_metrics.png**: Frame-für-Frame Analyse
4. **detection_confusion.png**: Confusion-Matrix-Style Darstellung
5. **summary_dashboard.png**: Umfassendes Dashboard mit allen Metriken
6. **model_comparison.png**: Vergleich mehrerer Modelle (bei Verwendung von example_usage.py)

## Ausgabe-Struktur

```
analysis/
├── mot_evaluation.py          # Haupt-Evaluation-Skript
├── example_usage.py            # Beispiele und erweiterte Nutzung
├── requirements.txt            # Dependencies
├── README.md                   # Diese Datei
├── plots/                      # Visualisierungen
│   ├── metrics_overview.png
│   ├── loss_functions.png
│   ├── frame_metrics.png
│   ├── detection_confusion.png
│   └── summary_dashboard.png
├── metrics_results.json        # Exportierte Metriken
└── comparison/                 # Modell-Vergleiche
    ├── model_comparison.png
    └── [model_name]/
        ├── plots/
        └── metrics.json
```

## Erweiterte Nutzung

### Custom IoU Threshold

```python
evaluator = MOTEvaluator(iou_threshold=0.7)
```

### Vergleich mehrerer Modelle

```python
from example_usage import compare_multiple_models
compare_multiple_models()
```

### Threshold-Sensitivitäts-Analyse

```python
from example_usage import evaluate_with_different_thresholds
evaluate_with_different_thresholds()
```

## Anpassung

### Eigene Loss-Funktion hinzufügen

```python
def compute_custom_loss(self):
    # Eigene Loss-Berechnung
    custom_loss = ...
    return custom_loss
```

### Eigene Visualisierung hinzufügen

```python
def _plot_custom_visualization(self, output_path):
    # Eigener Plot
    fig, ax = plt.subplots()
    # ... plotting code
    plt.savefig(output_path / 'custom_plot.png')
```

## Troubleshooting

### "No common frames between GT and predictions"
- Überprüfe, dass Frame-IDs in beiden Dateien übereinstimmen
- Prüfe das Dateiformat (siehe "Datenformat" oben)

### "File not found"
- Passe die Pfade in `mot_evaluation.py` oder `example_usage.py` an
- Stelle sicher, dass du vom MOTRv2 Root-Verzeichnis aus ausführst

### Plots sehen schlecht aus
- Erhöhe DPI: `plt.savefig(..., dpi=600)`
- Passe Figure-Size an: `plt.figure(figsize=(20, 15))`

## Für dein Forschungsprojekt

**Sobald du finale Ground Truth Daten hast:**

1. Ersetze den Pfad in `mot_evaluation.py` Zeile 723:
   ```python
   gt_json_path = "path/to/your/final_ground_truth.json"
   ```

2. Führe die Evaluation aus:
   ```bash
   python mot_evaluation.py
   ```

3. Alle Metriken, Losses und Visualisierungen werden automatisch generiert!

## Zitierung

Wenn du dieses Framework in deiner Forschung verwendest, erwähne bitte:
- MOTRv2 Paper: [Link zum Paper]
- MOT Challenge Metrics: https://motchallenge.net/

## Support

Bei Fragen oder Problemen:
1. Überprüfe die Datenformate
2. Schaue in `example_usage.py` für Beispiele
3. Prüfe die Konsolen-Ausgabe für detaillierte Fehlermeldungen
