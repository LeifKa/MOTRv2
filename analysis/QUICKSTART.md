# MOTRv2 Evaluation - Schnellstart

## 🚀 Sofort loslegen

### Demo mit synthetischen Daten

```bash
cd /home/es/es_es/es_lekamt00/BeachKI/MOTRv2
python3 analysis/demo_evaluation.py
```

**Output:**
- `analysis/demo_data/plots/` - 5 verschiedene Visualisierungen
- `analysis/demo_data/demo_metrics.json` - Alle Metriken als JSON

---

## 📋 Wenn du deine finalen GT-Daten hast

### 1. Führe MOTRv2 Inferenz aus

```bash
# Dein Inference-Befehl (Beispiel)
python3 submit_dance.py --exp_name my_experiment --checkpoint path/to/checkpoint.pth
```

### 2. Passe die Pfade in mot_evaluation.py an

Öffne `analysis/mot_evaluation.py` und ändere in der `main()` Funktion (Zeile ~767):

```python
# ANPASSEN: Deine Ground Truth JSON
gt_json_path = "data/Dataset/mot/deine_finale_gt.json"

# ANPASSEN: Deine Inference-Ergebnisse
pred_txt_path = "outputs/dein_experiment/tracking_results.txt"
```

### 3. Führe die Evaluation aus

```bash
python3 analysis/mot_evaluation.py
```

### 4. Ergebnisse ansehen

```bash
# Plots
ls analysis/plots/

# Metriken
cat analysis/metrics_results.json | python3 -m json.tool
```

---

## 🔬 Erweiterte Nutzung

### Mehrere Modelle vergleichen

```bash
python3 analysis/example_usage.py 2
```

**Vorher anpassen:** Editiere `example_usage.py` und füge deine Modell-Pfade in `models_to_compare` hinzu (Zeile ~51).

### Verschiedene IoU-Thresholds testen

```bash
python3 analysis/example_usage.py 3
```

### Programmatische Nutzung

```python
from mot_evaluation import MOTEvaluator

# Initialisiere
evaluator = MOTEvaluator(iou_threshold=0.5)

# Lade Daten
evaluator.load_ground_truth_json("path/to/gt.json")
evaluator.load_predictions_mot_format("path/to/predictions.txt")

# Evaluiere
metrics = evaluator.compute_metrics()
losses = evaluator.compute_loss_functions()

# Speichere
evaluator.print_summary()
evaluator.plot_results("output_dir/plots")
evaluator.save_metrics_json("output_dir/metrics.json")

# Zugriff auf Metriken
print(f"MOTA: {metrics['mota']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
print(f"Combined Loss: {losses['combined_loss']:.4f}")
```

---

## 📊 Generierte Visualisierungen

1. **metrics_overview.png**
   - Balkendiagramme für Precision, Recall, F1, IoU, MOTA
   - TP/FP/FN Counts

2. **loss_functions.png**
   - Vergleich aller Loss-Funktionen
   - Color-coded (grün=gut, rot=schlecht)

3. **frame_metrics.png**
   - Performance über Zeit
   - GT vs Predicted Detections pro Frame

4. **detection_confusion.png**
   - Confusion-Matrix Style
   - TP, FP, FN Visualisierung

5. **summary_dashboard.png**
   - Umfassendes Dashboard
   - Kombiniert alle wichtigen Metriken

---

## 🎯 Metriken-Interpretation

### Gute Werte
- **Precision/Recall/F1**: > 0.8
- **MOTA**: > 0.6
- **Avg IoU**: > 0.7
- **ID Switches**: < 10% der GT Detections

### Akzeptable Werte
- **Precision/Recall/F1**: 0.5 - 0.8
- **MOTA**: 0.3 - 0.6
- **Avg IoU**: 0.5 - 0.7
- **ID Switches**: 10-20%

### Schlechte Werte (Training nötig)
- **Precision/Recall/F1**: < 0.5
- **MOTA**: < 0.3
- **Avg IoU**: < 0.5
- **ID Switches**: > 20%

---

## 🔧 Troubleshooting

### "No common frames between GT and predictions"

**Problem:** Frame-IDs stimmen nicht überein

**Lösung:**
1. Überprüfe die `_extract_frame_id()` Funktion in `mot_evaluation.py`
2. Passe das Regex-Pattern an dein Naming-Schema an
3. Oder: Benenne deine Bilder einheitlich (z.B. `frame_000001.jpg`)

### Plots sehen komisch aus

**Lösung:**
```python
# In mot_evaluation.py, ändere:
plt.rcParams['figure.figsize'] = (20, 15)  # Größer
plt.savefig(..., dpi=600)  # Höhere Auflösung
```

### Speicher-Probleme bei großen Dateien

**Lösung:** Verwende einen Subset der Daten
```python
# Nur erste N Frames evaluieren
evaluator.gt_data = dict(list(evaluator.gt_data.items())[:N])
evaluator.pred_data = dict(list(evaluator.pred_data.items())[:N])
```

---

## 📚 Weitere Infos

- **Vollständige Dokumentation:** `analysis/README.md`
- **Beispielcode:** `analysis/example_usage.py`
- **Demo:** `analysis/demo_evaluation.py`

---

## ✅ Checkliste für dein Paper

- [ ] MOTA Score berechnet
- [ ] Precision/Recall/F1 berechnet
- [ ] Visualisierungen erstellt
- [ ] Mit Baseline verglichen (siehe `example_usage.py`)
- [ ] Threshold-Sensitivität getestet
- [ ] Metriken als JSON exportiert
- [ ] Plots in LaTeX/Paper eingefügt

---

**Viel Erfolg mit deinem Forschungsprojekt! 🎓**
