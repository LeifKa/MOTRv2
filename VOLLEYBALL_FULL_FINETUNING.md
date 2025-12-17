# MOTRv2 Multi-Class Fine-Tuning für Volleyball

## Übersicht

Dieses Dokument beschreibt das **partielle Fine-Tuning** von MOTRv2 auf dem Volleyball-Activity-Dataset-3 mit 8 Klassen. Im Gegensatz zum vorherigen Full-Fine-Tuning (das fehlgeschlagen ist) werden hier nur spezifische Layer trainiert, um **Catastrophic Forgetting** zu vermeiden.

## 🔴 Was war das Problem beim vorherigen Training?

Das vorherige Training schlug fehl, weil:

1. **Falsche Anzahl der Klassen**: Das Modell wurde mit `num_classes=1` (e2e_dance) trainiert, aber Volleyball hat 8 Klassen
2. **Fehlende Klassen-IDs in Detection Database**: Die Detection Database enthielt keine Klassen-Informationen
3. **Full Fine-Tuning**: ALLE Parameter wurden trainiert, was zu Catastrophic Forgetting führte
4. **Label-Filterung**: Nicht-Person-Labels wurden herausgefiltert

## ✅ Lösungen (bereits implementiert)

- ✅ `e2e_volleyball` mit `num_classes=8` in [models/motr.py:717](models/motr.py#L717) hinzugefügt
- ✅ Detection Database mit Klassen-IDs erweitert
- ✅ Label-Filterung für Volleyball deaktiviert in [datasets/dance.py](datasets/dance.py#L61-L65)
- ✅ Partielle Fine-Tuning-Strategien implementiert

## 📋 Voraussetzungen

- ✅ Volleyball-Datensatz in `Datasets/Volleyball-Activity-Dataset-3/` (COCO-Format)
- ✅ MOTRv2 in `MOTRv2/`
- ✅ Datensatz in MOT-Format konvertiert (`MOTRv2/data/Dataset/mot/volleyball_full/`)
- ✅ Pretrained Weights (`MOTRv2/weights/motrv2_dancetrack.pth`)
- ✅ Code-Anpassungen (siehe oben)

## 🎯 Training-Strategien

### Option 1: Minimal (empfohlen für Start)
**Layer**: `yolox_embed` only
**Vorteil**: Sicherste Methode, minimales Risiko von Forgetting
**Nachteil**: Begrenzte Anpassung an Multi-Class
**Verwenden wenn**: Du zuerst testen möchtest, ob das Training funktioniert

### Option 2: Moderate (⭐ empfohlen für Volleyball)
**Layer**: `yolox_embed` + `track_embed` + `class_embed`
**Vorteil**: Perfekt für Multi-Class, trainiert den Classifier für 8 Klassen
**Nachteil**: Etwas mehr Trainingszeit
**Verwenden wenn**: Du Multi-Class Tracking mit allen 8 Volleyball-Klassen willst

### Option 3: Aggressive
**Layer**: `yolox_embed` + `track_embed` + `class_embed` + `query_interaction`
**Vorteil**: Maximale Anpassung an Volleyball
**Nachteil**: Höheres Risiko von Overfitting
**Verwenden wenn**: Moderate-Strategie nicht gut genug performt

## 🔧 Setup-Schritte

### Schritt 1: Detection Database mit Klassen-IDs erstellen

Die Detection Database muss neu erstellt werden mit Klassen-IDs:

```bash
cd /home/es/es_es/es_lekamt00/BeachKI
conda activate pytorch
python MOTRv2/create_det_db_volleyball_full.py
```

**Erwartete Ausgabe:**
```
=== Processing Training Sequences ===
Processing sequence: train/seq1
  Added 17495 frames with detections

=== Processing Validation Sequences ===
Processing sequence: valid/seq1
  Added 4998 frames with detections

=== Saving Detection Database ===
Total entries: 22493
Output file: MOTRv2/data/Dataset/mot/det_db_volleyball_full_dfine.json
```

**Wichtig**: Überprüfe, dass die Detection Database jetzt Klassen-IDs enthält:
```json
{
  "volleyball_full/train/seq1/img1/000001": [
    "1141.00,421.00,192.00,350.00,1,1.000000\n"
                                    ^ Klassen-ID!
  ]
}
```

### Schritt 2: Trainings-Konfiguration überprüfen

Die Config in `MOTRv2/configs/volleyball_full_finetune.args` sollte sein:

```
--meta_arch motr
--dataset_file e2e_volleyball        ← WICHTIG: e2e_volleyball, nicht e2e_dance!
--epoch 10
--with_box_refine
--lr_drop 8
--lr 1e-5
--lr_backbone 1e-6
--pretrained ./weights/motrv2_dancetrack.pth
--batch_size 1
--det_db det_db_volleyball_full_dfine.json
--data_txt_path_train ./datasets/data_path/volleyball_full_train.txt
--output_dir outputs/finetune_volleyball_moderate
--exp_name volleyball_moderate_finetune
```

### Schritt 3: Fine-Tuning starten

#### Option A: Mit SLURM (empfohlen)

Erstelle ein neues SLURM-Skript `run_volleyball_moderate_finetune.slurm`:

```bash
#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:0:0
#SBATCH --output=volleyball_moderate_finetune_%j.out
#SBATCH --job-name=motrv2_vb_moderate

# Load modules
module load devel/miniforge/24.11.0-python-3.12
module load devel/cuda/12.4

# Set environment
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# CUDA optimization
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

# Activate conda
eval "$(conda shell.bash hook)"
conda activate pytorch

# Print job info
echo "========================================="
echo "MOTRv2 Moderate Fine-Tuning (Volleyball)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Go to MOTRv2 directory
cd /home/es/es_es/es_lekamt00/BeachKI/MOTRv2

# Run fine-tuning with MODERATE strategy
python tools/fine_tuning/finetune_for_dfine.py \
    $(cat configs/volleyball_full_finetune.args) \
    --train_strategy moderate \
    --resume ./weights/motrv2_dancetrack.pth

echo ""
echo "========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "========================================="
```

**Training starten:**
```bash
cd /home/es/es_es/es_lekamt00/BeachKI
sbatch run_volleyball_moderate_finetune.slurm
```

#### Option B: Interaktiv (zum Testen)

```bash
cd /home/es/es_es/es_lekamt00/BeachKI/MOTRv2
conda activate pytorch

python tools/fine_tuning/finetune_for_dfine.py \
    $(cat configs/volleyball_full_finetune.args) \
    --train_strategy moderate \
    --resume ./weights/motrv2_dancetrack.pth
```

**Andere Strategien:**
```bash
# Minimal (nur yolox_embed)
--train_strategy minimal

# Moderate (empfohlen für Multi-Class)
--train_strategy moderate

# Aggressive (mehr Layer)
--train_strategy aggressive
```

### Schritt 4: Training überwachen

#### Job-Status prüfen
```bash
squeue -u $USER
```

#### Live-Training-Log ansehen
```bash
tail -f volleyball_moderate_finetune_<JOB_ID>.out
```

#### Was du sehen solltest:

```
================================================================================
VOLLEYBALL MULTI-CLASS FINE-TUNING
================================================================================

Configuration:
  Dataset: e2e_volleyball
  Detection DB: det_db_volleyball_full_dfine.json
  Resume from: ./weights/motrv2_dancetrack.pth
  Learning rate: 1e-05
  Epochs: 10
  Training strategy: moderate
================================================================================

================================================================================
FINE-TUNING STRATEGY: yolox_embed, track_embed, class_embed
================================================================================

✓ TRAINING: track_embed.track_query_linear.weight
✓ TRAINING: track_embed.track_query_linear.bias
✓ TRAINING: yolox_embed.embed.0.weight
✓ TRAINING: yolox_embed.embed.0.bias
✓ TRAINING: yolox_embed.embed.2.weight
✓ TRAINING: yolox_embed.embed.2.bias
✓ TRAINING: class_embed.weight
✓ TRAINING: class_embed.bias

================================================================================
Parameter Summary:
  Trainable:         xxx,xxx (xx.xx%)
  Frozen:          xxx,xxx,xxx (xx.xx%)
  Total:           xxx,xxx,xxx
================================================================================
```

## 📊 Erwartete Ergebnisse

Nach dem Training:
- Checkpoints in: `MOTRv2/outputs/finetune_volleyball_moderate/`
- Modell kann 8 Volleyball-Klassen tracken:
  1. Volleyball-Players
  2. Defense-Move
  3. Attack
  4. Block
  5. Reception
  6. Service
  7. Setting
  8. Stand

## 🧪 Testing

### Checkpoint testen

```bash
cd /home/es/es_es/es_lekamt00/BeachKI/MOTRv2

python submit_dance.py \
    --resume outputs/finetune_volleyball_moderate/checkpoint.pth \
    --data_txt_path datasets/data_path/volleyball_full_valid.txt \
    --result_dir results/volleyball_moderate
```

### Frühere Checkpoints testen

Falls das finale Modell nicht gut funktioniert, teste frühere Epochen:

```bash
# Epoch 1
python submit_dance.py --resume outputs/finetune_volleyball_moderate/checkpoint0001.pth

# Epoch 2
python submit_dance.py --resume outputs/finetune_volleyball_moderate/checkpoint0002.pth

# Epoch 5
python submit_dance.py --resume outputs/finetune_volleyball_moderate/checkpoint0005.pth
```

## 🔍 Vergleich: Altes vs. Neues Training

### Vorheriges Training (fehlgeschlagen)
| Aspekt | Wert |
|--------|------|
| Skript | `main.py` (Full Training) |
| Trainierte Layer | ALLE Parameter |
| Dataset-Type | `e2e_dance` (num_classes=1) |
| Detection DB | Ohne Klassen-IDs |
| Ergebnis | ❌ Nichts getrackt |

### Neues Training (korrigiert)
| Aspekt | Wert |
|--------|------|
| Skript | `tools/fine_tuning/finetune_for_dfine.py` |
| Trainierte Layer | Nur `yolox_embed`, `track_embed`, `class_embed` |
| Dataset-Type | `e2e_volleyball` (num_classes=8) |
| Detection DB | Mit Klassen-IDs |
| Ergebnis | ✅ Sollte funktionieren |

## 📈 Layer-Erklärung

### Was macht jeder Layer?

**yolox_embed**:
- Konvertiert Detektionen in Feature-Embeddings
- Muss trainiert werden, um neue Detection-Charakteristiken zu lernen

**track_embed**:
- Erzeugt Track-Embeddings für besseres Tracking
- Wichtig für Unterscheidung verschiedener Objekte

**class_embed** (KRITISCH für Multi-Class!):
- Classifier-Layer für 8 Klassen
- MUSS trainiert werden für Multi-Class Tracking
- Output-Shape: [hidden_dim, 8]

**query_interaction**:
- Query-to-Query Interaktion für Track-Updates
- Optional für aggressive Fine-Tuning-Strategie

## 🐛 Troubleshooting

### Problem: "Missing keys" beim Checkpoint-Laden
```
⚠️ Missing keys: 11
   - class_embed.weight
   - class_embed.bias
```

**Lösung**: Das ist normal! Der pretrained Checkpoint hat `num_classes=1`, aber wir brauchen `num_classes=8`. Die fehlenden Keys werden zufällig initialisiert.

### Problem: Training konvergiert nicht
**Lösungen**:
1. Versuche `--train_strategy minimal` (nur yolox_embed)
2. Reduziere Learning Rate: `--lr 5e-6`
3. Erhöhe Epochen: `--epoch 15`

### Problem: Out of Memory (OOM)
**Lösungen**:
1. Batch-Size ist bereits 1 (minimum)
2. Reduziere `--num_queries` von 10 auf 5
3. Fordere GPU mit mehr VRAM an

### Problem: Modell trackt immer noch nichts
**Debug-Schritte**:
1. Überprüfe Detection Database: Sind Klassen-IDs vorhanden?
   ```bash
   head -20 MOTRv2/data/Dataset/mot/det_db_volleyball_full_dfine.json
   ```
2. Teste früheren Checkpoint (Epoch 1-3)
3. Versuche andere Strategie (minimal → moderate → aggressive)

## 📝 Nächste Schritte

Nach erfolgreichem Training:

1. **Evaluation**: Teste auf Test-Set und messe HOTA/IDF1/MOTA
2. **Visualisierung**: Erstelle Videos mit Tracking-Ergebnissen
3. **Weitere Optimierung**: Falls nötig, experimentiere mit:
   - Höherer Learning Rate (1e-4)
   - Mehr Epochen (20-30)
   - Aggressivere Strategie
   - Data Augmentation

## 🎓 Wichtige Erkenntnisse

1. **Partial > Full Fine-Tuning**: Trainiere nur notwendige Layer, nicht das gesamte Modell
2. **class_embed ist kritisch**: Für Multi-Class MUSS der Classifier trainiert werden
3. **Klassen-IDs sind essentiell**: Detection Database muss Klassen-Informationen enthalten
4. **Frühe Checkpoints testen**: Manchmal sind frühere Epochen besser (vor Overfitting)

---

**Erstellt**: 2025-12-17
**Autor**: Claude Code
**Version**: 2.0 (Komplett überarbeitet nach Fix der Probleme)