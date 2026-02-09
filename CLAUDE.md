# Claude Session Notes - MOTRv2 Project

Diese Datei enthält Zusammenfassungen wichtiger Entwicklungen und Kontextinformationen für zukünftige Claude-Sessions.

---

## Session 2026-02-09: GT-Daten Visualisierung

### Problem
- Benutzer wollte Ground Truth (GT) Annotationen für das Volleyball-Dataset visualisieren
- GT-Daten liegen im MOT-Format vor (`finetune/gt/gt/gt.txt`)
- Vorhandenes Visualisierungsskript war für JSON-Detektionen konzipiert, nicht für MOT-Format

### Lösung
Neues Visualisierungsskript erstellt: `tools/visualization/visualize_gt.py`
- Liest MOT-Format GT-Dateien (Frame, Track-ID, x, y, w, h, conf, class)
- Erstellt Video mit farbcodierten Bounding Boxes pro Track-ID
- Zeigt Frame-Nummer und Objekt-Anzahl pro Frame an

### Ausführung
```bash
python3 tools/visualization/visualize_gt.py \
  --images data/Dataset/mot/volleyball/finetune/gt/img1 \
  --gt data/Dataset/mot/volleyball/finetune/gt/gt/gt.txt \
  --output gt_visualization_finetune.mp4 \
  --fps 30
```

### Ergebnis
- **Video:** `gt_visualization_finetune.mp4`
- **Frames:** 18.950 (18.948 mit Annotationen)
- **Bounding Boxes:** 340.075 total
- **Auflösung:** 1920x1080
- **Video-Dauer:** 631,7 Sekunden (~10:32 Minuten bei 30 fps)

### Wichtige Erkenntnisse
- MOT-Format: Komma-separierte Werte (frame, id, x, y, w, h, conf, class)
- Jede Track-ID erhält konsistente Farbe durch Seed-basierte Farbgenerierung
- Video-Generierung für ~19k Frames dauert einige Minuten
- Nützlich für schnelle visuelle Überprüfung von Annotationsqualität

---

## Session 2026-02-09: Git Push Problem - Große Dateien

### Problem
- Git Push zu GitHub schlug fehl mit Fehler: "GH001: Large files detected"
- GitHub lehnte den Push ab wegen Dateien über 100 MB Limit

### Ursache
- 2 große Tracking-Output-Dateien in der Git-Historie:
  - `outputs/inference_v2_aggressive_lr5e6_ep5/tracking_inference.txt` (109 MiB)
  - `outputs/inference_v2_minimal_lr2e5_ep5/tracking_inference.txt` (104 MiB)
- Diese wurden im Commit `c839762` hinzugefügt (insgesamt 8 große tracking_inference.txt Dateien)

### Lösung
1. `outputs/` Verzeichnis zur `.gitignore` hinzugefügt
2. Backup-Branch `backup-before-cleanup` erstellt
3. Letzte 4 Commits zurückgesetzt und ohne große Output-Dateien neu committet
4. Erfolgreich zu GitHub gepusht

### Wichtige Erkenntnisse
- Output-Dateien (besonders Inference-Results) sollten nicht in Git getrackt werden
- `.gitignore` enthält jetzt: `outputs/` zusätzlich zu `*.out`
- Bei zukünftigen großen Dateien: entweder zur .gitignore hinzufügen oder Git LFS verwenden

### Repository-Details
- Remote: `github.com:LeifKa/MOTRv2.git`
- Main Branch: `main`
- Backup verfügbar unter Branch: `backup-before-cleanup`

---

## Session 2026-02-09: Finetuning CUDA-Crash und Checkpoint-Fehler

### Problem
Das Ball-Tracking Finetuning (`volleyball_ball_finetune`) brach wiederholt ab:
1. **Urspruenglicher Crash (Job 59202)**: Training lief 2+ Epochen, dann `CUDA error: unknown error` in `matched_boxlist_iou` (motr.py:241)
2. **ECC-Fehler (Jobs 59246, 59317, 59318, 59319)**: Mehrere Cluster-Knoten (`gpu104`, `gpu109`, `gpu111`) hatten defekte GPUs mit `uncorrectable ECC error` bzw. `No CUDA GPUs available` - Crash bereits bei `torch.cuda.set_device()`
3. **Checkpoint-Fehler (Job 59321)**: Epoch 0 lief komplett durch, aber beim Speichern: `RuntimeError: Parent directory outputs/volleyball_ball_finetune does not exist`

### Ursache
1. **Division durch Null**: `matched_boxlist_iou` in `models/structures/boxes.py` hatte keine Absicherung gegen Zero-Union (im Gegensatz zu `pairwise_iou`)
2. **GPU-Hardware**: Mehrere Knoten in der `gpu1`-Partition haben physische GPU-Speicherfehler (per `nvidia-smi` bestaetigt: `Volatile Uncorr. ECC: 1`)
3. **Fehlender Output-Ordner**: `output_dir` ist ein relativer Pfad (`outputs/volleyball_ball_finetune`), der beim Checkpoint-Speichern nicht existierte

### Loesung / Aenderungen

#### 1. `models/structures/boxes.py` - Division-by-Zero-Fix
```python
# Vorher (unsicher):
iou = inter / (area1 + area2 - inter)

# Nachher (abgesichert wie pairwise_iou):
union = area1 + area2 - inter
iou = torch.where(union > 0, inter / union, torch.zeros(...))
```

#### 2. `tools/fine_tuning/finetune_for_dfine.py` - Output-Dir erstellen
```python
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)  # NEU
```

#### 3. `slurmjobs/run_volleyball_ball_finetune.slurm` - Verbesserungen
- `PYTORCH_CUDA_ALLOC_CONF` -> `PYTORCH_ALLOC_CONF` (Deprecation-Fix)
- GPU-Diagnoseblock hinzugefuegt (`nvidia-smi` + Python CUDA-Check vor Training)

### Wichtige Erkenntnisse
- **ECC-Fehler sind Hardware-Probleme**, kein Code-Bug. Defekte Knoten mit `--exclude=gpu104,gpu109,gpu111` ausschliessen
- **Defekte GPUs erkennen**: `nvidia-smi` zeigt `Volatile Uncorr. ECC` - bei Wert > 0 ist die GPU defekt
- **Cluster-Umgebung**: PyTorch 2.9.0+cu128 auf NVIDIA L40S/H100, Treiber 580.82.07 (CUDA 13.0)
- **`module load devel/cuda/12.4`** ist eigentlich unnoetig, da PyTorch seine eigene CUDA 12.8 Runtime mitbringt
- **Manuelles Training**: Kann ohne Slurm direkt auf einem GPU-Knoten ausgefuehrt werden:
  ```bash
  cd /home/es/es_es/es_lekamt00/BeachKI/MOTRv2
  export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29500
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  python tools/fine_tuning/finetune_for_dfine.py @configs/volleyball_ball_finetune.args --train_strategy moderate
  ```

### Training-Metriken (Referenz)
- Epoch 0 (1830 Iterationen): ~14 Min auf L40S, Loss von ~42 auf ~32 gefallen
- Batch-Geschwindigkeit: ~0.47s/it auf L40S, max GPU-Speicher: 794 MB
- 5 Frames pro Sample, 10 Queries, sample_interval=2

---

## Session 2026-02-09: MOTRv2 Finetuning mit korrigierter GT und Visualisierungen

### Problem
- Nach korrigierter Ground Truth vom Kollegen sollten 6 neue Finetuning-Konfigurationen trainiert werden
- Alte Modelle mit fehlerhafter GT sollten archiviert werden
- Inferenz und Visualisierung für alle Modelle (alt + neu) erforderlich
- D-FINE Detections sollten visualisiert werden können

### Trainings-Konfigurationen
6 verschiedene Kombinationen wurden trainiert:

| Strategie | Learning Rate | Epochen (geplant) | Epochen (tatsächlich) |
|-----------|---------------|-------------------|------------------------|
| Moderate  | 1e-5          | 10                | 6                      |
| Moderate  | 1e-5          | 5                 | 5                      |
| Moderate  | 5e-6          | 10                | 6                      |
| Aggressive| 1e-5          | 10                | 6                      |
| Aggressive| 1e-5          | 5                 | 5                      |
| Aggressive| 5e-6          | 10                | 6                      |

**Ground Truth Details:**
- Quelle: `Datasets/Volleyball-Activity-Dataset-3/gt.txt` (vom Kollegen)
- Ursprünglich: 446,332 Annotationen
- Nach Filterung: 340,075 Annotationen über 18,948 Frames
- Format: MOT Challenge mit echten Track IDs

### Lösung und Änderungen

#### 1. Archivierung alter Modelle
Alle alten Trainings mit fehlerhafter GT wurden mit `_old_gt` Suffix umbenannt:
- `outputs/finetune_volleyball_combined_old_gt/`
- `outputs/finetune_vb_moderate_lr1e5_ep5_old_gt/`
- `outputs/finetune_vb_moderate_lr5e6_ep10_old_gt/`
- `outputs/finetune_vb_aggressive_lr1e5_ep10_old_gt/`
- `outputs/finetune_vb_aggressive_lr1e5_ep5_old_gt/`
- `outputs/finetune_vb_aggressive_lr5e6_ep10_old_gt/`

Auch Detection DB und gt.txt wurden archiviert:
- `det_db_volleyball_combined_old.json`
- `gt_old.txt`

#### 2. Neue Ground Truth Integration
Neue GT-Datei von Kollege verarbeitet:
```bash
# Filtern der GT auf verfügbare Frames
python3 << 'EOF'
gt_data = []
with open('Datasets/Volleyball-Activity-Dataset-3/gt.txt', 'r') as f:
    for line in f:
        frame_num = int(line.split(',')[0])
        if 1 <= frame_num <= 18948:
            gt_data.append(line)

with open('MOTRv2/data/Dataset/mot/volleyball/combined/gt/gt/gt.txt', 'w') as f:
    f.writelines(gt_data)
EOF
```

Neue Detection Database erstellt mit korrigierten Frame-Nummern.

#### 3. Training-Durchführung
6 SLURM Jobs gestartet (Jobs 53583-53588):
- Alle mit `--exclude=gpu104` (bekannte ECC-Fehler)
- 48h Zeit-Limit
- Verwendete config files und slurm scripts für jede Konfiguration

**Wichtiger Fehler entdeckt:**
Alle 4 "10-Epochen" Trainings liefen nur 6 Epochen (checkpoint0000-0005), dann Crash:
```
Exception raised from run at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:2063
/var/spool/slurmd/job53583/slurm_script: line 65: 2377913 Aborted (core dumped)
```

Ursache: NCCL (distributed training) Fehler, Job wurde als COMPLETED markiert trotz Crash.

#### 4. Visualisierungs-Skript für D-FINE Detections
Neues Skript erstellt: `tools/visualization/visualize_detections.py`

Features:
- Liest Detection Database JSON
- Zeigt Bounding Boxes mit Confidence Scores
- Farbkodierung nach Score:
  - Grün: Score > 0.7
  - Gelb: Score > 0.5
  - Orange: Score < 0.5
- Unterstützt Score-Threshold-Filter

**Wichtiger Bug-Fix:**
Debug-Ausgaben hinzugefügt, um fehlende Detections zu erkennen:
- Prüfung ob Detection DB leer ist
- Anzeige der ersten 5 Frames mit/ohne Detections

### Wichtige Erkenntnisse

#### 1. GPU-Hardware-Probleme
**Betroffene GPUs mit ECC-Fehlern:**
- `gpu104` - bereits bekannt
- `gpu126` - neu entdeckt (während Inferenz)

**Symptome:**
```
torch.AcceleratorError: CUDA error: uncorrectable ECC error encountered
```

**Lösung:** Beide GPUs in allen zukünftigen Jobs excluden:
```bash
#SBATCH --exclude=gpu104,gpu126
# oder für interaktive Sessions:
srun --exclude=gpu104,gpu126 --pty bash
```

#### 2. Detection Database für verschiedene Datasets
**Wichtige Unterscheidung:**
- **Training:** `det_db_volleyball_combined.json`
  - Keys: `volleyball/combined/gt/img1/XXXXXX`
  - 18,948 Frames
- **Test/Inferenz:** `det_db_beach_volleyball.json`
  - Keys: `volleyball/test/test1/img1/XXXXXX`
  - 956 Frames
- **Alte Finetune GT (falsch):** `inputs/finetune_gt_dfine.json`
  - Nur Ground Truth, keine echten Detections!
  - Nicht für Visualisierung verwenden

#### 3. NCCL Distributed Training Issues
- Trainings können nach mehreren Epochen mit NCCL-Fehler abstürzen
- SLURM markiert Job trotzdem als COMPLETED
- Lösung: Immer Checkpoints prüfen, nicht nur Job-Status

#### 4. Checkpoint-Nomenklatur
- `checkpoint.pth` = letzter Checkpoint
- `checkpoint0000.pth` = Epoch 0
- `checkpoint0004.pth` = Epoch 4 (5. Epoche, 0-indexed)
- Bei "10 Epochen" erwartet: checkpoint0009.pth

### Inferenz und Visualisierung

#### Inferenz-Commands (Neue GT)
Alle 6 Modelle mit korrigierter GT:

```bash
# Beispiel: Moderate LR=1e-5, Epochs=5
python submit_dance.py \
    --meta_arch motr \
    --dataset_file e2e_dance \
    --with_box_refine \
    --query_interaction_layer QIMv2 \
    --num_queries 10 \
    --sampler_lengths 2 \
    --sample_mode random_interval \
    --sample_interval 30 \
    --use_checkpoint \
    --resume ./outputs/finetune_vb_moderate_lr1e5_ep5/checkpoint0004.pth \
    --det_db det_db_beach_volleyball.json \
    --mot_path ./data/Dataset/mot \
    --output_dir outputs/inference_moderate_lr1e5_ep5 \
    --score_threshold 0.3 \
    --miss_tolerance 20
```

#### Visualisierung (Tracking Results)
```bash
python tools/visualization/visualize_tracking.py \
    --images ./data/Dataset/mot/volleyball/test/test1/img1 \
    --tracking outputs/inference_moderate_lr1e5_ep5/tracking_inference.txt \
    --output outputs/inference_moderate_lr1e5_ep5/visualization.mp4 \
    --fps 30
```

#### Visualisierung (D-FINE Detections)
```bash
python tools/visualization/visualize_detections.py \
    --images ./data/Dataset/mot/volleyball/test/test1/img1 \
    --det_db data/Dataset/mot/det_db_beach_volleyball.json \
    --video_path volleyball/test/test1 \
    --output outputs/detections_visualization.mp4 \
    --fps 30
```

### Dateistruktur

#### Neue Dateien
- `tools/visualization/visualize_detections.py` - D-FINE Detection Visualisierung
- `data/Dataset/mot/volleyball/combined/gt/gt/gt.txt` - Korrigierte GT (340,075 Annotationen)
- `data/Dataset/mot/det_db_volleyball_combined.json` - Neue Detection DB

#### Archivierte Dateien
- `data/Dataset/mot/volleyball/combined/gt/gt/gt_old.txt` - Alte fehlerhafte GT
- `data/Dataset/mot/det_db_volleyball_combined_old.json` - Alte Detection DB
- `outputs/*_old_gt/` - 6 alte Trainings mit fehlerhafter GT

#### Config Files (unverändert)
- `configs/volleyball_combined_finetune.args` - moderate, lr=1e-5, ep=10
- `configs/volleyball_combined_moderate_lr1e5_ep5.args`
- `configs/volleyball_combined_moderate_lr5e6_ep10.args`
- `configs/volleyball_combined_aggressive_lr1e5_ep10.args`
- `configs/volleyball_combined_aggressive_lr1e5_ep5.args`
- `configs/volleyball_combined_aggressive_lr5e6_ep10.args`

### Nächste Schritte

1. **Trainings analysieren:** Warum NCCL-Crash nach 6 Epochen?
2. **10-Epochen Jobs neu starten:** Möglicherweise mit Single-GPU statt distributed
3. **Alle Inferenzen laufen lassen:** 6 neue + 6 alte Modelle
4. **Visualisierungen vergleichen:** Qualitative Bewertung der Tracking-Ergebnisse
5. **Metriken berechnen:** HOTA, MOTA, IDF1 für quantitative Evaluation

### Referenz-Befehle

**Job Status prüfen:**
```bash
squeue -u es_lekamt00
sacct -j 53583,53584,53585,53586,53587,53588 --format=JobID,JobName,State,ExitCode,Elapsed
```

**Checkpoint-Files prüfen:**
```bash
ls -lh outputs/finetune_vb_*/checkpoint*.pth
```

**Interaktive GPU-Session (mit GPU-Exclude):**
```bash
srun --partition=gpu1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 \
     --exclude=gpu104,gpu126 --time=2:0:0 --pty bash
```

---

## Session 2026-01-13/15: MOTRv2 Fine-Tuning v2 - Korrektur und Systematisierung

### Problem
Nach mehreren Fine-Tuning-Versuchen mit Volleyball-Daten zeigte sich, dass das Tracking **überhaupt nicht funktionierte**:
- **Symptome:** Über 500 Tracks pro Video, statische Bounding Boxes, "Bogen"-Effekt entlang Trajektorien
- **Verhalten:** In fast jedem Frame wurde eine neue Query für Objekte erstellt, die sich nicht bewegte
- **Root Cause:** Die Detection Database (`det_db_volleyball_combined.json`) enthielt **identische Boxes wie die Ground Truth** mit score=1.0 statt echte Detector-Outputs

### Ursachenanalyse

#### Problem 1: Fehlerhafte Detection Database
Die alte Detection DB wurde aus einem JSON-Format ohne Track IDs konvertiert:
- **Alt (fehlerhaft):** GT Boxes wurden 1:1 in Detection DB kopiert (score=1.0)
- **Folge:** Modell lernte nicht, mit realistischen Detections (False Positives, variierende Scores) umzugehen
- **Beweis:** Frame 3 GT und Detection DB hatten **exakt identische Boxes**

#### Problem 2: Falsche Ground Truth im alten Training
Die alte `gt_old.txt` hatte **fortlaufende Track IDs** statt echte Tracks:
- Frame 1: IDs 1-17
- Frame 2: IDs 18-38 (komplett neue IDs!)
- Frame 3: IDs 39-57 (komplett neue IDs!)
- **Kein Objekt wurde über Frames verfolgt**

#### Problem 3: Fehlende Parameter-Freeze-Logik
Die Trainings-Strategien "moderate" und "aggressive" trainierten beide nur ~3% der Parameter, obwohl sie unterschiedliche Layer trainieren sollten.

### Lösung

#### 1. Neue Ground Truth verwenden
- **Datei:** `Datasets/Volleyball-Activity-Dataset-3/gt.txt` (446,332 Detections, 7,862 echte Tracks)
- **Format:** MOT Challenge Format mit echten Track IDs über Zeit
- **Umbenennung:** "combined" → "finetune" im gesamten Projekt für Klarheit

#### 2. D-FINE Detection Database erstellen
Neues Skript: `D-FINE/tools/inference/create_detection_db_for_finetune.py`
- Läuft D-FINE auf allen 18,950 Bildern des Trainings-Datasets
- Erstellt `det_db_volleyball_finetune_dfine.json` mit **echten Detections**:
  - Scores: 0.3-0.9 (realistisch)
  - 15-25 Detections pro Frame
  - Enthält False Positives und unsichere Detections

#### 3. Training-Strategy zu main.py hinzugefügt
**Code-Änderungen in `main.py`:**

```python
# Neues Argument
parser.add_argument('--train_strategy',
    choices=['minimal', 'moderate', 'aggressive', 'full'])

# Layer-Freeze nach Pretrained-Loading
strategies = {
    'minimal': ['yolox_embed'],                              # ~1-2%
    'moderate': ['yolox_embed', 'track_embed', 'class_embed'], # ~5-10%
    'aggressive': [..., 'query_interaction'],                # ~15-20%
}

# DDP Fix für eingefrorene Parameter
find_unused = args.train_strategy is not None and args.train_strategy != 'full'
model = DistributedDataParallel(model, find_unused_parameters=find_unused)
```

#### 4. Systematische Trainings-Matrix
9 Konfigurationen erstellt (Strategie × Learning Rate):

| Strategy | LR 5e-6 | LR 1e-5 | LR 2e-5 |
|----------|---------|---------|---------|
| **Minimal** (1-2%) | ✓ | ✓ | ✓ |
| **Moderate** (5-10%) | ✓ | ✓ | ✓ |
| **Aggressive** (15-20%) | ✓ | ✓ | ✓ |

Alle mit 5 Epochen, jeweils eigene Configs und SLURM-Skripte.

### Wichtige Änderungen

#### Dateien erstellt/geändert:
1. **`main.py`:**
   - `--train_strategy` Argument hinzugefügt
   - Parameter-Freeze-Logik implementiert
   - DDP `find_unused_parameters=True` für eingefrorene Layer
   - Custom Config-Parser für `@config.args` Syntax

2. **`D-FINE/tools/inference/create_detection_db_for_finetune.py`:**
   - Erstellt echte Detection DB mit D-FINE für Training

3. **Configs:** 9 neue `.args` Dateien:
   - `volleyball_finetune_{minimal,moderate,aggressive}_lr{5e6,1e5,2e5}_ep5.args`

4. **SLURM-Skripte:**
   - 9× Training: `run_volleyball_finetune_*_ep5.slurm`
   - 1× Inferenz: `run_inference_v2.slurm`
   - 1× Visualisierung: `run_visualization_v2.slurm`

5. **Helper-Skripte:**
   - `run_all_inferences_v2.sh` - Inferenz & Vis für alle 9 Modelle
   - `FINETUNE_CONFIGURATIONS.md` - Komplette Dokumentation

#### Ordner-Reorganisation:
```
outputs/
├── finetune_v1/          # Alte fehlerhafte Trainings verschoben
│   ├── finetune_volleyball_combined/
│   ├── *_OLD_BAD_GT/
│   └── inference_*/
└── finetune_vb_*_ep5/    # Neue v2 Trainings
    ├── minimal_lr{5e6,1e5,2e5}_ep5/
    ├── moderate_lr{5e6,1e5,2e5}_ep5/
    └── aggressive_lr{5e6,1e5,2e5}_ep5/
```

### Training-Ergebnisse

**Alle 9 Jobs erfolgreich abgeschlossen:**
- Jedes Training: 5 Epochen, ~18,943 Batches pro Epoch
- Output: 6 Checkpoints pro Modell (checkpoint.pth + checkpoint0000-0004.pth)
- Trainierte Parameter variieren korrekt nach Strategie:
  - Minimal: ~1,300,000 (2-3%)
  - Moderate: ~2,500,000 (5-10%)
  - Aggressive: ~6,000,000 (15-20%)

### Inferenz & Visualisierung

**Inferenz-Parameter (verwendet):**
```bash
--num_queries 10
--score_threshold 0.3
--miss_tolerance 20
--sampler_lengths 2
--sample_interval 30
```

**Detection DB für Inferenz:**
- Training: `det_db_volleyball_finetune_dfine.json` (volleyball/finetune/gt/)
- **Inferenz:** `det_db_beach_volleyball.json` (volleyball/test/test1/) ← Wichtig!

**Parameter-Empfehlungen:**
- `score_threshold`: Teste 0.2, 0.3, 0.4, 0.5 (Wichtigster Parameter!)
- `miss_tolerance`: 15-25 für schnelle Volleyball-Bewegungen
- `num_queries`: 10-15 ausreichend für Volleyball
- `sampler_lengths`: Teste 2, 3, 5 für mehr Kontext

### Wichtige Erkenntnisse

#### 1. Detection Database ist kritisch
- **Training DB ≠ Inferenz DB!** Keys müssen zum Dataset passen
- Ohne echte Detector-Outputs lernt MOTRv2 nicht zu tracken
- D-FINE Detections mit variierenden Scores sind essentiell

#### 2. Ground Truth Format
- **MOT Format ist Pflicht:** `frame,track_id,x,y,w,h,conf,class`
- Track IDs müssen über Frames konsistent sein
- Fortlaufende IDs (1,2,3...) pro Frame sind falsch!

#### 3. Training-Strategien funktionieren jetzt
```
Minimal:     1.3M params (2-3%)   - yolox_embed only
Moderate:    2.5M params (5-10%)  - + track_embed, class_embed
Aggressive:  6.0M params (15-20%) - + query_interaction
```

#### 4. DDP mit eingefrorenen Parametern
- `find_unused_parameters=True` ist **zwingend erforderlich**
- Ohne dies crasht Training mit "Expected to have finished reduction"

#### 5. Systematisches Experimentieren
- 9 Kombinationen (3 Strategien × 3 LRs) geben guten Überblick
- Erwartung: **moderate_lr1e5** oder **aggressive_lr1e5** beste Ergebnisse

### Nächste Schritte

1. **Inferenz-Ergebnisse evaluieren:** Vergleiche alle 9 Videos
2. **Metriken berechnen:** HOTA, MOTA, IDF1 auf Test-Set
3. **Parameter-Tuning:** score_threshold und miss_tolerance variieren
4. **Best Model auswählen:** Basierend auf visuellen und quantitativen Ergebnissen

### Referenz-Commands

**Training:**
```bash
sbatch run_volleyball_finetune_moderate_lr1e5_ep5.slurm
```

**Inferenz (alle 9 Modelle):**
```bash
sbatch run_inference_v2.slurm
```

**Visualisierung:**
```bash
sbatch run_visualization_v2.slurm
# oder
bash run_all_inferences_v2.sh visualization
```

**Einzelne Inferenz:**
```bash
python submit_dance.py \
    --resume ./outputs/finetune_vb_moderate_lr1e5_ep5/checkpoint0004.pth \
    --det_db det_db_beach_volleyball.json \
    --output_dir outputs/inference_v2_moderate_lr1e5_ep5 \
    --score_threshold 0.3 \
    --miss_tolerance 20
```

---

## Anleitung zum Aktualisieren dieser Datei

Um diese Datei am Ende einer Session zu aktualisieren, verwende folgenden Befehl:

```
Bitte fasse die wichtigsten Erkenntnisse und Änderungen aus dieser Session zusammen und füge sie zu CLAUDE.md hinzu. Füge das neue Datum als Überschrift hinzu und beschreibe: Problem, Lösung, wichtige Änderungen und Erkenntnisse.
```

