# MOTRv2 Full Finetuning auf Volleyball-Datensatz

## Übersicht

Dieses Dokument beschreibt die Schritte zum vollständigen Finetuning von MOTRv2 auf dem Volleyball-Activity-Dataset-3.

## Voraussetzungen

- ✅ Volleyball-Datensatz in `Datasets/Volleyball-Activity-Dataset-3/` (COCO-Format)
- ✅ MOTRv2 in `MOTRv2/`
- ✅ Datensatz bereits in MOT-Format konvertiert (`MOTRv2/data/Dataset/mot/volleyball_full/`)
- ✅ Pretrained Weights vorhanden (`MOTRv2/weights/motrv2_dancetrack.pth`)
- ✅ Training-Konfiguration vorhanden (`MOTRv2/configs/volleyball_full_finetune.args`)

## Setup-Schritte

### 1. Detection Database erstellen

Die Detection Database muss aus den Ground Truth Annotationen erstellt werden:

```bash
cd /home/es/es_es/es_lekamt00/BeachKI
conda activate pytorch
python create_det_db_volleyball_full.py
```

**Erwartete Ausgabe:**
- Datei erstellt: `MOTRv2/data/Dataset/mot/det_db_volleyball_full_dfine.json`
- ~17.495 Frames für Training
- ~5.000 Frames für Validation

### 2. Konfiguration überprüfen

Die Trainings-Konfiguration ist in `MOTRv2/configs/volleyball_full_finetune.args`:

```
--meta_arch motr
--dataset_file e2e_dance
--epoch 10
--with_box_refine
--lr_drop 8
--lr 1e-5
--lr_backbone 1e-6
--pretrained ./weights/motrv2_dancetrack.pth
--batch_size 1
--det_db det_db_volleyball_full_dfine.json
--data_txt_path_train ./datasets/data_path/volleyball_full_train.txt
--output_dir outputs/finetune_volleyball_full
--exp_name volleyball_full_dfine_finetune
```

**Wichtige Parameter:**
- **Learning Rate**: 1e-5 (sehr niedrig für Finetuning des gesamten Modells)
- **Epochen**: 10
- **Pretrained Model**: motrv2_dancetrack.pth
- **Output**: `outputs/finetune_volleyball_full/`

### 3. Training starten

#### Option A: Mit SLURM (empfohlen)

```bash
cd /home/es/es_es/es_lekamt00/BeachKI
sbatch run_volleyball_full_finetune.slurm
```

**Job-Details:**
- Partition: gpu8
- GPU: 1x GPU
- Zeit: 24 Stunden
- Output Log: `volleyball_full_finetune_<JOB_ID>.out`

#### Option B: Interaktiv (zum Testen)

```bash
cd /home/es/es_es/es_lekamt00/BeachKI/MOTRv2
conda activate pytorch

python main.py \
    --resume_train_from_checkpoint \
    $(cat configs/volleyball_full_finetune.args)
```

### 4. Training überwachen

#### Job-Status prüfen

```bash
# Aktuelle Jobs anzeigen
squeue -u $USER

# Ausgabe-Beispiel:
# JOB_ID  PARTITION  NAME           USER  ST  TIME  NODES
# 12345   gpu8       motrv2_vb_ft   user  R   2:30  1
```

**Status-Codes:**
- `PD` = Pending (wartet auf Ressourcen)
- `R` = Running (läuft)
- `CG` = Completing (beendet sich)
- `CD` = Completed (abgeschlossen)

#### Live-Training-Log ansehen

```bash
# Live-Log mit letzten 100 Zeilen
tail -n 100 -f volleyball_full_finetune_<JOB_ID>.out

# Zum Beenden: Strg+C
```

#### Detaillierte Job-Informationen

```bash
# Aktuelle Job-Details
scontrol show job <JOB_ID>

# Job-Historie mit Ressourcen-Verbrauch
sacct -j <JOB_ID> --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize

# Alle Jobs der letzten 24h
sacct -u $USER --starttime=$(date -d '24 hours ago' +%Y-%m-%d)
```

#### Job abbrechen (falls nötig)

```bash
scancel <JOB_ID>
```

#### TensorBoard (falls aktiviert)

```bash
tensorboard --logdir=MOTRv2/outputs/finetune_volleyball_full
```

### 5. Wichtig: SLURM Jobs laufen unabhängig!

**Du kannst während des Trainings:**
- ✅ VS Code schließen und neustarten
- ✅ Plugins updaten
- ✅ Deinen PC ausschalten
- ✅ Die SSH-Verbindung trennen

**Der SLURM Job läuft auf dem Cluster-Node weiter!**

Um später den Status zu prüfen, einfach wieder einloggen und `squeue -u $USER` ausführen.

## Datensatz-Informationen

### Volleyball-Activity-Dataset-3

- **Quelle**: Roboflow
- **Format**: COCO
- **Auflösung**: 1920x1080
- **Training-Bilder**: 17.495
- **Validierungs-Bilder**: ~5.000
- **Test-Bilder**: ~2.500

### Klassen (8 Kategorien)

0. Volleyball-Players (Hauptkategorie)
1. Defense-Move
2. attack
3. block
4. reception
5. service
6. setting
7. stand

### MOT-Format Struktur

```
MOTRv2/data/Dataset/mot/volleyball_full/
├── train/
│   └── seq1/
│       ├── img1/          # 17.495 Bilder (000001.jpg - 017495.jpg)
│       └── gt/
│           └── gt.txt     # Ground Truth Annotationen
└── valid/
    └── seq1/
        ├── img1/          # ~5.000 Bilder
        └── gt/
            └── gt.txt
```

## Unterschied zu vorherigem Training

### Vorheriges Training (yolox_embed)
- **Was wurde trainiert**: Nur der `yolox_embed` Parameter
- **Learning Rate**: Höher
- **Verwendung**: Embedding-Anpassung

### Aktuelles Training (Full Finetuning)
- **Was wird trainiert**: Gesamtes MOTRv2 Modell
- **Learning Rate**: 1e-5 (sehr niedrig)
- **Backbone LR**: 1e-6 (noch niedriger)
- **Verwendung**: Vollständige Modellanpassung an Volleyball-Domain

## Erwartete Ergebnisse

Nach dem Training werden die Checkpoints gespeichert in:
- `MOTRv2/outputs/finetune_volleyball_full/`

Die Checkpoints können dann verwendet werden für:
1. Tracking auf neuen Volleyball-Videos
2. Weitere Evaluation auf Test-Set
3. Transfer Learning auf ähnliche Sportarten

## Troubleshooting

### Problem: Detection DB nicht gefunden
```
Solution: Stelle sicher, dass create_det_db_volleyball_full.py erfolgreich ausgeführt wurde
```

### Problem: Out of Memory (OOM)
```
Solution: Batch-Size in der Config reduzieren (derzeit: 1)
Oder: Größere GPU anfordern in SLURM-Skript
```

### Problem: Training konvergiert nicht
```
Solution 1: Learning Rate anpassen (configs/volleyball_full_finetune.args)
Solution 2: Mehr Epochen trainieren (--epoch erhöhen)
Solution 3: Warmup-Steps hinzufügen
```

## Nächste Schritte

Nach erfolgreichem Training:

1. **Evaluation**: Teste das Modell auf dem Test-Set
2. **Inference**: Verwende das trainierte Modell für Tracking auf neuen Videos
3. **Fine-tuning**: Falls nötig, weitere Anpassungen mit niedrigerer LR

---

**Erstellt**: 2025-11-25
**Autor**: Claude Code
