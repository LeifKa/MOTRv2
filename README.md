# MOTRv2 – Beach Volleyball Tracking

Dieses Repository basiert auf [MOTRv2](https://arxiv.org/abs/2211.09791) und wurde für das Tracking von Beach-Volleyball-Spielern und -Ball angepasst. Als Detektor wird [D-FINE](../D-FINE) anstelle von YOLOX verwendet.

Das originale README ist unter [README_ORIGINAL.md](README_ORIGINAL.md) verfügbar.

---

## Installation

```bash
conda create -n motrv2 python=3.7
conda activate motrv2
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 -c pytorch
pip install -r requirements.txt

# MultiScaleDeformableAttention kompilieren
cd ./models/ops && sh ./make.sh
```

---

## Inferenz (Beach Volleyball)

Gewichte separat herunterladen und in `weights/` ablegen.

```bash
python submit_dance.py \
    --meta_arch motr \
    --dataset_file e2e_dance \
    --with_box_refine \
    --query_interaction_layer QIMv2 \
    --num_queries 10 \
    --use_checkpoint \
    --resume ./weights/motrv2_finetuned_volleyball.pth \
    --det_db ./inputs/detections/det_db_beach_volleyball.json \
    --mot_path ./data/Dataset/mot \
    --output_dir outputs/inference_beach_volleyball \
    --exp_name beach_volleyball_test \
    --score_threshold 0.7 \
    --miss_tolerance 20
```

**Parameter:**
- `--resume` – Gewichte des feingetunten Modells
- `--det_db` – Detection-Datenbank (JSON), erzeugt mit D-FINE (`tools/inference/torch_inf.py --motrv2`)
- `--mot_path` – Pfad zum Datensatz (MOT-Format)
- `--score_threshold` – Minimaler Konfidenz-Score für Tracks
- `--miss_tolerance` – Frames, nach denen ein verlorener Track gelöscht wird

Fertige Argument-Dateien (`.args`) für verschiedene Konfigurationen befinden sich in `configs/`.

---

## Visualisierung

```bash
python tools/visualization/visualize_tracking.py \
    --images ./data/Dataset/mot/volleyball/test/test1/img1 \
    --tracking outputs/inference_motrv2_yolox_vanilla/inference_motrv2_yolox_vanilla_th0.4_mt50.txt \
    --output outputs/inference_motrv2_yolox_vanilla/inference_motrv2_yolox_vanilla_th0.4_mt50.mp4 \
    --fps 30
```

---

## Hinzugefügte Dateien

### Konfigurationen (`configs/`)

| Datei | Beschreibung |
|---|---|
| `beach_volleyball.args` | Inferenz-Konfiguration für Beach Volleyball |
| `volleyball_finetune.args` | Standard-Finetuning (Spieler) |
| `volleyball_ball_finetune.args` | Finetuning inkl. Ball-Detektion |
| `volleyball_finetune_aggressive_*.args` | Finetuning-Varianten mit hoher Learning Rate |
| `volleyball_finetune_minimal_*.args` | Finetuning-Varianten mit minimaler Regularisierung |
| `volleyball_finetune_moderate_*.args` | Finetuning-Varianten mit moderater Learning Rate |
| `volleyball_full_finetune.args` | Vollständiges Finetuning aller Layer |
| `pseudo_labels_yolox.args` | Inferenz zur Pseudo-Label-Generierung |

### SLURM-Jobs (`slurmjobs/`)

| Datei | Beschreibung |
|---|---|
| `run_motrv2_tracking.slurm` | Tracking-Inferenz auf dem Cluster |
| `run_motr_finetune_volleyball.slurm` | Finetuning auf Volleyball-Daten |
| `run_motr_finetune_volleyball_fast.slurm` | Schnelles Finetuning (weniger Epochen) |
| `run_inference_v2.slurm` | Inferenz-Job (aktuellste Version) |
| `run_visualization_v2.slurm` | Visualisierung der Tracking-Ergebnisse |
| `run_volleyball_ball_*.slurm` | Finetuning und Inferenz mit Ball-Detektion |
| `run_vb_*.slurm` | Finetuning-Varianten (verschiedene Hyperparameter) |

### Visualisierungstools (`tools/visualization/`)

| Datei | Beschreibung |
|---|---|
| `visualize_tracking.py` | Erstellt MP4-Video aus MOTRv2-Tracking-Ergebnissen (MOT-TXT-Format) |
| `visualize_detections.py` | Visualisiert D-FINE-Detektionen als Video mit Bounding Boxes |
| `visualize_gt.py` | Visualisiert Ground-Truth-Annotierungen im MOT-Format |

### Finetuning-Tools (`tools/fine_tuning/`)

| Datei | Beschreibung |
|---|---|
| `finetune_for_dfine.py` | Trainiert ausschließlich den `yolox_embed`-Layer für D-FINE-Kompatibilität |
| `build_ball_det_db.py` | Erstellt Detection-DB (JSON) für MOTRv2 mit Ball-Detektionen |
| `convert_labelstudio_to_mot.py` | Konvertiert LabelStudio-JSON-Export ins MOT-Ground-Truth-Format |
| `convert_det_db_to_mot.py` | Konvertiert Detection-DB JSON ins MOT-TXT-Format |
| `create_det_db_for_ball_finetune.sh` | Shell-Script zur automatischen Erstellung der Ball-Detection-DB |

### Datenvorbereitung (`tools/data_prep/`)

| Datei | Beschreibung |
|---|---|
| `convert_to_gt.py` | Konvertiert MOTRv2-Tracking-Output ins MOT-Ground-Truth-Format (für Evaluation) |

### Analyse (`analysis/`)

| Datei | Beschreibung |
|---|---|
| `mot_evaluation.py` | Berechnet MOT-Metriken (HOTA, MOTA, IDF1, DetA, AssA) |
| `demo_evaluation.py` | Beispiel-Evaluation mit Demo-Daten |
| `example_usage.py` | Verwendungsbeispiele für `mot_evaluation.py` |
| `results/` | Evaluation-Ergebnisse und Plots für alle Konfigurationen |

### Dokumentation

| Datei | Beschreibung |
|---|---|
| `FINETUNE_CONFIGURATIONS.md` | Dokumentation aller Finetuning-Konfigurationen und ihrer Ergebnisse |
| `README_DIRECTORY.md` | Übersicht der Verzeichnisstruktur |
| `SUMMARY.md` | Projektzusammenfassung und Erkenntnisse |
| `CLAUDE.md` | Anweisungen für Claude Code (KI-Assistent) |

### Modifizierte Originaldateien

| Datei | Änderung |
|---|---|
| `submit_dance.py` | Angepasst für Beach-Volleyball-Daten und D-FINE als Detektor |
| `main.py` | Erweitert um Finetuning-Parameter |
| `datasets/dance.py` | Angepasst für Volleyball-Datensatz-Struktur |
