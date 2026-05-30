# QAIE-ABSA-2025-adaption - ABSA Quads Augmentierung & Training

Dieses Repository enthält Skripte zur Augmentierung von Aspect-Based Sentiment Analysis (ABSA) Datensätzen und zum anschließenden Training eines T5-Modells.

## Zentrale Konfiguration

Alle wichtigen Pfade und Hyperparameter (Modellnamen, Shots, Seeds, Tasks, Datasets) werden zentral in [qaie_const.py](qaie_const.py) verwaltet.

- **Shots:** 10, 50, 100
- **Seeds:** 0, 1, 2, 3, 4
- **Modell (vLLM):** google/gemma-4-31b-it (Chat/Augmentierung)
- **Modell (T5):** google-t5/t5-base (Training)

## Voraussetzungen

*   **Datenstruktur:** Die Basis-Datensätze müssen sich im ABSA Toolkit Ordner befinden: `/home/hellwig/absa-toolkit/data/`.
*   **LLM Zugriff:** Die Augmentierungs-Skripte nutzen `llm.py` mit `vLLM`.
*   **Umgebung:** Python mit installierten Abhängigkeiten (siehe `requirements.sh` für Setup-Hinweise).

## Ausführungsreihenfolge

### 1. Daten-Augmentierung
Nutzt vLLM mit Batching und 5 Seeds. Die originalen Few-Shot Daten werden automatisch in die Augmentierungs-Datei übernommen.
- `python 00_create_aug_unified.py`

Erzeugt Dateien in `01_augmentations/fs_examples/.../seed_{0-4}/aug.txt`.

### 2. Generierung impliziter Beispiele
Erzeugt Reasoning-Texte für alle Daten in `aug.txt`.
- `python 01_create_implicit_examples_unified.py`

Erzeugt `aug_im.txt` (pro Seed).

### 3. Training und Evaluation
Führt das Training des T5-Modells durch und speichert detaillierte Ergebnisse inkl. GPU-Metriken.
- `python 02_exec.py`

Die Ergebnisse werden unter `03_results/` gespeichert (Suffix `qaie`).

## Projektstruktur

- `qaie_const.py`: Zentrale Konstanten.
- `00_...`: Augmentierungs-Skripte.
- `01_...`: Generierung impliziter Begründungen.
- `02_...`: Training und Orchestrierung.
- `03_results/`: Speicherort für Ergebnisse und GPU-Statistiken.
- `data_utils.py` / `eval_utils.py`: Hilfsfunktionen.
- `llm.py`: vLLM Wrapper.
