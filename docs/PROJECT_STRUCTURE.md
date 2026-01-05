# Project Structure and Usage

This document describes the structured layout of the MastersResearch project after consolidation into a src-based package.

## Structure
- src/masters_research/
  - data/
    - loader.py — load and preprocess original station CSVs
    - graph.py — build complete station-pair edges with Haversine + altitude
    - temporal_loader.py — iterator over temporal snapshots
  - models/
    - temporal_spatial_classifier.py — GNN for node classification
  - pipeline/
    - dataset.py — build DynamicGraphTemporalSignalBatch from processed CSV
    - train.py — training loop, evaluation, and grid search orchestration
  - cli/
    - main.py — detects project root and runs training
- configs/default.json — hyperparameters, device, paths
- run.py — convenience runner that adds src/ to PYTHONPATH and invokes CLI
- src/masters_research/artifacts/
  - models/ — saved model weights and grid results CSV
  - plots/ — nodes-per-timestamp and summary graphs across runs
- Dataset/OriginalDataset/ — raw station CSVs and metadata
- Dataset/TransientData/ — processed/intermediate CSVs (e.g., corrected1_df.csv)
- ModelBackup/ — previous saved models and grid results
- Notebooks/ — exploratory notebooks and experiments
- requirements.txt — Python dependencies

## Setup (Windows)
1. Activate your virtualenv:
```powershell
& .\.venv\Scripts\Activate.ps1
```
2. Install dependencies (Torch first helps on Windows):
```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install "torch==2.2.0"
python -m pip install -r requirements.txt
```

## Run
From the project root:
```powershell
python run.py
```
This will load data, read the processed graph CSV from `configs/default.json`, train the GNN, and save results to `models/`.

## Configuration
Edit `configs/default.json` to change hyperparameters, device, and paths. By default, models save to `src/masters_research/artifacts/models` and plots to `src/masters_research/artifacts/plots`.

## Notes
- If `torch` or `torch-geometric` installs fail on Windows for Python 3.12, ensure you use compatible wheels and upgrade `pip`/`setuptools`/`wheel`.
- Datasets and notebooks remain in their original locations; pipeline code is consolidated under `src/masters_research`.
