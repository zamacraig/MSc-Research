# CLAUDE.md

Guidance for Claude when working in this repository.

## Project

Master's thesis research on **temporal graph neural networks for spatio-temporal sensor data quality classification**. The dataset is a South African weather station network (8 stations, ~10 years of monthly climate observations); the task is 3-class node quality classification per timestamp. The research narrative centers on **temporal distribution shift**, **feature drift**, and **mode collapse** — keep those framings when discussing results or suggesting changes.

## Environment

- **OS:** Windows 11, `bash` and PowerShell both available. Use forward-slash paths where possible.
- **Python:** 3.12, CPU-only. Virtualenv at `.venv/`.
- **Key deps** (see `requirements.txt`): `torch==2.2.0`, `torch-geometric==2.6.1`, `torch-geometric-temporal==0.54.0`, `scikit-learn==1.4.2`, `captum==0.7.0`.
- Torch/PyG install order matters on Windows — install `torch` first, then the rest.

## Running the pipeline

Entry point is `run.py` at the repo root, which prepends `src/` to `PYTHONPATH` and calls `masters_research.cli.main`.

```powershell
# activate venv
& .\.venv\Scripts\Activate.ps1

# default config
python run.py

# specific experiment config
python run.py --config configs/gat_swa_5c.json
```

All experiments are **config-driven** via JSON files in `configs/`. Prefer creating/editing a config over hardcoding hyperparameters in code.

## Layout

```
src/masters_research/
  data/         # loader.py, graph.py, temporal_loader.py — CSV load, spatial graph (Haversine + altitude), temporal snapshots
  models/       # 4 architectures: SimpleGCNClassifier, TemporalSpatialNodeClassifier (GCN+GRU),
                #                  TemporalSpatialAttentionNodeClassifier, TemporalAttentionGNN (GAT + selective attention)
  pipeline/     # train.py (large — training loop, grid search, explainability),
                # temporal_quality.py (5-dim dataset quality audit),
                # dataset.py, evaluate_best.py, compare_reports.py, model_type_report.py
  baselines/    # baseline_models.py — Random Forest, GBDT, SVM comparisons
  cli/main.py   # CLI entry
  artifacts/    # models/ (weights + grid results CSV), plots/
configs/        # per-experiment JSON hyperparameter files
Dataset/        # OriginalDataset/ (raw station CSVs) + TransientData/ (processed)
Notebooks/      # exploratory
docs/           # PROJECT_STRUCTURE.md
```

## Conventions

- **Config schema:** hyperparameter fields are usually lists (`"lr": [0.0005]`, `"seed": [42, 123, 456]`) so `train.py` can grid-search across them. Scalars are used for path/device/toggle fields.
- **Model selection** is controlled by config flags (`use_simple_gcn`, `use_temporal_attention`, `use_attention` / `use_gat`, `use_swa`) rather than a `model_name` string.
- **Regularization toolkit** used across architectures: focal loss (`focal_loss`, `focal_gamma`), label smoothing, entropy regularization, dropout, gradient clipping, SWA. These exist to combat class imbalance and mode collapse — don't remove them without checking why they were added.
- **Artifacts** land in `src/masters_research/artifacts/` (models + plots). Grid-search results are aggregated into CSVs there.
- **Temporal quality audit** (`temporal_quality.py`) scores the dataset on 5 dimensions (node coverage, timestamp regularity, label drift, feature drift, temporal autocorrelation) — it's a first-class part of the pipeline, not just a debug tool.

## When editing

- Prefer adding new config files under `configs/` for new experiments rather than mutating `default.json`.
- If you touch `train.py` (large orchestration file), make surgical edits — many code paths key off config flags.
- The data loader normalizes 8 raw weather features via MinMaxScaler and derives the classification target from the normalized feature mean; changes to normalization change label semantics.
- Baselines in `baselines/baseline_models.py` are used for comparative reporting — keep their interface aligned with what `compare_reports.py` / `model_type_report.py` consume.
