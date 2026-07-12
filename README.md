# MastersResearch

Master's research project on **temporal graph neural networks for spatio-temporal sensor data quality classification**.

The pipeline models a South African weather station network as a temporal graph and compares several GNN architectures against classical ML baselines, with a focus on handling **temporal distribution shift** and **feature drift**.

## Research goals

- Compare GNN architectures (plain GCN, GCN + GRU, GCN + temporal attention, GAT + selective attention) with classical baselines (Random Forest, GBDT, SVM) for temporal node classification.
- Study the effect of regularization (focal loss, label smoothing, entropy regularization, SWA) on class-imbalanced, drift-prone temporal graph data.
- Provide a reusable **temporal quality audit** that scores a dataset along five dimensions: node coverage, timestamp regularity, label drift, feature drift, and temporal autocorrelation.

## Dataset

- 8 South African weather stations, ~10 years of monthly observations.
- Raw features per station/month: temperature, humidity, rainfall, radiation, evaporation, wind speed (8 attributes total).
- Features are MinMax-normalized; the classification target (3 classes) is derived from the mean of the normalized features.
- Graph edges are built from station geography using Haversine distance + altitude difference.
- Raw CSVs live in `Dataset/OriginalDataset/`; processed intermediates in `Dataset/TransientData/`.

## Repository layout

```
src/masters_research/
  data/         # CSV loading, spatial graph construction, temporal snapshot iteration
  models/       # 4 GNN architectures (see below)
  pipeline/     # training loop, grid search, temporal quality audit, evaluation, reporting
  baselines/    # Random Forest / GBDT / SVM baselines
  cli/          # CLI entry point
  artifacts/    # saved model weights, grid-search result CSVs, plots
configs/        # per-experiment JSON hyperparameter files
Dataset/        # raw and processed data
Notebooks/      # exploratory analysis
docs/           # PROJECT_STRUCTURE.md
run.py          # top-level entry point
requirements.txt
```

## Models

| Model | File | Description |
|---|---|---|
| `SimpleGCNClassifier` | `models/simple_gcn_classifier.py` | 3-layer GCN, no temporal component; per-snapshot classification |
| `TemporalSpatialNodeClassifier` | `models/temporal_spatial_classifier.py` | GCN + GRU stack — original temporal baseline |
| `TemporalSpatialAttentionNodeClassifier` | `models/temporal_spatial_attention.py` | GCN + multi-head temporal attention over a history buffer |
| `TemporalAttentionGNN` | `models/temporal_attention_gnn.py` | GAT layers + selective focus on relevant historical patterns |

All models share input normalization, dropout, focal loss / label smoothing, entropy regularization, and optional Stochastic Weight Averaging (SWA).

## Setup (Windows)

```powershell
# activate the virtualenv
& .\.venv\Scripts\Activate.ps1

# install torch first (Windows wheel), then the rest
python -m pip install --upgrade pip setuptools wheel
python -m pip install "torch==2.2.0"
python -m pip install -r requirements.txt
```

Python 3.12, CPU-only wheels. If `torch` / `torch-geometric` install fails, upgrade `pip` / `setuptools` / `wheel` first and re-run.

## Running

From the project root:

```powershell
# default config
python run.py

# specific experiment
python run.py --config configs/gat_swa_5c.json
```

`run.py` prepends `src/` to `PYTHONPATH`, then invokes the CLI, which loads the JSON config, runs training (with grid search over any list-valued hyperparameters), and writes artifacts to `src/masters_research/artifacts/`.

## Configuration

Experiments are fully driven by JSON files under `configs/`. Prefer creating a new config over editing `default.json`.

- Hyperparameter fields are typically lists (e.g. `"lr": [0.0005]`, `"seed": [42, 123, 456]`) to enable grid search; path/device/toggle fields are scalars.
- Model choice is toggled by boolean flags (`use_simple_gcn`, `use_temporal_attention`, `use_attention` / `use_gat`, `use_swa`).
- See `configs/default.json` for the full schema. Notable variants: `configs/gat_swa_5c.json` (GAT + SWA), `configs/gat_gru.json` (GAT + GRU), `configs/deep_gru_cv.json` (deeper GRU with CV), `configs/gridsearch_focused_5c.json` (focused grid search).

## Outputs

- **Model weights & grid results:** `src/masters_research/artifacts/models/`
- **Plots (per-timestamp node counts, summary graphs):** `src/masters_research/artifacts/plots/`
- **Comparative reports** across model types are produced by `pipeline/compare_reports.py` and `pipeline/model_type_report.py`.

## Tech stack

PyTorch 2.2, PyTorch Geometric 2.6, torch-geometric-temporal, scikit-learn, Captum (for feature-attribution explainability), matplotlib / seaborn / networkx for visualization.

## Further reading

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for a walkthrough of the src-layout and file responsibilities.
