"""Sample-efficiency (learning curve) experiment across all architectures.

Answers the question: "does GNN performance still improve when we add more
training data, or is it already saturated?" — which in turn tells us whether
the small dataset (~800 station-months) is a real bottleneck for the deep
models or whether they are already at their achievable ceiling.

For each training fraction f in {0.25, 0.50, 0.75, 1.0}:
  - Keep the outer test set fixed (unchanged fraction of the full data).
  - Keep the inner val set fixed (unchanged fraction of the trainval pool).
  - Take the *last* f * N training snapshots (preserving temporal ordering —
    subsampling from the tail keeps the training window contiguous with val
    and mirrors how real deployments accumulate data).
  - Train each of SimpleGCN, TemporalGCN_GRU, TemporalAttentionGNN once
    (single seed) via the existing pipeline `_train_one_fold`.
  - Train a plain RandomForest on the same station-month feature vectors
    (rows) from the same subsampled window.

Output:
  - <plots_dir>/learning_curve.png   — test F1 vs fraction, one line per model
  - <plots_dir>/learning_curve.csv   — raw numbers (fraction, model, test_f1)

Standalone usage:
    python -m masters_research.pipeline.learning_curve --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric_temporal.signal import DynamicGraphTemporalSignalBatch, temporal_signal_split

from masters_research.data.temporal_loader import TemporalDataLoader
from masters_research.pipeline.dataset import create_temporal_dataset, interleaved_temporal_split
from masters_research.pipeline.train import _train_one_fold, test_model


# ---------------------------------------------------------------------------
# Snapshot subsampling
# ---------------------------------------------------------------------------

def _tail_subset(dataset: DynamicGraphTemporalSignalBatch, n_keep: int) -> DynamicGraphTemporalSignalBatch:
    """Return a dataset consisting of the LAST `n_keep` snapshots of the input.

    Tail-subsampling preserves temporal contiguity with the val slice that
    follows and mirrors the real-world deployment case where new snapshots
    accumulate — you always retrain on the most-recent-N months.
    """
    total = dataset.snapshot_count
    n_keep = max(1, min(n_keep, total))
    idxs = list(range(total - n_keep, total))
    return DynamicGraphTemporalSignalBatch(
        edge_indices=[dataset.edge_indices[i] for i in idxs],
        edge_weights=[dataset.edge_weights[i] for i in idxs],
        features=[dataset.features[i] for i in idxs],
        targets=[dataset.targets[i] for i in idxs],
        batches=[dataset.batches[i] for i in idxs],
        timestamps=[dataset.timestamps[i] for i in idxs],
    )


def _flatten_snapshots(dataset: DynamicGraphTemporalSignalBatch) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate every snapshot's node-feature and label tensors into
    tabular arrays for sklearn baselines. Each row = one (station, month)."""
    X, y = [], []
    n = getattr(dataset, "snapshot_count", None) or len(dataset)
    for i in range(n):
        snap = dataset[i]
        xi = snap.x.detach().cpu().numpy()
        yi = snap.y.detach().cpu().numpy().astype(int)
        X.append(xi); y.append(yi)
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def _make_cfg(model_type: str, base_cfg: dict, seed: int) -> dict:
    """Produce a per-model cfg_run dict driven by a shared healthy base."""
    cfg = dict(base_cfg)
    cfg["seed"] = seed
    cfg["model_type"] = model_type
    # Unset the boolean flags so model_type wins unambiguously
    cfg["use_simple_gcn"] = (model_type == "simple_gcn")
    cfg["use_temporal_attention"] = (model_type == "temporal_attention")
    cfg["use_attention"] = False
    return cfg


def _run_gnn(model_type: str, train_ds, val_ds, test_ds, base_cfg: dict, seed: int,
             num_classes: int, run_models_dir: Path) -> float:
    """Train a GNN via the existing pipeline and return test F1 on `test_ds`."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg_run = _make_cfg(model_type, base_cfg, seed)
    result = _train_one_fold(
        fold_label=f"lc_{model_type}_{seed}",
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg_run=cfg_run,
        cfg=base_cfg,
        device=torch.device("cpu"),
        run_models_dir=run_models_dir,
        num_classes_run=num_classes,
        trainval_dataset=None,
    )
    test_loader = TemporalDataLoader(test_ds)
    preds, labels, _ = test_model(result["model"], test_loader)
    return float(f1_score(labels, preds, average="weighted"))


def _run_rf(train_ds, test_ds, seed: int) -> float:
    """Train a RandomForest on flattened (station-month, features) tabular data."""
    X_tr, y_tr = _flatten_snapshots(train_ds)
    X_te, y_te = _flatten_snapshots(test_ds)
    # Deliberately small grid to keep the learning curve run cheap. RF is
    # famously robust — even default hyperparameters give near-optimal
    # results on small tabular datasets, so an expensive GridSearchCV would
    # spend runtime on marginal improvements.
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    return float(f1_score(y_te, rf.predict(X_te), average="weighted"))


def _run_keras(train_ds, test_ds, seed: int, num_classes: int, epochs: int = 60) -> float:
    """Train a small MLP baseline mirroring the architecture used in
    baselines/baseline_models.py (Dense 128 -> 64 -> 32 -> 32 -> softmax).

    Kept as a separate, lazy-imported function so users who don't have
    TensorFlow installed can still run the learning curve for RF+GNNs.
    Fixed hyperparameters (no grid) to keep the per-fraction cost bounded.
    """
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TF INFO/WARNING spam
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.utils import to_categorical

    tf.keras.utils.set_random_seed(seed)

    X_tr, y_tr = _flatten_snapshots(train_ds)
    X_te, y_te = _flatten_snapshots(test_ds)
    y_tr_1h = to_categorical(y_tr, num_classes=num_classes)

    model = Sequential([
        Input(shape=(X_tr.shape[1],)),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_tr, y_tr_1h, epochs=epochs, batch_size=32, verbose=0,
              validation_split=0.1)
    preds = np.argmax(model.predict(X_te, verbose=0), axis=1)
    return float(f1_score(y_te, preds, average="weighted"))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(df: pd.DataFrame, out_path: Path) -> None:
    """Test F1 vs training fraction, one line per model.

    If multiple seeds per (model, fraction) are present, the line is the
    mean across seeds and a shaded band spans ±1 standard deviation —
    which is what a proper learning curve looks like for noisy models.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5))
    order = ["SimpleGCN", "TemporalGCN_GRU", "TemporalAttentionGNN", "RandomForest", "KerasMLP"]
    palette = {"SimpleGCN": "#1f77b4", "TemporalGCN_GRU": "#ff7f0e",
               "TemporalAttentionGNN": "#2ca02c", "RandomForest": "#d62728",
               "KerasMLP": "#9467bd"}
    agg = df.groupby(["model", "train_fraction"])["test_f1"].agg(["mean", "std", "count"]).reset_index()
    for model in order:
        sub = agg[agg["model"] == model].sort_values("train_fraction")
        if sub.empty:
            continue
        color = palette.get(model, None)
        # Std of a single observation is NaN — fall back to 0 for plotting so the
        # single-seed case degrades gracefully to a plain line.
        stds = sub["std"].fillna(0.0).to_numpy()
        means = sub["mean"].to_numpy()
        fracs = sub["train_fraction"].to_numpy()
        ax.plot(fracs, means, marker="o", linewidth=2, label=model, color=color)
        if (stds > 0).any():
            ax.fill_between(fracs, means - stds, means + stds, alpha=0.15, color=color)

    ax.set_xlabel("Training-data fraction")
    ax.set_ylabel("Test F1 (weighted, mean ±1 std across seeds)")
    n_seeds = int(agg["count"].max()) if not agg.empty else 1
    ax.set_title(f"Learning curve — sample efficiency across architectures (n_seeds={n_seeds})")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(base_path: Path, config_path: Path, seeds: tuple[int, ...] = (42,),
         fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)) -> Path:
    base_path = Path(base_path).resolve()
    cfg = json.loads(Path(config_path).read_text())
    # Collapse any list-valued config field to its first entry — grid semantics
    # aren't wanted here; we're running a single seed per (model, fraction).
    for k, v in list(cfg.items()):
        if isinstance(v, list) and v:
            cfg[k] = v[0]

    # Data
    data_path = base_path / cfg.get("data_path", "Dataset/TransientData/processedData.csv")
    df = pd.read_csv(data_path)
    dataset = create_temporal_dataset(
        df,
        seasonal_normalization=cfg.get("seasonal_normalization", True),
        add_lag_features=cfg.get("add_lag_features", True),
        n_classes=int(cfg.get("num_classes", 3)),
    )
    cfg["in_channels"] = int(dataset.features[0].shape[1])
    num_classes = int(cfg.get("num_classes", 3))

    # Splits — mirror the training pipeline exactly.
    test_holdout = float(cfg.get("test_holdout_ratio", 0.2))
    trainval, test_ds = interleaved_temporal_split(dataset, train_ratio=1.0 - test_holdout)
    inner_train_full, val_ds = temporal_signal_split(trainval, train_ratio=cfg.get("train_ratio", 0.8))
    n_train_full = inner_train_full.snapshot_count
    print(f"Learning curve setup: full inner-train = {n_train_full} snapshots, "
          f"val = {val_ds.snapshot_count}, test = {test_ds.snapshot_count}")

    plots_dir = base_path / cfg.get("plots_dir", "src/masters_research/artifacts/plots")
    models_dir = base_path / cfg.get("save_dir", "src/masters_research/artifacts/models") / "_learning_curve"
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for frac in fractions:
        n_keep = max(1, int(round(n_train_full * frac)))
        subs_train = _tail_subset(inner_train_full, n_keep)
        print(f"\n=== Fraction {frac:.2f} -> training on {subs_train.snapshot_count} snapshots ===")

        for seed in seeds:
            print(f"\n  [seed={seed}]")
            for model_type, pretty in [
                ("simple_gcn", "SimpleGCN"),
                ("temporal_gru", "TemporalGCN_GRU"),
                ("temporal_attention", "TemporalAttentionGNN"),
            ]:
                try:
                    f1 = _run_gnn(
                        model_type, subs_train, val_ds, test_ds, cfg, seed,
                        num_classes, models_dir / pretty / f"frac_{int(frac*100)}_seed_{seed}",
                    )
                    print(f"    {pretty:22s} test F1 = {f1:.4f}")
                except Exception as e:
                    print(f"    {pretty:22s} FAILED: {e}")
                    f1 = float("nan")
                rows.append({"model": pretty, "train_fraction": frac, "seed": seed,
                             "n_train_snapshots": subs_train.snapshot_count, "test_f1": f1})

            # RandomForest on the same (subsampled train, unchanged test).
            try:
                f1 = _run_rf(subs_train, test_ds, seed)
                print(f"    {'RandomForest':22s} test F1 = {f1:.4f}")
            except Exception as e:
                print(f"    RandomForest           FAILED: {e}")
                f1 = float("nan")
            rows.append({"model": "RandomForest", "train_fraction": frac, "seed": seed,
                         "n_train_snapshots": subs_train.snapshot_count, "test_f1": f1})

            # Keras MLP baseline — mirrors baseline_models.py architecture.
            # Wrapped in try/except so a missing TensorFlow install doesn't kill
            # the whole learning-curve experiment.
            try:
                f1 = _run_keras(subs_train, test_ds, seed, num_classes)
                print(f"    {'KerasMLP':22s} test F1 = {f1:.4f}")
            except Exception as e:
                print(f"    KerasMLP               FAILED: {e}")
                f1 = float("nan")
            rows.append({"model": "KerasMLP", "train_fraction": frac, "seed": seed,
                         "n_train_snapshots": subs_train.snapshot_count, "test_f1": f1})

    out_df = pd.DataFrame(rows)
    out_csv = plots_dir / "learning_curve.csv"
    out_df.to_csv(out_csv, index=False)
    out_png = plots_dir / "learning_curve.png"
    _plot(out_df, out_png)

    # Also emit a compact per-(model, fraction) mean±std summary.
    agg = (out_df.groupby(["model", "train_fraction"])["test_f1"]
                .agg(["mean", "std", "count"]).reset_index()
                .sort_values(["train_fraction", "model"]))
    agg_csv = plots_dir / "learning_curve_summary.csv"
    agg.to_csv(agg_csv, index=False)

    print(f"\nSaved learning curve to:\n  {out_csv}\n  {out_png}\n  {agg_csv}")
    print("\nMean ± std per (model, fraction):")
    print(agg.to_string(index=False))
    return out_png


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/improve_v1.json",
                        help="Base config for GNN hyperparameters (list-valued fields are collapsed to their first entry).")
    parser.add_argument("--seeds", default="42,123,456",
                        help="Comma-separated list of seeds. Each (model, fraction) is trained once per seed; the plot shows mean ± std across seeds.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    seed_list = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    main(project_root, cfg_path, seeds=seed_list)
