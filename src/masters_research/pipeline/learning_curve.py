"""Sample-efficiency (learning curve) experiment across all architectures.

Answers the question: "does GNN performance still improve when we add more
training data, or is it already saturated?" — which in turn tells us whether
the small dataset (~800 station-months) is a real bottleneck for the deep
models or whether they are already at their achievable ceiling.

For each training fraction f in {0.25, 0.50, 0.75, 1.0}:
  - Keep the outer test set fixed and the inner val set fixed, using the same
    outer hold-out and deterministic inner split as pipeline.train.
  - Keep the *last* f * N training snapshots (the most recent months).
  - Train SimpleGCN, TemporalGCN_GRU and TemporalAttentionGNN through the
    main training loop (`_train_one_fold`), and RandomForest / Keras MLP with
    the exact settings of baselines_runner, once per seed.
  - Score each model on its own training subset AND on the fixed test set.
    Only labelled (station, month) cases are used, as in the main evaluation.

Output (prefixed with run_name when set):
  - <plots_dir>/learning_curve.png          — train/test F1 vs fraction per model
  - <plots_dir>/learning_curve.csv          — raw per-seed numbers
  - <plots_dir>/learning_curve_summary.csv  — mean/std per (model, fraction)

For per-epoch (or per-tree) learning curves of the main runs, see
pipeline/training_curves.py.

Standalone usage:
    python -m masters_research.pipeline.learning_curve --config configs/spi3_ss_rerun.json
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
from masters_research.pipeline.dataset import (
    create_temporal_dataset, interleaved_temporal_split, stratified_temporal_signal_split,
)
from masters_research.pipeline.train import _train_one_fold, test_model
from masters_research.pipeline.baselines_runner import (
    flatten_snapshots, _fit_random_forest, _fit_keras_mlp,
)


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


# ---------------------------------------------------------------------------
# Model runners — each returns (train_f1, test_f1)
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
    # Per-epoch curves are not needed here and would cost an extra pass per epoch.
    cfg["record_training_curves"] = False
    return cfg


def _f1(labels, preds) -> float:
    return float(f1_score(labels, preds, average="weighted"))


def _run_gnn(model_type: str, train_ds, val_ds, test_ds, base_cfg: dict, seed: int,
             num_classes: int, run_models_dir: Path) -> tuple[float, float]:
    """Train a GNN via the main pipeline's training loop (same checkpoint
    selection, SWA, early stopping) and score it on train and test."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg_run = _make_cfg(model_type, base_cfg, seed)
    result = _train_one_fold(
        fold_label=f"lc_{model_type}_{seed}",
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg_run=cfg_run,
        cfg=cfg_run,
        device=torch.device("cpu"),
        run_models_dir=run_models_dir,
        num_classes_run=num_classes,
        trainval_dataset=None,
    )
    tr_preds, tr_labels, _ = test_model(result["model"], TemporalDataLoader(train_ds))
    te_preds, te_labels, _ = test_model(result["model"], TemporalDataLoader(test_ds))
    return _f1(tr_labels, tr_preds), _f1(te_labels, te_preds)


def _run_baseline(name: str, train_ds, test_ds, seed: int, cfg: dict, num_classes: int) -> tuple[float, float]:
    """RF / Keras MLP with the exact settings used in the main comparison
    (baselines_runner), on labelled station-month rows only."""
    X_tr, y_tr = flatten_snapshots(train_ds)
    X_te, y_te = flatten_snapshots(test_ds)
    if name == "RandomForest":
        model, predict = _fit_random_forest(X_tr, y_tr, seed, cfg)
    else:
        model, predict = _fit_keras_mlp(X_tr, y_tr, seed, cfg, num_classes)
    return _f1(y_tr, predict(model, X_tr)[0]), _f1(y_te, predict(model, X_te)[0])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(df: pd.DataFrame, out_path: Path) -> None:
    """Train and test F1 vs training fraction, one panel per model.

    Mean across seeds with a ±1 std band. The gap between the two lines is
    the generalisation gap; a test line still rising at fraction 1.0 means
    more data would help, a flat one means the model has saturated.
    """
    order = [m for m in ["SimpleGCN", "TemporalGCN_GRU", "TemporalAttentionGNN", "RandomForest", "KerasMLP"]
             if m in set(df["model"])]
    agg = (df.groupby(["model", "train_fraction"])[["train_f1", "test_f1"]]
             .agg(["mean", "std"]).reset_index())
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    n_seeds = int(df.groupby(["model", "train_fraction"]).size().max())
    fig, axes = plt.subplots(1, len(order), figsize=(3.4 * len(order), 3.8), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], order):
        sub = agg[agg["model"] == model].sort_values("train_fraction")
        x = sub["train_fraction"].to_numpy()
        for part, color, name in [("train", "#2a6fb0", "Train"), ("test", "#d0662b", "Test")]:
            m = sub[f"{part}_f1_mean"].to_numpy()
            sd = sub[f"{part}_f1_std"].fillna(0.0).to_numpy()
            ax.plot(x, m, marker="o", linewidth=1.8, color=color, label=name)
            ax.fill_between(x, m - sd, m + sd, alpha=0.15, color=color)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Training-data fraction")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("Weighted F1")
    axes[0][0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Sample-size learning curves (mean ± 1 std, {n_seeds} seeds)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
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
        # Leak-free preprocessing — see pipeline.train for rationale.
        test_holdout_ratio=float(cfg.get("test_holdout_ratio", 0.2)),
        stratified_split=bool(cfg.get("stratified_split", False)),
        forecast_horizon=int(cfg.get("forecast_horizon", 1)),
        target_type=str(cfg.get("target_type", "feature_mean")),
        target_window=int(cfg.get("target_window", 3)),
        base_path=base_path,
    )
    cfg["in_channels"] = int(dataset.features[0].shape[1])
    num_classes = int(cfg.get("num_classes", 3))

    # Splits — mirror the training pipeline exactly (same outer hold-out and
    # same deterministic inner train/val rule as pipeline.train).
    test_holdout = float(cfg.get("test_holdout_ratio", 0.2))
    trainval, test_ds = interleaved_temporal_split(dataset, train_ratio=1.0 - test_holdout)
    if bool(cfg.get("stratified_split", False)):
        inner_train_full, val_ds = stratified_temporal_signal_split(
            trainval, train_ratio=float(cfg.get("train_ratio", 0.8)), num_classes=num_classes)
    else:
        inner_train_full, val_ds = temporal_signal_split(trainval, train_ratio=cfg.get("train_ratio", 0.8))
    n_train_full = inner_train_full.snapshot_count
    print(f"Learning curve setup: full inner-train = {n_train_full} snapshots, "
          f"val = {val_ds.snapshot_count}, test = {test_ds.snapshot_count}")

    plots_dir = base_path / cfg.get("plots_dir", "src/masters_research/artifacts/plots")
    models_dir = base_path / cfg.get("save_dir", "src/masters_research/artifacts/models") / "_learning_curve"
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    gnns = [("simple_gcn", "SimpleGCN"), ("temporal_gru", "TemporalGCN_GRU"),
            ("temporal_attention", "TemporalAttentionGNN")]
    rows = []
    for frac in fractions:
        n_keep = max(1, int(round(n_train_full * frac)))
        subs_train = _tail_subset(inner_train_full, n_keep)
        n_cases = int(flatten_snapshots(subs_train)[1].size)
        print(f"\n=== Fraction {frac:.2f} -> {subs_train.snapshot_count} snapshots "
              f"({n_cases} labelled cases) ===")

        for seed in seeds:
            print(f"\n  [seed={seed}]")
            jobs = [(pretty, lambda mt=mt, pretty=pretty: _run_gnn(
                        mt, subs_train, val_ds, test_ds, cfg, seed, num_classes,
                        models_dir / pretty / f"frac_{int(frac*100)}_seed_{seed}"))
                    for mt, pretty in gnns]
            jobs += [(name, lambda name=name: _run_baseline(name, subs_train, test_ds, seed, cfg, num_classes))
                     for name in ("RandomForest", "KerasMLP")]
            for pretty, job in jobs:
                try:
                    tr_f1, te_f1 = job()
                    print(f"    {pretty:22s} train F1 = {tr_f1:.4f}  test F1 = {te_f1:.4f}")
                except Exception as e:
                    print(f"    {pretty:22s} FAILED: {e}")
                    tr_f1 = te_f1 = float("nan")
                rows.append({"model": pretty, "train_fraction": frac, "seed": seed,
                             "n_train_snapshots": subs_train.snapshot_count,
                             "n_train_cases": n_cases, "train_f1": tr_f1, "test_f1": te_f1})

    prefix = f"{cfg['run_name']}_" if cfg.get("run_name") else ""
    out_df = pd.DataFrame(rows)
    out_csv = plots_dir / f"{prefix}learning_curve.csv"
    out_df.to_csv(out_csv, index=False)
    out_png = plots_dir / f"{prefix}learning_curve.png"
    _plot(out_df, out_png)

    # Also emit a compact per-(model, fraction) mean±std summary.
    agg = (out_df.groupby(["model", "train_fraction"])[["train_f1", "test_f1"]]
                .agg(["mean", "std"]).round(4))
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index().sort_values(["model", "train_fraction"])
    agg_csv = plots_dir / f"{prefix}learning_curve_summary.csv"
    agg.to_csv(agg_csv, index=False)

    print(f"\nSaved learning curve to:\n  {out_csv}\n  {out_png}\n  {agg_csv}")
    print("\nMean ± std per (model, fraction):")
    print(agg.to_string(index=False))
    return out_png


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spi3_ss_rerun.json",
                        help="Base config for GNN hyperparameters (list-valued fields are collapsed to their first entry).")
    parser.add_argument("--seeds", default="42,123,456,789,1024",
                        help="Comma-separated list of seeds. Each (model, fraction) is trained once per seed; the plot shows mean ± std across seeds.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    seed_list = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    main(project_root, cfg_path, seeds=seed_list)
