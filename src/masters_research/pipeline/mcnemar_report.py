"""McNemar significance-test report across the five candidate models.

Answers the question that comparative-F1 tables leave open: *given two
classifiers evaluated on identical samples, is the difference in their error
patterns real or attributable to chance?* Uses the paired non-parametric
McNemar test (with Yates continuity correction) plus a per-class breakdown
with Bonferroni-corrected p-values.

Inputs
------
GNN predictions come from the ``test_probs.npz`` files already saved by
:mod:`pipeline.train` in each run directory — no GNN retrain is required,
because the same 207-sample validation set is used deterministically across
all GNN runs. Baseline (Random Forest, Keras MLP) predictions are trained
fresh here on the *same* seasonally-normalised processed data as the GNNs
so that every model's prediction vector is aligned sample-by-sample.

Outputs
-------
- ``plots/mcnemar_predictions.npz``   — every model's per-sample predictions
  and the shared label vector, saved for reproducibility.
- ``plots/mcnemar_table.tex``         — a LaTeX-ready significance table
  covering the four thesis-critical pairwise comparisons.
- Console — the same table printed as plain text.

Standalone usage
----------------
    python -m masters_research.pipeline.mcnemar_report
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2 as chi2_dist
from sklearn.ensemble import RandomForestClassifier
from torch_geometric_temporal.signal import temporal_signal_split

from masters_research.pipeline.dataset import (
    create_temporal_dataset,
    interleaved_temporal_split,
    stratified_temporal_signal_split,
)


# ---------------------------------------------------------------------------
# GNN prediction loading
# ---------------------------------------------------------------------------

def _find_best_run_dir(models_dir: Path, model_type: str, df: pd.DataFrame) -> Optional[Path]:
    """Return the on-disk directory of the best run for a given model type,
    ranked by final_test_f1 then best_f1 (matches model_type_report logic)."""
    sub = df[df["model_type"] == model_type].copy() if "model_type" in df.columns else df.copy()
    if sub.empty:
        return None
    for col in ("final_test_f1", "best_f1", "val_f1"):
        if col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sort_cols = [c for c in ("final_test_f1", "best_f1", "val_f1") if c in sub.columns]
    sub = sub.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    best = sub.iloc[0]
    run_dir = models_dir / model_type / str(best["run_id"])
    return run_dir if run_dir.exists() else None


def _load_gnn_predictions(models_dir: Path, df: pd.DataFrame) -> dict:
    """Return {display_name: (labels, preds)} for each GNN architecture whose
    best-run has a test_probs.npz on disk. Labels must match across models
    (deterministic val split) — an assertion is raised otherwise so the
    caller notices a split-config drift immediately."""
    out: dict = {}
    canonical_labels: Optional[np.ndarray] = None
    for display, folder in [
        ("SimpleGCN", "SimpleGCN"),
        ("TemporalGCN_GRU", "TemporalGCN_GRU"),
        ("TemporalAttentionGNN", "TemporalAttentionGNN"),
    ]:
        run_dir = _find_best_run_dir(models_dir, folder, df)
        if run_dir is None:
            print(f"[{display}] no best-run directory found — skipping.")
            continue
        npz_path = run_dir / "test_probs.npz"
        if not npz_path.exists():
            print(f"[{display}] test_probs.npz missing at {npz_path} — "
                  f"rerun training with the recent train.py to backfill.")
            continue
        with np.load(npz_path, allow_pickle=True) as data:
            labels = np.asarray(data["labels"]).astype(int)
            probs = np.asarray(data["probs"])
        preds = probs.argmax(axis=1).astype(int)
        if canonical_labels is None:
            canonical_labels = labels
        elif not np.array_equal(labels, canonical_labels):
            raise ValueError(
                f"[{display}] label vector disagrees with the first-loaded GNN. "
                f"McNemar requires identical test samples across compared models. "
                f"Check that all best-runs used the same outer split and train_ratio."
            )
        out[display] = (labels, preds)
        print(f"[{display}] loaded {len(preds)} predictions from {run_dir.name}")
    return out


# ---------------------------------------------------------------------------
# Baseline training on the same processed data / same split
# ---------------------------------------------------------------------------

def _flatten_snapshots(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate every snapshot's (nodes, features) block into a single
    tabular matrix so sklearn/Keras models see (station-month, feature) rows.
    Mirrors the flatten used in pipeline/learning_curve.py."""
    xs, ys = [], []
    n = getattr(dataset, "snapshot_count", None) or len(dataset)
    for i in range(n):
        snap = dataset[i]
        xs.append(snap.x.detach().cpu().numpy())
        ys.append(snap.y.detach().cpu().numpy().astype(int))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _train_rf(train_ds, test_ds, seed: int) -> tuple[np.ndarray, np.ndarray]:
    X_tr, y_tr = _flatten_snapshots(train_ds)
    X_te, y_te = _flatten_snapshots(test_ds)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        random_state=seed, n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    return y_te, rf.predict(X_te).astype(int)


def _train_mlp(train_ds, test_ds, seed: int, num_classes: int, epochs: int = 60) -> tuple[np.ndarray, np.ndarray]:
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
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
        Dense(128, activation="relu"), BatchNormalization(), Dropout(0.3),
        Dense(64, activation="relu"), BatchNormalization(), Dropout(0.3),
        Dense(32, activation="relu"), BatchNormalization(),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_tr, y_tr_1h, epochs=epochs, batch_size=32, verbose=0, validation_split=0.1)
    preds = np.argmax(model.predict(X_te, verbose=0), axis=1).astype(int)
    return y_te, preds


# ---------------------------------------------------------------------------
# McNemar computation
# ---------------------------------------------------------------------------

def _mcnemar_binary(preds_a: np.ndarray, preds_b: np.ndarray,
                    labels: np.ndarray) -> dict:
    """Classic McNemar test on paired correct/incorrect indicators.

    Contingency cells:
        a = both correct       b = A correct, B wrong
        c = A wrong, B correct d = both wrong

    Test statistic (with Yates continuity correction):
        chi2 = (|b - c| - 1)^2 / (b + c),  df = 1

    Returns a dict with b, c, chi2, p, and a plain-english direction of the
    disagreement so callers can label the significance direction correctly.
    """
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    denom = b + c
    if denom == 0:
        chi2 = 0.0
        p = 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / denom
        p = float(1.0 - chi2_dist.cdf(chi2, df=1))
    if b > c:
        direction = "A more correct"
    elif c > b:
        direction = "B more correct"
    else:
        direction = "tie"
    return {"b": b, "c": c, "chi2": chi2, "p": p, "direction": direction}


def _mcnemar_per_class(preds_a: np.ndarray, preds_b: np.ndarray,
                       labels: np.ndarray, num_classes: int) -> list[dict]:
    """Run binary McNemar restricted to samples of each true class."""
    results = []
    for cls in range(num_classes):
        mask = (labels == cls)
        if mask.sum() == 0:
            continue
        results.append({
            "class": cls,
            "n_samples": int(mask.sum()),
            **_mcnemar_binary(preds_a[mask], preds_b[mask], labels[mask]),
        })
    return results


def _bonferroni_hold_out(results: list[dict], m: int) -> list[dict]:
    """In-place add of Bonferroni-corrected p-values to a list of test dicts."""
    for r in results:
        r["p_bonf"] = min(1.0, r["p"] * m)
    return results


# ---------------------------------------------------------------------------
# LaTeX table formatting
# ---------------------------------------------------------------------------

def _fmt_p(p: float) -> str:
    """LaTeX-friendly p-value formatter with the usual small-value convention."""
    if p < 0.001:
        return r"$<$ 0.001"
    return f"{p:.3f}"


def _to_latex_table(pair_results: list[dict], caption: str, label: str) -> str:
    """Build the LaTeX table string. One row per comparison, both the
    correct-vs-incorrect summary and its three per-class Bonferroni-adjusted
    p-values live on the same row for compactness."""
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{" + caption + r"}",
        r"    \label{" + label + r"}",
        r"    \resizebox{\textwidth}{!}{%",
        r"        \begin{tabular}{@{}lccccccc@{}}",
        r"            \hline",
        r"            \textbf{Comparison (A vs B)} & \textbf{$b$} & \textbf{$c$} & "
        r"\textbf{$\chi^2$} & \textbf{$p$} & "
        r"\textbf{$p_{\text{class 0}}$} & \textbf{$p_{\text{class 1}}$} & "
        r"\textbf{$p_{\text{class 2}}$} \\ \hline",
    ]
    for r in pair_results:
        overall = r["overall"]
        per_class_ps = {rr["class"]: rr["p_bonf"] for rr in r["per_class"]}
        row = (
            f"            {r['name']} & "
            f"{overall['b']} & {overall['c']} & "
            f"{overall['chi2']:.3f} & {_fmt_p(overall['p'])} & "
            f"{_fmt_p(per_class_ps.get(0, 1.0))} & "
            f"{_fmt_p(per_class_ps.get(1, 1.0))} & "
            f"{_fmt_p(per_class_ps.get(2, 1.0))} \\\\"
        )
        lines.append(row)
    lines.extend([r"            \hline", r"        \end{tabular}%", r"    }", r"\end{table}"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data-setup helpers — must mirror the training pipeline exactly
# ---------------------------------------------------------------------------

def _rebuild_splits(base_path: Path, cfg: dict):
    """Recreate the exact outer + inner split that every GNN best-run used.

    Kept deterministic (no random seed dependence) so the baselines see the
    same 207-sample validation set the GNNs did.
    """
    data_path = base_path / cfg.get("data_path", "Dataset/TransientData/processedData.csv")
    df = pd.read_csv(data_path)
    dataset = create_temporal_dataset(
        df,
        seasonal_normalization=cfg.get("seasonal_normalization", True),
        add_lag_features=cfg.get("add_lag_features", True),
        n_classes=int(cfg.get("num_classes", 3)),
        # Leak-free preprocessing — trainval-only fit for scaler / seasonal
        # stats / tercile bin edges. Must mirror the outer split.
        test_holdout_ratio=float(cfg.get("test_holdout_ratio", 0.2)),
        stratified_split=bool(cfg.get("stratified_split", False)),
    )
    test_holdout = float(cfg.get("test_holdout_ratio", 0.2))
    # Match the outer split style used by the GNN pipeline: interleaved if
    # stratified_split is on, otherwise chronological.
    if bool(cfg.get("stratified_split", False)):
        trainval, _outer_test = interleaved_temporal_split(dataset, train_ratio=1.0 - test_holdout)
    else:
        trainval, _outer_test = temporal_signal_split(dataset, train_ratio=1.0 - test_holdout)

    inner_ratio = float(cfg.get("train_ratio", 0.8))
    if bool(cfg.get("stratified_split", False)):
        train_ds, val_ds = stratified_temporal_signal_split(
            trainval, train_ratio=inner_ratio,
            num_classes=int(cfg.get("num_classes", 3)),
        )
    else:
        train_ds, val_ds = temporal_signal_split(trainval, train_ratio=inner_ratio)
    return train_ds, val_ds


def _guess_best_run_cfg(models_dir: Path, df: pd.DataFrame) -> dict:
    """Use the params column of any best-run to seed the split reconstruction.
    All three GNN best-runs should share the split-relevant fields
    (seasonal_normalization, add_lag_features, test_holdout_ratio,
    stratified_split, train_ratio) — if they don't, the ValueError raised
    later by _load_gnn_predictions will make the misalignment obvious."""
    for model_type in ("SimpleGCN", "TemporalGCN_GRU", "TemporalAttentionGNN"):
        run_dir = _find_best_run_dir(models_dir, model_type, df)
        if run_dir is None:
            continue
        try:
            info = (run_dir.parent / "best_run_info.txt").read_text() if (run_dir.parent / "best_run_info.txt").exists() else ""
        except Exception:
            info = ""
        row = df[df["run_id"] == run_dir.name]
        if not row.empty and "params" in row.columns:
            try:
                params = json.loads(row.iloc[0]["params"])
                return params
            except Exception:
                pass
    return {}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(base_path: Path | str | None = None, seed: int = 42) -> Path:
    base_path = Path(base_path or Path(__file__).resolve().parents[3]).resolve()
    models_dir = base_path / "src" / "masters_research" / "artifacts" / "models"
    plots_dir = base_path / "src" / "masters_research" / "artifacts" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results_csv = models_dir / "grid_results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"grid_results.csv missing at {results_csv}")
    df = pd.read_csv(results_csv)

    # --- GNN predictions (no retrain, load from disk) ---
    gnn_preds = _load_gnn_predictions(models_dir, df)
    if not gnn_preds:
        raise RuntimeError(
            "No GNN test_probs.npz files were found. Retrain any of the GNN "
            "architectures with enable_explainability: true to generate them."
        )

    # Canonical shared labels for the validation set every GNN best-run used.
    canonical_labels = next(iter(gnn_preds.values()))[0]
    n_val = len(canonical_labels)
    num_classes = int(canonical_labels.max()) + 1
    print(f"Shared test/val set: {n_val} samples, {num_classes} classes.\n")

    # --- Baseline training on identical splits ---
    cfg = _guess_best_run_cfg(models_dir, df)
    # Fall back to sensible defaults if params were unparseable.
    cfg.setdefault("seasonal_normalization", True)
    cfg.setdefault("add_lag_features", True)
    cfg.setdefault("test_holdout_ratio", 0.2)
    cfg.setdefault("train_ratio", 0.8)
    cfg.setdefault("stratified_split", False)
    cfg.setdefault("num_classes", num_classes)
    cfg.setdefault("data_path", "Dataset/TransientData/processedData.csv")

    train_ds, val_ds = _rebuild_splits(base_path, cfg)
    n_val_local = val_ds.snapshot_count * val_ds[0].x.shape[0]
    if n_val_local != n_val:
        raise ValueError(
            f"Split reconstruction produced {n_val_local} validation samples but "
            f"the GNN best-runs used {n_val}. Check that "
            f"stratified_split / train_ratio / test_holdout_ratio in the "
            f"best-run's params match the reconstruction defaults."
        )

    print("Training RandomForest on the same trainval pool...")
    y_rf, rf_preds = _train_rf(train_ds, val_ds, seed=seed)
    print(f"RandomForest       accuracy = {(rf_preds == y_rf).mean():.4f}")

    print("Training Keras MLP on the same trainval pool...")
    y_mlp, mlp_preds = _train_mlp(train_ds, val_ds, seed=seed, num_classes=num_classes)
    print(f"KerasMLP           accuracy = {(mlp_preds == y_mlp).mean():.4f}")

    # Alignment guard: baseline labels must match the GNN canonical labels
    # sample-for-sample, otherwise the McNemar tests are meaningless.
    if not np.array_equal(y_rf, canonical_labels):
        raise ValueError(
            "RandomForest label vector does not match the GNN canonical labels — "
            "the reconstructed split differs from the one used by the GNN pipeline."
        )

    all_preds = {
        "RandomForest": rf_preds,
        "KerasMLP": mlp_preds,
        **{k: v[1] for k, v in gnn_preds.items()},
    }

    # Persist every model's predictions plus the shared labels so any
    # downstream analysis can reproduce the significance tests from disk.
    np.savez(
        plots_dir / "mcnemar_predictions.npz",
        labels=canonical_labels,
        **all_preds,
    )
    print(f"\nSaved aligned predictions to {plots_dir / 'mcnemar_predictions.npz'}\n")

    # --- Run the four thesis-critical comparisons ---
    comparisons = [
        ("RF vs KerasMLP", "RandomForest", "KerasMLP"),
        ("RF vs SimpleGCN", "RandomForest", "SimpleGCN"),
        ("RF vs TemporalAttentionGNN", "RandomForest", "TemporalAttentionGNN"),
        ("TemporalAttentionGNN vs SimpleGCN", "TemporalAttentionGNN", "SimpleGCN"),
    ]
    pair_results = []
    for name, a, b in comparisons:
        if a not in all_preds or b not in all_preds:
            print(f"Skipping {name}: missing predictions.")
            continue
        overall = _mcnemar_binary(all_preds[a], all_preds[b], canonical_labels)
        per_class = _bonferroni_hold_out(
            _mcnemar_per_class(all_preds[a], all_preds[b], canonical_labels, num_classes),
            m=num_classes,
        )
        pair_results.append({"name": name, "overall": overall, "per_class": per_class})

    # --- Console summary ---
    print("\n" + "=" * 72)
    print("McNemar significance results (Yates-corrected; per-class p Bonferroni-adjusted)")
    print("=" * 72)
    header = f"{'Comparison':40s}  {'b':>4s} {'c':>4s} {'chi2':>7s} {'p':>10s}"
    print(header)
    for r in pair_results:
        o = r["overall"]
        p_str = "<0.001" if o["p"] < 0.001 else f"{o['p']:.3f}"
        print(f"{r['name']:40s}  {o['b']:>4d} {o['c']:>4d} {o['chi2']:>7.3f} {p_str:>10s}")

    # --- LaTeX table ---
    tex = _to_latex_table(
        pair_results,
        caption="McNemar significance tests across the four thesis-critical "
                "pairwise comparisons. $b$ and $c$ are the discordant "
                "contingency cells (A correct \\& B wrong, and vice versa); "
                "$\\chi^2$ uses the Yates continuity correction with one "
                "degree of freedom. Per-class $p$-values are Bonferroni "
                "corrected for three simultaneous tests.",
        label="tab:mcnemar-tests",
    )
    tex_path = plots_dir / "mcnemar_table.tex"
    tex_path.write_text(tex)
    print(f"\nLaTeX table saved to {tex_path}\n")
    print(tex)
    return tex_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    main(project_root)
