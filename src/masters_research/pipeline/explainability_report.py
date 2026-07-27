"""Model-level aggregation of GNN explainability outputs.

The training pipeline produces per-node (GNNExplainer) and per-class
(Integrated Gradients) explanations for every run. This module rolls those
per-run artifacts up to the *architecture* level so each GNN model type gets
a single canonical set of explainability plots — averaged across every
explained node, class, and run of that architecture.

For each model type (SimpleGCN / TemporalGCN_GRU / TemporalGAT_GRU /
TemporalAttentionGNN), reads the raw arrays saved by pipeline/train.py:

  - explainability_gnnexplainer.npz    (feature_masks, edge_masks per run)
  - explainability_ig.npz              (attributions per run, shape [C, N, F])
  - test_probs.npz                     (labels + probs for ROC aggregation)

and emits five plots into models/<ModelType>/explainability/:

  - feature_importance.png    — mean GNNExplainer feature importance
                                (aggregated across nodes AND runs) with
                                ±1 std error bars.
  - edge_mask_hist.png        — pooled edge-mask distribution across all
                                explained nodes and runs of that model type.
  - ig_mean_attr.png          — mean IG attribution per feature, averaged
                                across classes, nodes, and runs.
  - ig_box.png                — feature-level attribution box plot pooling
                                every (class, node, run) attribution.
  - roc_curves.png            — one-vs-rest ROC per class with a mean line
                                across runs and a ±1 std band.

Standalone usage:
    python -m masters_research.pipeline.explainability_report
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


MODEL_TYPES = ["SimpleGCN", "TemporalGCN_GRU", "TemporalGAT_GRU", "TemporalAttentionGNN"]


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _resolve_models_dir(base_path: Path) -> Path:
    candidate = base_path / "src" / "masters_research" / "artifacts" / "models"
    if candidate.exists():
        return candidate
    alt = base_path / "models"
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Could not find models/ under {base_path}.")


def _iter_run_dirs(model_type_dir: Path) -> list[Path]:
    """Return every run subdirectory under models/<ModelType>/ (excludes
    reserved paths like 'per_class' or 'explainability')."""
    if not model_type_dir.exists():
        return []
    reserved = {"per_class", "explainability"}
    return sorted(
        p for p in model_type_dir.iterdir()
        if p.is_dir() and p.name not in reserved
    )


def _load_npz(path: Path) -> Optional[dict]:
    """Safe npz loader — returns None on missing file / read error."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            return {k: data[k] for k in data.files}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _collect_gnn_explanations(runs: list[Path]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Concatenate GNNExplainer feature masks and edge masks across every run
    of a model type.

    Returns:
        feature_masks_pooled  — shape (total_explained_nodes, num_features)
        edge_mask_values      — 1-D array of every edge-mask value observed
        feature_names         — matching feature-name list (best effort)
    """
    feat_stacks: list[np.ndarray] = []
    edge_values: list[np.ndarray] = []
    feature_names: list[str] = []

    for run_dir in runs:
        data = _load_npz(run_dir / "explainability_gnnexplainer.npz")
        if data is None:
            continue
        fm = data.get("feature_masks")
        em = data.get("edge_masks")
        fn = data.get("feature_names")
        if fn is not None and len(feature_names) == 0:
            feature_names = [str(n) for n in fn.tolist()]

        # Feature masks: (nodes, features) — may be object dtype if ragged;
        # skip anything that isn't a plain rectangular numeric block.
        if fm is not None and fm.dtype != object and fm.ndim == 2:
            feat_stacks.append(fm)
        # Edge masks are per-node 1-D arrays inside an object array; flatten
        # everything into a single 1-D pool for histogramming.
        if em is not None and em.size > 0:
            if em.dtype == object:
                for arr in em.tolist():
                    if arr is not None:
                        edge_values.append(np.asarray(arr).ravel())
            else:
                edge_values.append(np.asarray(em).ravel())

    feature_masks_pooled = (
        np.concatenate(feat_stacks, axis=0) if feat_stacks else np.zeros((0, 0))
    )
    edge_mask_values = np.concatenate(edge_values) if edge_values else np.zeros(0)
    return feature_masks_pooled, edge_mask_values, feature_names


def _collect_ig_attributions(runs: list[Path]) -> tuple[np.ndarray, list[str]]:
    """Stack IG attribution tensors across every run.

    Each run contributes a tensor of shape (C, N, F). We reshape each to
    (C*N, F) and concatenate along axis 0, so the result is a 2-D matrix
    where every row is one (class, node, run) attribution vector — ready to
    average or box-plot per feature.
    """
    rows: list[np.ndarray] = []
    feature_names: list[str] = []
    for run_dir in runs:
        data = _load_npz(run_dir / "explainability_ig.npz")
        if data is None:
            continue
        attr = data.get("attributions")
        fn = data.get("feature_names")
        if fn is not None and len(feature_names) == 0:
            feature_names = [str(n) for n in fn.tolist()]
        if attr is None or attr.ndim != 3:
            continue
        C, N, F = attr.shape
        rows.append(attr.reshape(C * N, F))
    if not rows:
        return np.zeros((0, 0)), feature_names
    # Some runs may have different num_classes if the config changed —
    # only concatenate rows that share the feature dimension of the first
    # collected run.
    feature_dim = rows[0].shape[1]
    filtered = [r for r in rows if r.shape[1] == feature_dim]
    return np.concatenate(filtered, axis=0), feature_names


def _collect_test_probs(runs: list[Path]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return [(labels, probs), ...] for every run that saved test_probs."""
    out = []
    for run_dir in runs:
        data = _load_npz(run_dir / "test_probs.npz")
        if data is None:
            continue
        labels = data.get("labels")
        probs = data.get("probs")
        if labels is None or probs is None:
            continue
        out.append((np.asarray(labels), np.asarray(probs)))
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_feature_importance(pooled: np.ndarray, feature_names: list[str], out_path: Path) -> None:
    """Bar chart of mean feature importance across every explained node and
    every run of the model type, with error bars showing ±1 std."""
    if pooled.size == 0:
        return
    means = pooled.mean(axis=0)
    stds = pooled.std(axis=0)
    labels = feature_names if len(feature_names) == means.shape[0] else [str(i) for i in range(means.shape[0])]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    xs = np.arange(len(means))
    ax.bar(xs, means, yerr=stds, capsize=3, color="#4c72b0", edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Feature importance (mean ± std)")
    ax.set_title(f"GNNExplainer feature importance — pooled across {pooled.shape[0]} node explanations")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_edge_mask_hist(values: np.ndarray, out_path: Path) -> None:
    """Histogram of every edge-mask value pooled across all nodes and runs."""
    if values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(values, bins=30, ax=ax, color="#dd8452", edgecolor="black", linewidth=0.4)
    ax.axvline(float(values.mean()), color="red", linestyle="--", linewidth=1,
               label=f"mean = {values.mean():.3f}")
    ax.set_xlabel("Edge mask value")
    ax.set_ylabel("Count")
    ax.set_title(f"GNNExplainer edge-mask distribution — pooled across {values.size} edges")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_ig_mean(pooled: np.ndarray, feature_names: list[str], out_path: Path) -> None:
    """Mean IG attribution per feature, averaged across classes/nodes/runs."""
    if pooled.size == 0:
        return
    means = pooled.mean(axis=0)
    stds = pooled.std(axis=0)
    labels = feature_names if len(feature_names) == means.shape[0] else [str(i) for i in range(means.shape[0])]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    xs = np.arange(len(means))
    colors = ["#55a868" if m >= 0 else "#c44e52" for m in means]
    ax.bar(xs, means, yerr=stds, capsize=3, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("IG attribution (mean ± std)")
    ax.set_title(f"Integrated Gradients attribution — pooled across {pooled.shape[0]} (class,node,run) triples")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_ig_box(pooled: np.ndarray, feature_names: list[str], out_path: Path) -> None:
    """Per-feature box plot of IG attributions across classes/nodes/runs."""
    if pooled.size == 0:
        return
    F = pooled.shape[1]
    labels = feature_names if len(feature_names) == F else [str(i) for i in range(F)]

    long = pd.DataFrame(pooled, columns=labels).melt(var_name="feature", value_name="attribution")
    fig, ax = plt.subplots(figsize=(max(9, 0.55 * F), 4.5))
    sns.boxplot(data=long, x="feature", y="attribution", ax=ax, color="#4c72b0",
                fliersize=1.5, linewidth=0.7)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("IG attribution")
    ax.set_xlabel("Feature")
    ax.set_title(f"IG attribution distribution per feature — {pooled.shape[0]} attributions pooled")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_aggregated_roc(runs_data: list[tuple[np.ndarray, np.ndarray]], out_path: Path) -> None:
    """One-vs-rest ROC per class with a mean line across runs and ±1 std band.

    Uses a common FPR grid so per-run ROCs can be pointwise averaged.
    """
    if not runs_data:
        return

    # Infer number of classes from probs shape.
    num_classes = runs_data[0][1].shape[1]
    fpr_grid = np.linspace(0.0, 1.0, 201)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    palette = plt.cm.tab10(np.linspace(0, 1, max(3, num_classes)))

    for cls in range(num_classes):
        tprs_interp: list[np.ndarray] = []
        aucs: list[float] = []
        for labels, probs in runs_data:
            if probs.shape[1] != num_classes:
                continue
            try:
                y_bin = label_binarize(labels, classes=list(range(num_classes)))
                if y_bin.shape[1] != num_classes:
                    # Some sklearn versions collapse binary case — skip.
                    continue
                fpr, tpr, _ = roc_curve(y_bin[:, cls], probs[:, cls])
                tpr_at_grid = np.interp(fpr_grid, fpr, tpr)
                tpr_at_grid[0] = 0.0
                tprs_interp.append(tpr_at_grid)
                aucs.append(auc(fpr, tpr))
            except Exception:
                continue
        if not tprs_interp:
            continue
        tprs = np.stack(tprs_interp, axis=0)
        mean_tpr = tprs.mean(axis=0)
        std_tpr = tprs.std(axis=0)
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))
        color = palette[cls]
        ax.plot(fpr_grid, mean_tpr, color=color, linewidth=2,
                label=f"Class {cls} (AUC = {mean_auc:.3f} ± {std_auc:.3f})")
        ax.fill_between(fpr_grid,
                        np.clip(mean_tpr - std_tpr, 0, 1),
                        np.clip(mean_tpr + std_tpr, 0, 1),
                        color=color, alpha=0.15)

    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curves (mean ± std across {len(runs_data)} runs)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(base_path: Path | str | None = None) -> Path:
    base_path = Path(base_path or Path(__file__).resolve().parents[3]).resolve()
    models_dir = _resolve_models_dir(base_path)

    for model_type in MODEL_TYPES:
        model_type_dir = models_dir / model_type
        runs = _iter_run_dirs(model_type_dir)
        if not runs:
            continue

        # GNNExplainer aggregations
        feat_pooled, edge_values, feat_names_gnn = _collect_gnn_explanations(runs)
        # IG aggregations
        ig_pooled, feat_names_ig = _collect_ig_attributions(runs)
        # ROC data
        probs_data = _collect_test_probs(runs)

        if (feat_pooled.size == 0 and ig_pooled.size == 0 and not probs_data):
            print(f"[{model_type}] no explainability arrays found in any run "
                  f"(older runs may have been trained before array saving was added).")
            continue

        feature_names = feat_names_gnn or feat_names_ig
        out_dir = model_type_dir / "explainability"
        out_dir.mkdir(parents=True, exist_ok=True)

        _plot_feature_importance(feat_pooled, feature_names, out_dir / "feature_importance.png")
        _plot_edge_mask_hist(edge_values, out_dir / "edge_mask_hist.png")
        _plot_ig_mean(ig_pooled, feature_names, out_dir / "ig_mean_attr.png")
        _plot_ig_box(ig_pooled, feature_names, out_dir / "ig_box.png")
        _plot_aggregated_roc(probs_data, out_dir / "roc_curves.png")

        print(f"[{model_type}] model-level explainability saved to {out_dir} "
              f"(GNNExplainer nodes pooled={feat_pooled.shape[0] if feat_pooled.size else 0}, "
              f"IG rows pooled={ig_pooled.shape[0] if ig_pooled.size else 0}, "
              f"ROC runs={len(probs_data)})")

    return models_dir


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    main(project_root)
