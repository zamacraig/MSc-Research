"""Per-class breakdown visuals for the completed grid of runs.

Complements pipeline/model_type_report.py, which produces best-run classification
reports and cross-model comparisons. This module fills three specific gaps:

  1. Row-normalised confusion matrix heatmaps  — every cell reads as "of true class
     X, this fraction was predicted as Y", i.e. per-class recall broken out by
     which wrong class absorbed the misclassifications. Much easier to read for
     boundary-class failures than raw counts.

  2. Per-class metric boxplots across seeds   — for each architecture, show the
     distribution of per-class precision / recall / F1 across every run of that
     type. Seed variance is the main story with a small dataset; users need to
     see it, not just the winning point.

  3. Per-class F1 heatmap (runs × classes)     — one row per run, one column per
     class, coloured by F1. Reveals whether some runs excel at some classes but
     not others (specialisation) or are uniformly strong/weak (stability).

Plus a cross-model comparison:

  4. Per-class F1 grouped boxplot              — architectures on the x-axis,
     hue = class, showing full seed distribution of per-class F1. The single
     plot from which architecture-vs-architecture claims per class can be made.

Inputs (all already emitted by pipeline/train.py):
  - <models_dir>/grid_results.csv                              (master index)
  - <models_dir>/<model_type>/<run_id>/<run_id>_per_class_metrics.csv
  - <models_dir>/<model_type>/<run_id>/<run_id>_confusion_matrix.csv

Outputs:
  - <models_dir>/<model_type>/per_class/*.png                  (per-model visuals)
  - <models_dir>/model_type_reports/per_class/*.png            (cross-model visuals)

Standalone usage:
    python -m masters_research.pipeline.per_class_report
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


MODEL_TYPE_ORDER = ["RandomForest", "KerasMLP",
                    "SimpleGCN", "TemporalGCN_GRU", "TemporalGAT_GRU", "TemporalAttentionGNN"]


# ---------------------------------------------------------------------------
# Loading & normalisation helpers
# ---------------------------------------------------------------------------

def _parse_params(raw: Any) -> dict:
    try:
        return json.loads(raw) if isinstance(raw, str) else dict(raw)
    except Exception:
        return {}


BASELINE_TYPES = {"RandomForest", "KerasMLP"}


def _classify_run(params: dict) -> str:
    """Same mapping as pipeline/train.py::_model_type_from_cfg."""
    model_type = params.get("model_type")
    # Non-graph baselines record their display name directly; the flag-based
    # branch order below only applies to the GNN family.
    if model_type in BASELINE_TYPES:
        return model_type
    if model_type == "simple_gcn" or bool(params.get("use_simple_gcn", False)):
        return "SimpleGCN"
    if model_type == "temporal_attention" or bool(params.get("use_temporal_attention", False)):
        return "TemporalAttentionGNN"
    if bool(params.get("use_attention", False)):
        return "TemporalGAT_GRU"
    return "TemporalGCN_GRU"


def _resolve_results_csv(base_path: Path) -> Path:
    candidate = base_path / "src" / "masters_research" / "artifacts" / "models" / "grid_results.csv"
    if candidate.exists():
        return candidate
    alt = base_path / "models" / "grid_results.csv"
    if alt.exists():
        return alt
    raise FileNotFoundError("grid_results.csv not found in artifacts/models nor models/.")


def _resolve_run_dir(row: pd.Series, models_dir: Path, model_type: str) -> Optional[Path]:
    """Locate a run's on-disk directory using (in order) the explicit run_models_dir
    column, then the nested models/<model_type>/<run_id>/ layout, then legacy flat."""
    run_id = str(row["run_id"])
    run_dir_str = str(row.get("run_models_dir", "") or "").strip()
    if run_dir_str:
        p = Path(run_dir_str)
        if p.exists():
            return p
    for candidate in (models_dir / model_type / run_id, models_dir / run_id):
        if candidate.exists():
            return candidate
    return None


def _classify_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a `_model_type` column to grid_results, preferring the explicit column
    (new schema) and falling back to reconstructing from `params`."""
    out = df.copy()
    if "model_type" in out.columns and out["model_type"].notna().any():
        out["_model_type"] = out["model_type"].fillna(
            out.get("params", pd.Series([""] * len(out))).apply(
                lambda raw: _classify_run(_parse_params(raw))
            )
        )
    else:
        out["_model_type"] = out["params"].apply(lambda raw: _classify_run(_parse_params(raw)))
    return out


def _load_all_per_class(df: pd.DataFrame, models_dir: Path) -> pd.DataFrame:
    """Read every run's per_class_metrics.csv and stack into a long-format frame
    with columns: model_type, run_id, class, precision, recall, f1, support, seed."""
    rows = []
    for _, r in df.iterrows():
        mtype = r["_model_type"]
        run_dir = _resolve_run_dir(r, models_dir, mtype)
        if run_dir is None:
            continue
        pc_path = run_dir / f"{r['run_id']}_per_class_metrics.csv"
        if not pc_path.exists():
            continue
        pc = pd.read_csv(pc_path)
        # Try to extract seed from params for grouping / colouring
        params = _parse_params(r.get("params", ""))
        seed = params.get("seed", None)
        for _, cr in pc.iterrows():
            rows.append({
                "model_type": mtype,
                "run_id": r["run_id"],
                "seed": seed,
                "class": int(cr["class"]),
                "precision": float(cr["precision"]),
                "recall": float(cr["recall"]),
                "f1": float(cr["f1"]),
                "support": int(cr["support"]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-model visuals
# ---------------------------------------------------------------------------

def _plot_normalised_confusion(cm_csv: Path, run_id: str, out_path: Path) -> None:
    """Row-normalised confusion matrix — each row sums to 1.0. Cell (i,j) = fraction
    of true-class-i samples that were predicted as class j."""
    cm_df = pd.read_csv(cm_csv, index_col=0)
    classes = list(cm_df.index)
    cm = cm_df.values.astype(float)
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    cm_norm = cm / row_sum

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        vmin=0.0, vmax=1.0, cbar_kws={"label": "recall"},
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Row-normalised Confusion Matrix\n(best run: {run_id})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_per_class_boxplot(model_pc: pd.DataFrame, model_type: str, out_path: Path) -> None:
    """Box+strip plot of per-class precision/recall/F1 distribution across seeds."""
    if model_pc.empty:
        return
    long = model_pc.melt(
        id_vars=["run_id", "seed", "class"],
        value_vars=["precision", "recall", "f1"],
        var_name="metric", value_name="value",
    )
    n_runs = model_pc["run_id"].nunique()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=long, x="class", y="value", hue="metric",
        showcaps=True, fliersize=0, ax=ax, palette="Set2",
    )
    sns.stripplot(
        data=long, x="class", y="value", hue="metric",
        dodge=True, alpha=0.55, size=4, ax=ax, palette="Set2",
        edgecolor="black", linewidth=0.3,
    )
    # Deduplicate legend (boxplot + stripplot both add handles).
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], labels[:3], title="metric", loc="lower left")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{model_type} — per-class metric distribution across {n_runs} runs")
    ax.set_ylabel("Score")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_per_class_heatmap(model_pc: pd.DataFrame, model_type: str, out_path: Path) -> None:
    """Heatmap of per-class F1 with rows=runs (sorted by mean F1), cols=classes.

    Reveals whether runs specialise (high F1 on one class, low on another) or are
    uniformly good/bad. Also serves as a quick visual scan for run outliers.
    """
    if model_pc.empty:
        return
    pivot = model_pc.pivot_table(index="run_id", columns="class", values="f1")
    pivot = pivot.assign(_mean=pivot.mean(axis=1)).sort_values("_mean", ascending=False).drop(columns="_mean")
    if pivot.empty:
        return

    height = max(3.5, 0.35 * len(pivot))
    fig, ax = plt.subplots(figsize=(6.5, height))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="YlGnBu",
        vmin=0.0, vmax=1.0, cbar_kws={"label": "F1"}, ax=ax,
    )
    ax.set_title(f"{model_type} — per-class F1 across runs (rows sorted by mean F1)")
    ax.set_xlabel("Class")
    ax.set_ylabel("run_id")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-model visuals
# ---------------------------------------------------------------------------

def _plot_cross_model_grouped_boxplot(all_pc: pd.DataFrame, out_path: Path) -> None:
    """Grouped box+strip: x=class, hue=model_type, y=F1 across all seeds.

    The single figure that summarises "which architecture does best on which class,
    with what variance". Directly answers the thesis's central comparison question.
    """
    if all_pc.empty:
        return

    present = [m for m in MODEL_TYPE_ORDER if m in all_pc["model_type"].unique()]
    fig, ax = plt.subplots(figsize=(max(8, len(present) * 2.5), 5))
    sns.boxplot(
        data=all_pc, x="class", y="f1", hue="model_type",
        hue_order=present, showcaps=True, fliersize=0, ax=ax, palette="tab10",
    )
    sns.stripplot(
        data=all_pc, x="class", y="f1", hue="model_type",
        hue_order=present, dodge=True, alpha=0.55, size=4, ax=ax,
        palette="tab10", edgecolor="black", linewidth=0.3,
    )
    handles, labels = ax.get_legend_handles_labels()
    n = len(present)
    ax.legend(handles[:n], labels[:n], title="architecture", loc="lower left")
    ax.set_ylim(0, 1.02)
    ax.set_title("Per-class F1 by architecture (all seeds shown)")
    ax.set_ylabel("F1")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_cross_model_metric_heatmap(all_pc: pd.DataFrame, out_path: Path) -> None:
    """Heatmap: rows = model_type, columns = (class, metric), values = median across
    seeds. Compact "all the numbers on one page" summary."""
    if all_pc.empty:
        return
    med = (
        all_pc.groupby(["model_type", "class"])[["precision", "recall", "f1"]]
        .median().reset_index()
    )
    med["col"] = med["class"].astype(str) + "\n" + "f1"
    # Build a wide frame with (model_type × [(class, precision), (class, recall), (class, f1)])
    wide_rows = []
    for mtype, grp in med.groupby("model_type"):
        row = {"model_type": mtype}
        for _, r in grp.iterrows():
            row[f"c{int(r['class'])}·prec"] = r["precision"]
            row[f"c{int(r['class'])}·rec"]  = r["recall"]
            row[f"c{int(r['class'])}·f1"]   = r["f1"]
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows).set_index("model_type")
    # Order rows in canonical model order and columns by class then metric
    wide = wide.reindex([m for m in MODEL_TYPE_ORDER if m in wide.index])
    col_order = sorted(
        wide.columns,
        key=lambda c: (int(c.split("c")[1].split("·")[0]), ["prec", "rec", "f1"].index(c.split("·")[1])),
    )
    wide = wide[col_order]

    fig, ax = plt.subplots(figsize=(max(8, 0.85 * len(col_order)), 0.9 + 0.5 * len(wide)))
    sns.heatmap(
        wide, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=0.0, vmax=1.0, cbar_kws={"label": "score"}, ax=ax,
    )
    ax.set_title("Per-class metrics by architecture (median across seeds)")
    ax.set_xlabel("class · metric")
    ax.set_ylabel("architecture")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Best-run selection (mirrors model_type_report._best_run_per_type)
# ---------------------------------------------------------------------------

def _best_run_per_type(df: pd.DataFrame) -> dict[str, pd.Series]:
    df = df.copy()
    score = pd.to_numeric(df.get("final_test_f1", pd.Series([np.nan] * len(df))), errors="coerce")
    score = score.fillna(pd.to_numeric(df.get("best_f1", pd.Series([np.nan] * len(df))), errors="coerce"))
    df["_score"] = score
    best: dict[str, pd.Series] = {}
    for label, group in df.groupby("_model_type"):
        group = group.dropna(subset=["_score"])
        if group.empty:
            continue
        best[label] = group.sort_values("_score", ascending=False).iloc[0]
    return best


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(base_path: Path | str) -> Path:
    base_path = Path(base_path).resolve()
    results_csv = _resolve_results_csv(base_path)
    models_dir = results_csv.parent

    df = _classify_frame(pd.read_csv(results_csv))
    all_pc = _load_all_per_class(df, models_dir)
    if all_pc.empty:
        print("per_class_report: no per_class_metrics.csv files found under any run directory.")
        return models_dir

    best_rows = _best_run_per_type(df)

    # ---- Per-model outputs ----
    for model_type in MODEL_TYPE_ORDER:
        model_pc = all_pc[all_pc["model_type"] == model_type]
        if model_pc.empty:
            continue

        per_model_dir = models_dir / model_type / "per_class"
        per_model_dir.mkdir(parents=True, exist_ok=True)

        _plot_per_class_boxplot(
            model_pc, model_type,
            per_model_dir / "per_class_metric_boxplot.png",
        )
        _plot_per_class_heatmap(
            model_pc, model_type,
            per_model_dir / "per_class_f1_heatmap.png",
        )

        best = best_rows.get(model_type)
        if best is not None:
            run_dir = _resolve_run_dir(best, models_dir, model_type)
            if run_dir is not None:
                cm_csv = run_dir / f"{best['run_id']}_confusion_matrix.csv"
                if cm_csv.exists():
                    _plot_normalised_confusion(
                        cm_csv, str(best["run_id"]),
                        per_model_dir / "confusion_matrix_normalised.png",
                    )
        print(f"[{model_type}] per-class visuals -> {per_model_dir}")

    # ---- Cross-model outputs ----
    cross_dir = models_dir / "model_type_reports" / "per_class"
    cross_dir.mkdir(parents=True, exist_ok=True)
    _plot_cross_model_grouped_boxplot(all_pc, cross_dir / "per_class_f1_by_architecture.png")
    _plot_cross_model_metric_heatmap(all_pc, cross_dir / "per_class_metric_heatmap.png")

    # Also emit the long-format frame as CSV so downstream analysis (or the thesis
    # results chapter) can rebuild any plot without re-parsing the run directories.
    all_pc.to_csv(cross_dir / "per_class_metrics_long.csv", index=False)

    print(f"Cross-model per-class visuals -> {cross_dir}")
    return cross_dir


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    main(project_root)
