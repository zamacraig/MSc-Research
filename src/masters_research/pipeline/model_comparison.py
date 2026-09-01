"""Cross-family comparison chart: baseline vs GNN.

Reads the five classification_report.txt files already emitted by the training
and baseline pipelines and produces two grouped bar charts:

  1. plots/model_comparison_metrics.png    — accuracy, weighted recall,
                                             weighted F1 per model (the answer
                                             to "how do these models stack up
                                             on the headline metrics").
  2. plots/model_comparison_per_class_f1.png — per-class F1 per model (the
                                              answer to "where does each
                                              model win or lose").

Family colouring makes the baseline-vs-GNN separation legible at a glance.

Inputs (read from models/ tree — nothing else needed):
  - models/RandomForest/RandomForest_classification_report.txt
  - models/Keras/Keras_classification_report.txt
  - models/<GNNArchitecture>/classification_report.txt    (from model_type_report)

Standalone usage:
    python -m masters_research.pipeline.model_comparison
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Model registry with (display_name, family, path-in-models-tree).
# Order here is the plotting order (left -> right).
MODELS: list[tuple[str, str, str]] = [
    # Baselines are now trained inside the main pipeline (pipeline/baselines_runner.py),
    # so their reports live under the same models/<ModelType>/ layout as the GNNs.
    ("RandomForest",         "baseline", "RandomForest/classification_report.txt"),
    ("KerasMLP",             "baseline", "KerasMLP/classification_report.txt"),
    ("SimpleGCN",            "gnn",      "SimpleGCN/classification_report.txt"),
    ("TemporalGCN_GRU",      "gnn",      "TemporalGCN_GRU/classification_report.txt"),
    ("TemporalAttentionGNN", "gnn",      "TemporalAttentionGNN/classification_report.txt"),
]

FAMILY_COLOURS = {"baseline": "#5a7dbe", "gnn": "#c65b5b"}


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

def _parse_classification_report(path: Path) -> Optional[dict]:
    """Parse a sklearn classification_report text file into a flat dict.

    Returns keys:
        accuracy, weighted_precision, weighted_recall, weighted_f1,
        macro_precision, macro_recall, macro_f1,
        class_0_precision, class_0_recall, class_0_f1_score,
        class_1_precision, class_1_recall, class_1_f1_score,
        class_2_precision, class_2_recall, class_2_f1_score,
    or None if the file is unreadable / malformed.
    """
    if not path.exists():
        return None
    text = path.read_text().strip().splitlines()

    result: dict = {}
    # Per-class rows: `<label>  <prec>  <rec>  <f1>  <support>`
    row_pat = re.compile(
        r"^\s*(\S+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]+)\s*$"
    )
    acc_pat = re.compile(r"^\s*accuracy\s+([0-9]*\.?[0-9]+)\s+([0-9]+)\s*$")
    avg_pat = re.compile(
        r"^\s*(macro avg|weighted avg)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]+)\s*$"
    )

    for line in text:
        m = row_pat.match(line)
        if m:
            lbl, prec, rec, f1, sup = m.groups()
            # Only capture rows whose label is an integer class id; skip the
            # "macro avg" / "weighted avg" / "accuracy" summary lines which
            # have their own patterns below.
            if lbl.lstrip("-").isdigit():
                result[f"class_{lbl}_precision"] = float(prec)
                result[f"class_{lbl}_recall"] = float(rec)
                result[f"class_{lbl}_f1_score"] = float(f1)
                result[f"class_{lbl}_support"] = int(sup)
            continue
        m_acc = acc_pat.match(line)
        if m_acc:
            result["accuracy"] = float(m_acc.group(1))
            continue
        m_avg = avg_pat.match(line)
        if m_avg:
            which, prec, rec, f1, _ = m_avg.groups()
            prefix = "macro" if which == "macro avg" else "weighted"
            result[f"{prefix}_precision"] = float(prec)
            result[f"{prefix}_recall"] = float(rec)
            result[f"{prefix}_f1"] = float(f1)
    return result if result else None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _resolve_models_dir(base_path: Path) -> Path:
    """Locate models/ regardless of whether we're called from repo root."""
    candidate = base_path / "src" / "masters_research" / "artifacts" / "models"
    if candidate.exists():
        return candidate
    alt = base_path / "models"
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Could not find models/ under {base_path}.")


def _auto_ylim(values, pad: float = 0.06) -> tuple[float, float]:
    """Pick a y-range that frames the data with a little headroom.

    Always anchors below the smallest bar so nothing is clipped, and never
    exceeds 1.0 (these are all bounded [0,1] metrics). Falls back to the full
    range if the values are degenerate.
    """
    v = np.asarray([x for x in np.asarray(values, dtype=float) if np.isfinite(x)])
    if v.size == 0:
        return 0.0, 1.0
    lo = max(0.0, float(v.min()) - pad)
    hi = min(1.0, float(v.max()) + pad)
    if hi - lo < 0.05:            # degenerate spread
        return max(0.0, lo - 0.05), min(1.0, hi + 0.05)
    return lo, hi


def _annotate_bars(ax: plt.Axes, fmt: str = "{:.4f}") -> None:
    """Print each bar's numeric value on top of the bar.

    Default precision is 4 decimal places so headline metrics that agree to
    two decimals (e.g. RF vs Keras at 0.90 vs 0.90) are still distinguishable
    on the chart itself without needing the underlying CSV.
    """
    for patch in ax.patches:
        h = patch.get_height()
        if np.isnan(h) or h <= 0:
            continue
        ax.annotate(fmt.format(h),
                    xy=(patch.get_x() + patch.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5)


def _plot_headline_metrics(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart of accuracy / weighted recall / weighted F1 per model."""
    metrics = [("accuracy", "Accuracy"),
               ("weighted_recall", "Recall (weighted)"),
               ("weighted_f1", "F1 (weighted)")]
    metric_ids = [m[0] for m in metrics]
    plot_df = df.melt(
        id_vars=["model", "family"],
        value_vars=metric_ids, var_name="metric", value_name="value",
    )
    label_map = dict(metrics)
    plot_df["metric_label"] = plot_df["metric"].map(label_map)
    metric_order = [m[1] for m in metrics]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    # 3 shades per model, keyed to metric — using a sequential palette keeps
    # the visual density down (a categorical palette per model + per metric
    # would be too busy). Grey-scale by metric works and keeps focus on
    # cross-model differences.
    metric_palette = {"Accuracy": "#4c72b0", "Recall (weighted)": "#55a868",
                      "F1 (weighted)": "#c44e52"}
    sns.barplot(
        data=plot_df, x="model", y="value", hue="metric_label",
        hue_order=metric_order, palette=metric_palette, ax=ax,
    )

    # Colour the model tick labels by family so the baseline/GNN split is
    # visible without needing a separate legend.
    for tick_label, (_, row) in zip(ax.get_xticklabels(), df.iterrows()):
        tick_label.set_color(FAMILY_COLOURS[row["family"]])
        tick_label.set_fontweight("bold")

    # Zoom the y-axis to the range actually populated by the results so small
    # between-model differences stay legible. Derived from the data rather than
    # hard-coded: a fixed floor silently renders every bar invisible whenever
    # scores fall below it.
    _lo, _hi = _auto_ylim(plot_df["value"])
    ax.set_ylim(_lo, _hi)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("Model comparison — headline metrics (weighted averages)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Metric", loc="lower left", framealpha=0.9)
    _annotate_bars(ax)

    # Family legend as an inset — small text describing the tick-label colour code.
    ax.text(0.99, -0.18,
            "Model labels: blue = baseline (RF, MLP)   red = graph neural network",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, style="italic", color="#333333")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_per_class_f1(df: pd.DataFrame, out_path: Path) -> None:
    """Companion chart: per-class F1 by model, exposing where each model wins."""
    cols = [c for c in ("class_0_f1_score", "class_1_f1_score", "class_2_f1_score") if c in df.columns]
    if not cols:
        return
    long = df.melt(id_vars=["model", "family"], value_vars=cols,
                   var_name="class", value_name="f1")
    long["class"] = long["class"].str.extract(r"class_(\d+)_")[0]

    fig, ax = plt.subplots(figsize=(11, 5))
    class_palette = {"0": "#4c72b0", "1": "#dd8452", "2": "#55a868"}
    sns.barplot(
        data=long, x="model", y="f1", hue="class",
        hue_order=["0", "1", "2"], palette=class_palette, ax=ax,
    )

    for tick_label, (_, row) in zip(ax.get_xticklabels(), df.iterrows()):
        tick_label.set_color(FAMILY_COLOURS[row["family"]])
        tick_label.set_fontweight("bold")

    _lo, _hi = _auto_ylim(long["f1"])
    ax.set_ylim(_lo, _hi)
    ax.set_ylabel("F1-score")
    ax.set_xlabel("")
    ax.set_title("Per-class F1 by model — where each family wins or loses")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Class (0=low, 1=medium, 2=high)", loc="lower left", framealpha=0.9)
    _annotate_bars(ax)

    ax.text(0.99, -0.18,
            "Model labels: blue = baseline (RF, MLP)   red = graph neural network",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, style="italic", color="#333333")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(base_path: Path | str | None = None) -> pd.DataFrame:
    base_path = Path(base_path or Path(__file__).resolve().parents[3]).resolve()
    models_dir = _resolve_models_dir(base_path)

    rows = []
    missing = []
    for display, family, rel_path in MODELS:
        report_path = models_dir / rel_path
        parsed = _parse_classification_report(report_path)
        if parsed is None:
            missing.append(display)
            continue
        parsed["model"] = display
        parsed["family"] = family
        rows.append(parsed)

    if missing:
        print(f"Note: no classification report found for {', '.join(missing)} "
              f"(skipping — check {models_dir}).")

    if not rows:
        raise RuntimeError("No model reports were found — nothing to plot.")

    df = pd.DataFrame(rows)
    # Enforce plotting order matching the MODELS registry
    order = [m for m, _, _ in MODELS if m in df["model"].values]
    df["model"] = pd.Categorical(df["model"], categories=order, ordered=True)
    df = df.sort_values("model").reset_index(drop=True)

    plots_dir = base_path / "src" / "masters_research" / "artifacts" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_out = plots_dir / "model_comparison.csv"
    df.to_csv(csv_out, index=False)

    _plot_headline_metrics(df, plots_dir / "model_comparison_metrics.png")
    _plot_per_class_f1(df, plots_dir / "model_comparison_per_class_f1.png")

    print(f"Saved comparison outputs:")
    print(f"  {plots_dir / 'model_comparison_metrics.png'}")
    print(f"  {plots_dir / 'model_comparison_per_class_f1.png'}")
    print(f"  {csv_out}")
    print("\nHeadline metrics table:")
    show_cols = ["model", "family", "accuracy", "weighted_recall", "weighted_f1"]
    print(df[show_cols].to_string(index=False))
    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    main(project_root)
