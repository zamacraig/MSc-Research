"""Compare the 4 GNN architectures and emit a classification report for each.

The 4 architectures (see masters_research.models.__init__ and
pipeline/train.py::_train_one_fold's branch order) are distinguished purely by
hyperparameter flags, not a single explicit field:

    SimpleGCN          - use_simple_gcn=True (or model_type == "simple_gcn")
    TemporalAttentionGNN - use_temporal_attention=True (or model_type == "temporal_attention")
    TemporalGAT_GRU    - use_attention=True (GAT spatial conv + GRU)
    TemporalGCN_GRU    - none of the above (plain GCN spatial conv + GRU)

Every run launched via pipeline/train.py appends a row (with its full
hyperparameter dict) to artifacts/models/grid_results.csv, and writes a
confusion matrix CSV under artifacts/models/<run_id>/. Counts in that matrix
exactly determine the per-sample (true, pred) pairs, so this script
reconstructs them and feeds sklearn's classification_report directly -
no need to reload checkpoints or re-run inference.
"""
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread
from sklearn.metrics import confusion_matrix

from masters_research.pipeline.evaluate_best import (
    _save_cm,
    _save_classification_report,
    _save_classification_report_text,
)
from masters_research.pipeline.compare_reports import (
    _parse_classification_report_text,
    _grouped_bars_per_class,
    _radar_chart_f1,
    _averages_comparison,
)

MODEL_TYPE_ORDER = ["SimpleGCN", "TemporalGCN_GRU", "TemporalGAT_GRU", "TemporalAttentionGNN"]


def _parse_params(raw: Any) -> dict:
    try:
        return json.loads(raw) if isinstance(raw, str) else dict(raw)
    except Exception:
        return {}


def _classify_run(params: dict) -> str:
    """Map a run's hyperparameters to one of the 4 architectures, mirroring the
    branch order in pipeline/train.py::_train_one_fold."""
    model_type = params.get("model_type")
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


def _best_run_per_type(df: pd.DataFrame) -> dict[str, pd.Series]:
    df = df.copy()
    df["_model_type"] = df["params"].apply(lambda raw: _classify_run(_parse_params(raw)))
    df["_score"] = pd.to_numeric(df["final_test_f1"], errors="coerce")
    df["_score"] = df["_score"].fillna(pd.to_numeric(df["best_f1"], errors="coerce"))

    best: dict[str, pd.Series] = {}
    for label, group in df.groupby("_model_type"):
        group = group.dropna(subset=["_score"])
        if group.empty:
            continue
        best[label] = group.sort_values("_score", ascending=False).iloc[0]
    return best


def _reconstruct_labels_preds(cm_csv: Path) -> tuple[np.ndarray, np.ndarray, list]:
    cm_df = pd.read_csv(cm_csv, index_col=0)
    classes = [int(c) for c in cm_df.index]
    y_true, y_pred = [], []
    for true_c in cm_df.index:
        for pred_c in cm_df.columns:
            count = int(cm_df.loc[true_c, pred_c])
            if count:
                y_true.extend([int(true_c)] * count)
                y_pred.extend([int(pred_c)] * count)
    return np.array(y_true), np.array(y_pred), classes


def _accuracy_f1_bar_chart(summary_metrics: dict[str, dict], out_path: Path) -> None:
    """Grouped bar chart of accuracy/macro-F1/weighted-F1 per model type."""
    plot_df = pd.DataFrame(
        [
            {"model_type": label, "metric": "accuracy", "value": s.get("accuracy", np.nan)}
            for label, s in summary_metrics.items() if s
        ]
        + [
            {"model_type": label, "metric": "macro f1", "value": s.get("macro_f1", np.nan)}
            for label, s in summary_metrics.items() if s
        ]
        + [
            {"model_type": label, "metric": "weighted f1", "value": s.get("weighted_f1", np.nan)}
            for label, s in summary_metrics.items() if s
        ]
    )
    if plot_df.empty:
        return

    plt.figure(figsize=(max(8, len(summary_metrics) * 2.2), 5))
    sns.barplot(data=plot_df, x="model_type", y="value", hue="metric")
    plt.ylim(0, 1)
    plt.title("Accuracy & F1 by Model Type")
    plt.xlabel("Model Type")
    plt.ylabel("Score")
    plt.xticks(rotation=15, ha="right")
    plt.legend(title="Metric")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _per_class_breakdown(class_metrics: dict[str, Optional[pd.DataFrame]]) -> pd.DataFrame:
    """Long-format table of precision/recall/f1/support per (model_type, class)."""
    rows = []
    for label in MODEL_TYPE_ORDER:
        df = class_metrics.get(label)
        if df is None or df.empty:
            continue
        for cls, r in df.iterrows():
            rows.append(
                {
                    "model_type": label,
                    "class": cls,
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1-score": r["f1-score"],
                    "support": r["support"],
                }
            )
    return pd.DataFrame(rows)


def _confusion_side_by_side(cm_pngs: dict[str, Path], out_path: Path) -> None:
    imgs = [(name, imread(str(p))) for name, p in cm_pngs.items() if p.exists()]
    if not imgs:
        return
    fig, axes = plt.subplots(1, len(imgs), figsize=(6 * len(imgs), 5))
    axes = np.atleast_1d(axes)
    for ax, (name, img) in zip(axes, imgs):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(name)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(base_path: Path | str) -> Path:
    base_path = Path(base_path).resolve()
    results_csv = _resolve_results_csv(base_path)
    models_dir = results_csv.parent
    out_dir = models_dir / "model_type_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)
    best_rows = _best_run_per_type(df)

    missing = [m for m in MODEL_TYPE_ORDER if m not in best_rows]
    if missing:
        print(f"No completed runs found for: {', '.join(missing)} (skipping).")

    class_metrics: dict[str, Optional[pd.DataFrame]] = {}
    summary_metrics: dict[str, dict] = {}
    cm_pngs: dict[str, Path] = {}

    for label in MODEL_TYPE_ORDER:
        row = best_rows.get(label)
        if row is None:
            continue
        run_id = row["run_id"]
        cm_csv = models_dir / run_id / f"{run_id}_confusion_matrix.csv"
        if not cm_csv.exists():
            print(f"  [{label}] Skipping {run_id}: confusion matrix not found at {cm_csv}")
            continue

        y_true, y_pred, classes = _reconstruct_labels_preds(cm_csv)
        cm = confusion_matrix(y_true, y_pred, labels=classes)

        out_png = out_dir / f"{label}_confusion_matrix.png"
        out_csv = out_dir / f"{label}_confusion_matrix.csv"
        _save_cm(cm, classes, out_png, out_csv)
        cm_pngs[label] = out_png

        cr_png = out_dir / f"{label}_classification_report.png"
        cr_csv = out_dir / f"{label}_classification_report.csv"
        cr_txt = out_dir / f"{label}_classification_report.txt"
        _save_classification_report(y_true, y_pred, classes, cr_png, cr_csv)
        _save_classification_report_text(y_true, y_pred, classes, cr_txt)

        class_metrics[label], summary_metrics[label] = _parse_classification_report_text(cr_txt)
        print(f"[{label}] best run: {run_id} (test F1={row['_score']:.4f}) -> {cr_txt}")

    if not class_metrics:
        raise RuntimeError("No model type had a usable completed run; nothing to report.")

    _grouped_bars_per_class(class_metrics, "precision", out_dir / "per_class_precision_by_model_type.png")
    _grouped_bars_per_class(class_metrics, "recall", out_dir / "per_class_recall_by_model_type.png")
    _grouped_bars_per_class(class_metrics, "f1-score", out_dir / "per_class_f1_by_model_type.png")
    _radar_chart_f1(class_metrics, out_dir / "radar_f1_by_model_type.png")
    _averages_comparison(summary_metrics, out_dir / "averages_comparison_by_model_type.png")
    _accuracy_f1_bar_chart(summary_metrics, out_dir / "accuracy_f1_by_model_type.png")
    _confusion_side_by_side(cm_pngs, out_dir / "confusion_matrices_side_by_side.png")

    summary_df = pd.DataFrame(summary_metrics).T
    summary_df.index.name = "model_type"
    summary_df.to_csv(out_dir / "model_type_comparison_summary.csv")

    breakdown_df = _per_class_breakdown(class_metrics)
    breakdown_df.to_csv(out_dir / "per_class_breakdown_by_model_type.csv", index=False)
    print("\nPer-class breakdown by model type:")
    print(breakdown_df.to_string(index=False))

    print(f"\nSaved per-model-type classification reports + comparison plots to: {out_dir}")
    return out_dir


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    main(project_root)
