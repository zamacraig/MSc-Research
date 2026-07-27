import json
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch_geometric_temporal.signal import temporal_signal_split

from masters_research.pipeline.dataset import create_temporal_dataset
from masters_research.data.temporal_loader import TemporalDataLoader
from masters_research.pipeline.compare_reports import main as generate_comparison_plots


def _load_config(base_path: Path) -> Dict[str, Any]:
    cfg_path = base_path / "configs" / "default.json"
    try:
        return json.loads(cfg_path.read_text())
    except Exception:
        return {}


def _resolve_results_csv(base_path: Path, cfg: Dict[str, Any]) -> Path:
    # Preferred location (as per training config)
    default_models = base_path / "src" / "masters_research" / "artifacts" / "models"
    save_dir_cfg = cfg.get("save_dir", str(default_models))
    save_dir = Path(save_dir_cfg)
    if not save_dir.is_absolute():
        save_dir = base_path / save_dir
    candidate = save_dir / "grid_results.csv"
    if candidate.exists():
        return candidate
    # Fallback to top-level models/
    alt = base_path / "models" / "grid_results.csv"
    if alt.exists():
        return alt
    raise FileNotFoundError("grid_results.csv not found in configured artifacts nor in models/.")


def _pick_best_row(df: pd.DataFrame) -> pd.Series:
    # Prefer highest final test F1; fall back to val_f1 / best_f1 on legacy CSVs.
    sort_cols = [c for c in ("final_test_f1", "val_f1", "best_f1", "f1") if c in df.columns]
    if not sort_cols:
        return df.iloc[0]
    df_sorted = df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return df_sorted.iloc[0]


def _classify_run(params: Dict[str, Any]) -> str:
    """Mirror pipeline/train.py::_model_type_from_cfg so we can rebuild per-run paths
    for legacy CSVs that predate the model_type column."""
    model_type = params.get("model_type")
    if model_type == "simple_gcn" or bool(params.get("use_simple_gcn", False)):
        return "SimpleGCN"
    if model_type == "temporal_attention" or bool(params.get("use_temporal_attention", False)):
        return "TemporalAttentionGNN"
    if bool(params.get("use_attention", False)):
        return "TemporalGAT_GRU"
    return "TemporalGCN_GRU"


def _resolve_run_dirs(best: pd.Series, params: Dict[str, Any], results_csv: Path,
                      base_path: Path, cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    """Return (run_models_dir, run_plots_dir) for the best run, preferring the new
    columns written by train.py and falling back to reconstructing paths from
    (save_dir, plots_dir, model_type, run_id) for legacy CSVs."""
    models_dir_str = str(best.get("run_models_dir", "") or "").strip()
    plots_dir_str = str(best.get("run_plots_dir", "") or "").strip()
    if models_dir_str and plots_dir_str:
        return Path(models_dir_str), Path(plots_dir_str)

    model_type = str(best.get("model_type", "") or "").strip() or _classify_run(params)
    run_id = str(best["run_id"])

    models_root = results_csv.parent
    default_plots = base_path / "src" / "masters_research" / "artifacts" / "plots"
    plots_root = Path(cfg.get("plots_dir", str(default_plots)))
    if not plots_root.is_absolute():
        plots_root = base_path / plots_root

    return models_root / model_type / run_id, plots_root / model_type / run_id


def _load_model(params: Dict[str, Any], num_classes: int, device: torch.device):
    use_attention = bool(params.get("use_attention", False))
    in_channels = int(params.get("in_channels", 8))
    hidden_channels = int(params.get("hidden_channels", 16))

    if use_attention:
        try:
            from masters_research.models.temporal_spatial_attention import (
                TemporalSpatialAttentionNodeClassifier,
            )
            model = TemporalSpatialAttentionNodeClassifier(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_classes=num_classes,
                heads=int(params.get("gat_heads", 4)),
            )
        except Exception:
            from masters_research.models.temporal_spatial_classifier import (
                TemporalSpatialNodeClassifier,
            )
            model = TemporalSpatialNodeClassifier(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_classes=num_classes,
            )
    else:
        from masters_research.models.temporal_spatial_classifier import (
            TemporalSpatialNodeClassifier,
        )

        model = TemporalSpatialNodeClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
        )

    return model.to(device)


def _evaluate(model, loader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for snapshot in loader:
            out = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            pred = out.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(snapshot.y.cpu().numpy())
    return np.array(preds), np.array(labels)


def _save_cm(cm: np.ndarray, classes: list, out_png: Path, out_csv: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    df_cm.to_csv(out_csv, index_label="true\\pred")

    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Best Checkpoint)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def _save_classification_report(labels: np.ndarray, preds: np.ndarray, classes: list, out_png: Path, out_csv: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Build sklearn classification report (dict) and save CSV
    report_dict = classification_report(labels, preds, labels=classes, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(out_csv)

    # Prepare heatmap for class-only rows and core metrics
    class_rows = [str(c) for c in classes if str(c) in report_df.index]
    if not class_rows:
        return
    plot_df = report_df.loc[class_rows, ["precision", "recall", "f1-score"]]
    # Ensure numeric
    plot_df = plot_df.astype(float)

    plt.figure(figsize=(6.5, 0.5 + 0.5 * len(class_rows)))
    sns.heatmap(plot_df, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.0, vmax=1.0, cbar=True)
    plt.title("Classification Report (Best Checkpoint)")
    plt.xlabel("Metric")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def _save_classification_report_text(labels: np.ndarray, preds: np.ndarray, classes: list, out_txt: Path) -> None:
    """Save the original sklearn text layout including per-class rows and summary lines."""
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    # Use explicit target names for clearer headers
    target_names = [str(c) for c in classes]
    report_str = classification_report(
        labels,
        preds,
        labels=classes,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    out_txt.write_text(report_str)


def main(base_path: Path | str) -> Path:
    """Load best model from grid_results.csv and save its confusion matrix.

    Returns the path to the saved PNG confusion matrix.
    """
    base_path = Path(base_path).resolve()
    cfg = _load_config(base_path)

    results_csv = _resolve_results_csv(base_path, cfg)
    df = pd.read_csv(results_csv)
    best = _pick_best_row(df)

    # Params may be a JSON string; fall back to config for missing keys
    params = {}
    try:
        raw = best.get("params")
        params = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except Exception:
        params = {}

    # Merge defaults for robustness
    merged = {**cfg, **params}

    # Device
    dev = merged.get("device", None)
    if dev == "cpu":
        device = torch.device("cpu")
    elif dev == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    data_path = merged.get("data_path", "Dataset/TransientData/processedData.csv")
    data_path = base_path / data_path if not Path(data_path).is_absolute() else Path(data_path)
    final_df = pd.read_csv(data_path)
    dataset = create_temporal_dataset(final_df)
    train_ratio = float(merged.get("train_ratio", 0.8))
    _, test_dataset = temporal_signal_split(dataset, train_ratio=train_ratio)
    test_loader = TemporalDataLoader(test_dataset)

    # Model
    num_classes = int(merged.get("num_classes", 5))
    model = _load_model(merged, num_classes=num_classes, device=device)

    # We prefer the saved best checkpoint, else fall back to final model path.
    best_model_path = str(best.get("best_model_path", "")).strip()
    model_path = best_model_path if best_model_path else str(best.get("model_path"))
    model_path = Path(model_path)
    if not model_path.is_absolute():
        # Resolve relative to grid_results location
        model_path = results_csv.parent / model_path.name
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Evaluate
    preds, labels = _evaluate(model, test_loader)
    classes = list(range(num_classes))
    cm = confusion_matrix(labels, preds, labels=classes)

    # Route best-run reports into models/<model_type>/<run_id>/ and plots/<model_type>/<run_id>/
    run_models_dir, run_plots_dir = _resolve_run_dirs(best, merged, results_csv, base_path, cfg)
    run_models_dir.mkdir(parents=True, exist_ok=True)
    run_plots_dir.mkdir(parents=True, exist_ok=True)

    out_png = run_plots_dir / f"{best['run_id']}_confusion_matrix_best.png"
    out_csv = run_models_dir / f"{best['run_id']}_confusion_matrix_best.csv"
    _save_cm(cm, classes, out_png, out_csv)

    # Classification report (CSV + heatmap + text)
    cr_png = run_plots_dir / f"{best['run_id']}_classification_report_best.png"
    cr_csv = run_models_dir / f"{best['run_id']}_classification_report_best.csv"
    _save_classification_report(labels, preds, classes, cr_png, cr_csv)
    cr_txt = run_models_dir / f"{best['run_id']}_classification_report_best.txt"
    _save_classification_report_text(labels, preds, classes, cr_txt)

    print(f"Best run: {best['run_id']} | model: {model_path}")
    print(f"Saved confusion matrix to: {out_png} and {out_csv}")
    print(f"Saved classification report to: {cr_png}, {cr_csv} and {cr_txt}")

    # Generate comparison plots using RF/Keras/GNN reports in artifacts/models
    try:
        generate_comparison_plots(base_path)
    except Exception as e:
        print(f"Comparison plot generation skipped due to error: {e}")
    return out_png


if __name__ == "__main__":
    # Allow running directly for quick manual use
    project_root = Path(__file__).resolve().parents[3]
    main(project_root)
