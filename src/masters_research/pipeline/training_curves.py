"""Per-model learning curves: how fit and generalisation evolve during training.

Every model run records a training history and this module turns it into
plots. The x-axis is whatever "training progress" means for that model:

  * GNNs (SimpleGCN, TemporalGCN_GRU, TemporalAttentionGNN) and the Keras MLP:
    epochs.
  * RandomForest: number of trees. The full forest is fitted once and the
    curve is read off by averaging the first k trees' probabilities, so the
    curve describes exactly the forest that was evaluated — recording it
    does not change the model.

For each step two metrics are logged on BOTH the training and validation
partitions:

  * weighted F1 — the metric reported everywhere else in the thesis;
  * cross-entropy (NLL) — plain, unweighted, so train and val are on the same
    scale. This is deliberately NOT the training objective (focal loss with
    class weights, label smoothing, entropy and L2 penalties), which is also
    logged as ``objective`` for the GNNs but is not comparable to a val loss.

Only labelled (station, month) cases are scored, exactly as in evaluation.
The test partition is never touched here.

Outputs
-------
Per run:
    <run_models_dir>/training_history[_<label>].csv
    <run_plots_dir>/training_curve[_<label>].png
Aggregate (``aggregate_curves``):
    <plots_dir>/<prefix>training_curves_by_model.png  — mean ± 1 std across
        runs (seeds / folds), one column per model type
    <plots_dir>/<prefix>training_curves_summary.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

MODEL_ORDER = ["SimpleGCN", "TemporalGCN_GRU", "TemporalAttentionGNN", "RandomForest", "KerasMLP"]
HISTORY_COLUMNS = ["step", "train_f1", "val_f1", "train_nll", "val_nll"]


def nll(labels: np.ndarray, probs: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class (unweighted CE)."""
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if labels.size == 0:
        return float("nan")
    p = np.clip(probs[np.arange(labels.size), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def weighted_f1(labels, preds) -> float:
    if len(labels) == 0:
        return float("nan")
    return float(f1_score(labels, preds, average="weighted"))


def history_row(step: int, tr_labels, tr_preds, tr_probs, va_labels, va_preds, va_probs, **extra) -> dict:
    row = {
        "step": int(step),
        "train_f1": weighted_f1(tr_labels, tr_preds),
        "val_f1": weighted_f1(va_labels, va_preds),
        "train_nll": nll(tr_labels, tr_probs),
        "val_nll": nll(va_labels, va_probs),
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Per-model history builders for the non-graph baselines
# ---------------------------------------------------------------------------

def random_forest_history(rf, X_tr, y_tr, X_val, y_val, step: int = 10) -> list[dict]:
    """Train/val metrics of the first k trees, k = step, 2*step, ..., n_trees.

    sklearn's forest prediction is the mean of per-tree class probabilities,
    so averaging the first k trees reproduces a k-tree forest exactly.
    """
    n = len(rf.estimators_)
    classes = rf.classes_
    tree_tr = np.stack([t.predict_proba(X_tr) for t in rf.estimators_])
    tree_va = np.stack([t.predict_proba(X_val) for t in rf.estimators_])
    cum_tr, cum_va = np.cumsum(tree_tr, axis=0), np.cumsum(tree_va, axis=0)
    ks = sorted(set(list(range(step, n + 1, step)) + [1, n]))
    rows = []
    for k in ks:
        p_tr, p_va = cum_tr[k - 1] / k, cum_va[k - 1] / k
        rows.append(history_row(
            k, y_tr, classes[p_tr.argmax(1)], _full_probs(p_tr, classes),
            y_val, classes[p_va.argmax(1)], _full_probs(p_va, classes),
        ))
    return rows


def _full_probs(p: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Expand forest probabilities to label-indexed columns (a class absent
    from a training subset would otherwise shift the column positions)."""
    out = np.zeros((p.shape[0], int(classes.max()) + 1))
    out[:, classes.astype(int)] = p
    return out


def keras_history_callback(X_tr, y_tr, X_val, y_val, rows: list):
    """Keras callback that appends one history row per epoch to ``rows``.

    Uses inference-mode predictions (no dropout, frozen BatchNorm), so the
    training trajectory is unaffected.
    """
    import tensorflow as tf

    class _History(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            p_tr = self.model.predict(X_tr, verbose=0)
            p_va = self.model.predict(X_val, verbose=0)
            rows.append(history_row(
                epoch + 1, y_tr, p_tr.argmax(1), p_tr, y_val, p_va.argmax(1), p_va,
                objective=float((logs or {}).get("loss", np.nan)),
            ))

    return _History()


# ---------------------------------------------------------------------------
# Saving and plotting one run
# ---------------------------------------------------------------------------

def save_run_history(history: list[dict], run_models_dir: Path, run_plots_dir: Path | None,
                     title: str, x_label: str = "Epoch", selected_step: int | None = None,
                     label: str | None = None) -> Path | None:
    if not history:
        return None
    suffix = f"_{label}" if label else ""
    df = pd.DataFrame(history)
    df["selected"] = df["step"] == selected_step if selected_step is not None else False
    run_models_dir = Path(run_models_dir)
    run_models_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_models_dir / f"training_history{suffix}.csv"
    df.to_csv(csv_path, index=False)
    if run_plots_dir is not None:
        try:
            _plot_run(df, Path(run_plots_dir) / f"training_curve{suffix}.png", title, x_label, selected_step)
        except Exception as e:
            print(f"    training curve plot failed: {e}")
    return csv_path


def _plot_run(df: pd.DataFrame, out_path: Path, title: str, x_label: str, selected_step) -> None:
    fig, (ax_f1, ax_loss) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, ylab in [(ax_f1, "f1", "Weighted F1"), (ax_loss, "nll", "Cross-entropy (NLL)")]:
        ax.plot(df["step"], df[f"train_{metric}"], label="Train", color="#2a6fb0", linewidth=1.8)
        ax.plot(df["step"], df[f"val_{metric}"], label="Validation", color="#d0662b", linewidth=1.8)
        if selected_step is not None:
            ax.axvline(selected_step, color="0.4", linestyle="--", linewidth=1, label="Selected checkpoint")
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    ax_f1.set_ylim(0, 1)
    ax_f1.legend(loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregation across runs
# ---------------------------------------------------------------------------

def collect_histories(models_dir: Path, run_prefix: str | None = None) -> pd.DataFrame:
    """Load every training_history*.csv under models/<ModelType>/<run_id>/."""
    frames = []
    pat = re.compile(rf"^{re.escape(run_prefix)}(_f\d+)?_") if run_prefix else None
    for path in Path(models_dir).glob("*/*/training_history*.csv"):
        model_type, run_id = path.parent.parent.name, path.parent.name
        if model_type not in MODEL_ORDER or (pat and not pat.match(run_id)):
            continue
        df = pd.read_csv(path)
        df["model"], df["run_id"] = model_type, run_id
        fold = re.search(r"_f(\d+)_", run_id)
        df["fold"] = int(fold.group(1)) if fold else -1
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_curves(models_dir: Path, plots_dir: Path, run_prefix: str | None = None) -> Path | None:
    """Mean ± 1 std learning curve per model type across all its runs.

    GNN runs stop at different epochs under early stopping; at each epoch the
    mean is taken over the runs still training, and the number of such runs
    is saved in the summary CSV.
    """
    hist = collect_histories(models_dir, run_prefix)
    if hist.empty:
        print("No training histories found; skipping learning-curve aggregation.")
        return None
    models = [m for m in MODEL_ORDER if m in set(hist["model"])]
    metrics = ["train_f1", "val_f1", "train_nll", "val_nll"]
    summary = (hist.groupby(["model", "step"])[metrics].agg(["mean", "std"]))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary["n_runs"] = hist.groupby(["model", "step"]).size()
    summary = summary.reset_index()

    prefix = f"{run_prefix}_" if run_prefix else ""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = plots_dir / f"{prefix}training_curves_summary.csv"
    summary.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(2, len(models), figsize=(3.6 * len(models), 6.4), squeeze=False)
    for j, model in enumerate(models):
        s = summary[summary["model"] == model].sort_values("step")
        n_runs = hist[hist["model"] == model]["run_id"].nunique()
        x_label = "Trees" if model == "RandomForest" else "Epoch"
        for i, (metric, ylab) in enumerate([("f1", "Weighted F1"), ("nll", "Cross-entropy (NLL)")]):
            ax = axes[i][j]
            for part, color, name in [("train", "#2a6fb0", "Train"), ("val", "#d0662b", "Validation")]:
                m = s[f"{part}_{metric}_mean"].to_numpy()
                sd = s[f"{part}_{metric}_std"].fillna(0).to_numpy()
                ax.plot(s["step"], m, color=color, linewidth=1.8, label=name)
                ax.fill_between(s["step"], m - sd, m + sd, color=color, alpha=0.15)
            if i == 0:
                ax.set_title(f"{model}\n({n_runs} runs)", fontsize=10)
                ax.set_ylim(0, 1)
            ax.set_xlabel(x_label)
            if j == 0:
                ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.3)
    axes[0][0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Learning curves per model (mean ± 1 std across runs; validation partition)")
    fig.tight_layout()
    out_png = plots_dir / f"{prefix}training_curves_by_model.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Learning curves written to:\n  {out_png}\n  {summary_csv}")
    return out_png


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate per-run training histories into learning curves.")
    parser.add_argument("--models-dir", default="src/masters_research/artifacts/models")
    parser.add_argument("--plots-dir", default="src/masters_research/artifacts/plots")
    parser.add_argument("--run-prefix", default=None)
    a = parser.parse_args()
    aggregate_curves(Path(a.models_dir), Path(a.plots_dir), a.run_prefix)
