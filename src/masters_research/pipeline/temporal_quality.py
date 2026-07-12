"""
temporal_quality.py
-------------------
Rates the temporal quality of the graph dataset across five dimensions:

  1. Node Coverage       – fraction of nodes present per timestamp
  2. Timestamp Regularity – how evenly spaced the timestamps are
  3. Label Drift          – how much the class distribution shifts over time
                           (Jensen-Shannon divergence between rolling windows)
  4. Feature Drift        – how much feature means shift over time
                           (normalised mean-absolute deviation across timestamps)
  5. Temporal Autocorrelation – how predictable each timestamp is from the previous
                                (avg label overlap between adjacent snapshots)

Each dimension is scored 0–1 (higher = better quality).
A composite score (weighted mean) is also reported.

Usage (standalone or from train.py):
    from masters_research.pipeline.temporal_quality import run_temporal_quality_audit
    report = run_temporal_quality_audit(final_df, plots_dir=Path("artifacts/plots"))
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import jensenshannon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base-2) between two probability vectors."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # Smooth to avoid log(0)
    eps = 1e-10
    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()
    return float(jensenshannon(p, q, base=2))


def _infer_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if "_x" in c
        and "Compno" not in c
        and "Target" not in c
        and "TimeStamp" not in c
    ]


# ---------------------------------------------------------------------------
# Dimension 1 – Node Coverage
# ---------------------------------------------------------------------------

def score_node_coverage(df: pd.DataFrame, node_col: str = "Compno_x") -> dict:
    """Score = mean fraction of unique nodes appearing in each timestamp."""
    total_nodes = df[node_col].nunique()
    per_ts = df.groupby("TimeStamp")[node_col].nunique()
    coverage = per_ts / total_nodes
    score = float(coverage.mean())
    return {
        "score": score,
        "mean_coverage": score,
        "min_coverage": float(coverage.min()),
        "max_coverage": float(coverage.max()),
        "per_timestamp": coverage,
    }


# ---------------------------------------------------------------------------
# Dimension 2 – Timestamp Regularity
# ---------------------------------------------------------------------------

def score_timestamp_regularity(df: pd.DataFrame, ts_col: str = "TimeStamp") -> dict:
    """Score = 1 - coefficient of variation of inter-timestamp gaps.

    A perfectly regular series scores 1.0; an irregular one scores lower.
    """
    timestamps = pd.to_datetime(df[ts_col].unique())
    timestamps = np.sort(timestamps)
    if len(timestamps) < 2:
        return {"score": 1.0, "n_timestamps": len(timestamps), "gaps_days": np.array([])}

    gaps = np.diff(timestamps).astype("timedelta64[D]").astype(float)
    mean_gap = gaps.mean()
    std_gap = gaps.std()
    cv = std_gap / mean_gap if mean_gap > 0 else 0.0
    # Clip so CV > 1 still maps to a positive score
    score = float(max(0.0, 1.0 - min(cv, 1.0)))
    return {
        "score": score,
        "n_timestamps": len(timestamps),
        "mean_gap_days": float(mean_gap),
        "std_gap_days": float(std_gap),
        "cv_gap": float(cv),
        "gaps_days": gaps,
    }


# ---------------------------------------------------------------------------
# Dimension 3 – Label Drift
# ---------------------------------------------------------------------------

def score_label_drift(
    df: pd.DataFrame,
    target_col: str = "Target_x",
    node_col: str = "Compno_x",
    window: int = 10,
) -> dict:
    """Score = 1 - mean pairwise JS divergence between adjacent rolling windows.

    Uses per-node target values (one label per node per timestamp).
    """
    # One row per (timestamp, node)
    ts_node = (
        df.groupby(["TimeStamp", node_col])[target_col]
        .first()
        .reset_index()
    )
    timestamps = sorted(ts_node["TimeStamp"].unique())
    n_classes = int(ts_node[target_col].max()) + 1

    # Build class distribution per timestamp
    dists = []
    for ts in timestamps:
        labels = ts_node.loc[ts_node["TimeStamp"] == ts, target_col].values
        counts = np.bincount(labels.astype(int), minlength=n_classes).astype(float)
        dists.append(counts / counts.sum())

    dists = np.array(dists)  # (T, C)

    # Rolling window average distributions
    def window_mean(i):
        lo = max(0, i - window // 2)
        hi = min(len(dists), i + window // 2 + 1)
        return dists[lo:hi].mean(axis=0)

    js_scores = []
    for i in range(1, len(dists)):
        js = _js_divergence(window_mean(i - 1), window_mean(i))
        js_scores.append(js)

    mean_js = float(np.mean(js_scores)) if js_scores else 0.0
    # JS divergence ranges 0–1; invert so high drift → low score
    score = float(1.0 - mean_js)

    return {
        "score": score,
        "mean_js_divergence": mean_js,
        "max_js_divergence": float(max(js_scores)) if js_scores else 0.0,
        "js_per_step": np.array(js_scores),
        "dist_per_timestamp": dists,
        "timestamps": timestamps,
        "n_classes": n_classes,
    }


# ---------------------------------------------------------------------------
# Dimension 4 – Feature Drift
# ---------------------------------------------------------------------------

def score_feature_drift(df: pd.DataFrame) -> dict:
    """Score = 1 - mean normalised feature-mean drift across timestamps.

    For each feature, computes std(per-timestamp means) / overall_std.
    Score is 1 minus the mean across features (clipped to [0, 1]).
    """
    feature_cols = _infer_feature_cols(df)
    if not feature_cols:
        return {"score": 1.0, "per_feature": {}, "feature_cols": []}

    ts_means = df.groupby("TimeStamp")[feature_cols].mean()
    overall_std = df[feature_cols].std().replace(0, 1)
    normalised_drift = (ts_means.std() / overall_std).fillna(0)

    mean_drift = float(normalised_drift.mean())
    score = float(max(0.0, 1.0 - min(mean_drift, 1.0)))

    return {
        "score": score,
        "mean_normalised_drift": mean_drift,
        "per_feature": normalised_drift.to_dict(),
        "feature_cols": feature_cols,
        "ts_means": ts_means,
    }


# ---------------------------------------------------------------------------
# Dimension 5 – Temporal Autocorrelation
# ---------------------------------------------------------------------------

def score_temporal_autocorrelation(
    df: pd.DataFrame,
    target_col: str = "Target_x",
    node_col: str = "Compno_x",
) -> dict:
    """Score = mean Jaccard overlap of dominant classes between adjacent timestamps.

    High autocorrelation means the class structure changes slowly — good for
    temporal models that rely on recent context.
    """
    ts_node = (
        df.groupby(["TimeStamp", node_col])[target_col]
        .first()
        .reset_index()
    )
    timestamps = sorted(ts_node["TimeStamp"].unique())

    def dominant_set(ts):
        labels = set(ts_node.loc[ts_node["TimeStamp"] == ts, target_col].astype(int).unique())
        return labels

    overlaps = []
    for i in range(1, len(timestamps)):
        a = dominant_set(timestamps[i - 1])
        b = dominant_set(timestamps[i])
        if a | b:
            overlaps.append(len(a & b) / len(a | b))

    score = float(np.mean(overlaps)) if overlaps else 1.0
    return {
        "score": score,
        "mean_jaccard_overlap": score,
        "min_overlap": float(min(overlaps)) if overlaps else 1.0,
        "overlaps_per_step": np.array(overlaps),
    }


# ---------------------------------------------------------------------------
# Report + Plots
# ---------------------------------------------------------------------------

def _plot_label_drift(result: dict, plots_dir: Path):
    dists = result["dist_per_timestamp"]   # (T, C)
    timestamps = result["timestamps"]
    n_classes = result["n_classes"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Stacked area – class proportion over time
    ax = axes[0]
    df_plot = pd.DataFrame(dists, index=timestamps, columns=[f"Class {c}" for c in range(n_classes)])
    try:
        df_plot.plot.area(ax=ax, colormap="tab10", alpha=0.8)
    except Exception as e:
        print(f"  Warning: pandas plotting backend unavailable ({e}). Falling back to matplotlib stackplot.")
        x_vals = np.arange(len(timestamps))
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        ax.stackplot(x_vals, df_plot.T, colors=colors[:n_classes], alpha=0.8)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([str(ts) for ts in timestamps], rotation=45, ha="right", fontsize=7)
    ax.set_title("Class Distribution Over Time (Label Drift)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Proportion")
    ax.legend(loc="upper right", fontsize=8)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)

    # JS divergence per step
    ax2 = axes[1]
    js = result["js_per_step"]
    ax2.plot(range(1, len(js) + 1), js, marker="o", markersize=3, linewidth=1)
    ax2.axhline(result["mean_js_divergence"], color="red", linestyle="--", label=f"Mean = {result['mean_js_divergence']:.3f}")
    ax2.set_title("JS Divergence Between Adjacent Time-Windows")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("JS Divergence (0=identical, 1=opposite)")
    ax2.legend()

    fig.tight_layout()
    out = plots_dir / "temporal_quality_label_drift.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_feature_drift(result: dict, plots_dir: Path):
    if not result["feature_cols"]:
        return None
    ts_means = result["ts_means"]
    fig, ax = plt.subplots(figsize=(14, 5))
    for col in result["feature_cols"]:
        ax.plot(ts_means.index, ts_means[col], label=col, alpha=0.7, linewidth=1)
    ax.set_title("Feature Means Over Time (Feature Drift)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Mean Value")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    out = plots_dir / "temporal_quality_feature_drift.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_coverage(result: dict, plots_dir: Path):
    coverage = result["per_timestamp"]
    fig, ax = plt.subplots(figsize=(14, 4))
    try:
        coverage.plot(kind="bar", ax=ax, color="steelblue", alpha=0.8)
    except Exception as e:
        print(f"  Warning: pandas plotting backend unavailable ({e}). Falling back to matplotlib bar plot.")
        ax.bar(range(len(coverage)), coverage.values, color="steelblue", alpha=0.8)
        ax.set_xticks(range(len(coverage)))
        ax.set_xticklabels([str(ts) for ts in coverage.index], rotation=45, ha="right", fontsize=6)
    ax.axhline(result["mean_coverage"], color="red", linestyle="--", label=f"Mean = {result['mean_coverage']:.3f}")
    ax.set_title("Node Coverage per Timestamp")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Fraction of Nodes Present")
    ax.legend()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=6)
    fig.tight_layout()
    out = plots_dir / "temporal_quality_node_coverage.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_summary_radar(scores: dict[str, float], plots_dir: Path):
    labels = list(scores.keys())
    values = [scores[k] for k in labels]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.plot(angles, values_plot, "o-", linewidth=2, color="steelblue")
    ax.fill(angles, values_plot, alpha=0.25, color="steelblue")
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Temporal Quality Radar", pad=20)
    fig.tight_layout()
    out = plots_dir / "temporal_quality_radar.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Split Comparison
# ---------------------------------------------------------------------------

def _extract_labels_and_features(dataset) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract all labels and optionally feature matrices from a temporal dataset."""
    n = getattr(dataset, "snapshot_count", None) or len(dataset)
    all_labels = []
    all_features = []
    for i in range(n):
        snap = dataset[i]
        y = snap.y
        if hasattr(y, "cpu"):
            y = y.cpu().numpy()
        y = np.asarray(y).astype(int)
        if y.size:
            all_labels.append(y)
        x = getattr(snap, "x", None)
        if x is not None:
            if hasattr(x, "cpu"):
                x = x.cpu().numpy()
            all_features.append(np.asarray(x))
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    features = np.concatenate(all_features, axis=0) if all_features else None
    return labels, features


def compare_splits(
    split_a,
    split_b,
    name_a: str = "Split A",
    name_b: str = "Split B",
    num_classes: int = 5,
    plots_dir: Path | None = None,
    plot_prefix: str = "split_quality",
) -> dict:
    """Compare two temporal dataset splits for distribution shift.

    Reports:
      - Per-class label distribution in each split
      - JS divergence between the two label distributions
      - Per-feature mean/std shift (as z-score difference)
      - Composite shift score: 1 - JS  (1 = identical, 0 = opposite distributions)

    Parameters
    ----------
    split_a, split_b : DynamicGraphTemporalSignalBatch or similar
    name_a, name_b   : Human-readable names for the splits
    num_classes      : Expected number of target classes
    plots_dir        : If given, saves comparison plots there
    plot_prefix      : Filename prefix for produced plots
    """
    labels_a, feats_a = _extract_labels_and_features(split_a)
    labels_b, feats_b = _extract_labels_and_features(split_b)

    # --- Label distribution ---
    counts_a = np.bincount(labels_a, minlength=num_classes).astype(float)
    counts_b = np.bincount(labels_b, minlength=num_classes).astype(float)
    dist_a = counts_a / counts_a.sum()
    dist_b = counts_b / counts_b.sum()
    js = _js_divergence(dist_a, dist_b)
    label_shift_score = float(1.0 - js)

    # --- Feature distribution shift ---
    feature_shift = {}
    feature_shift_score = 1.0
    if feats_a is not None and feats_b is not None:
        n_features = feats_a.shape[1]
        mean_a = feats_a.mean(axis=0)
        mean_b = feats_b.mean(axis=0)
        std_pool = np.sqrt((feats_a.var(axis=0) + feats_b.var(axis=0)) / 2 + 1e-8)
        z_shift = np.abs(mean_a - mean_b) / std_pool
        for fi in range(n_features):
            feature_shift[fi] = float(z_shift[fi])
        mean_z = float(z_shift.mean())
        # Normalise: z-shift of 2 or more = very significant
        feature_shift_score = float(max(0.0, 1.0 - min(mean_z / 2.0, 1.0)))
    else:
        mean_z = 0.0

    composite = (label_shift_score + feature_shift_score) / 2.0

    def _grade(s):
        if s >= 0.85: return "Minimal shift"
        if s >= 0.70: return "Moderate shift"
        if s >= 0.50: return "Significant shift"
        return "Severe shift"

    print("\n" + "=" * 60)
    print(f"  SPLIT QUALITY: {name_a}  vs  {name_b}")
    print("=" * 60)
    print(f"  Samples:  {name_a}={labels_a.size}  |  {name_b}={labels_b.size}")
    print()
    print(f"  {'Class':<8} {name_a:>12}  {name_b:>12}  {'Diff':>8}")
    print(f"  {'-'*46}")
    for c in range(num_classes):
        pa, pb = dist_a[c], dist_b[c]
        diff = pb - pa
        marker = "  <-- drift" if abs(diff) > 0.08 else ""
        print(f"  {c:<8} {pa:>11.1%}  {pb:>11.1%}  {diff:>+8.1%}{marker}")
    print()
    print(f"  Label JS divergence : {js:.4f}  (0=identical, 1=opposite)")
    print(f"  Label shift score   : {label_shift_score:.4f}  → {_grade(label_shift_score)}")
    if feats_a is not None:
        print(f"  Feature mean z-shift: {mean_z:.4f}  (avg across features)")
        print(f"  Feature shift score : {feature_shift_score:.4f}  → {_grade(feature_shift_score)}")
    print(f"  Composite shift     : {composite:.4f}  → {_grade(composite)}")
    if js > 0.15:
        print(f"\n  [!] High label drift detected (JS={js:.3f}).")
        print(f"      Model trained on {name_a} will likely underperform on {name_b}.")
        print(f"      Dominant drifted classes: " +
              ", ".join(str(c) for c in range(num_classes)
                         if abs(dist_b[c] - dist_a[c]) > 0.05))
    print("=" * 60)

    saved_plots = {}
    if plots_dir is not None:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Side-by-side class distribution bar chart
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        classes = list(range(num_classes))

        ax = axes[0]
        x = np.arange(num_classes)
        width = 0.35
        ax.bar(x - width/2, dist_a * 100, width, label=name_a, alpha=0.8, color="steelblue")
        ax.bar(x + width/2, dist_b * 100, width, label=name_b, alpha=0.8, color="coral")
        ax.set_xlabel("Class")
        ax.set_ylabel("Proportion (%)")
        ax.set_title(f"Class Distribution: {name_a} vs {name_b}\nJS={js:.3f}")
        ax.set_xticks(x)
        ax.legend()

        ax2 = axes[1]
        diff_pct = (dist_b - dist_a) * 100
        colors = ["coral" if d > 0 else "steelblue" for d in diff_pct]
        ax2.bar(classes, diff_pct, color=colors, alpha=0.85)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("Class")
        ax2.set_ylabel("Proportion diff (%, B minus A)")
        ax2.set_title(f"Class Drift: {name_b} relative to {name_a}")
        ax2.set_xticks(classes)

        fig.tight_layout()
        out = plots_dir / f"{plot_prefix}_label_drift.png"
        fig.savefig(out)
        plt.close(fig)
        saved_plots["label_drift"] = out

        # Feature z-shift bar if available
        if feats_a is not None and feature_shift:
            fig2, ax3 = plt.subplots(figsize=(10, 4))
            fi_labels = [str(i) for i in feature_shift]
            fi_vals = list(feature_shift.values())
            colors2 = ["coral" if v > 1.0 else "steelblue" for v in fi_vals]
            ax3.bar(fi_labels, fi_vals, color=colors2, alpha=0.85)
            ax3.axhline(1.0, color="red", linestyle="--", label="z=1 threshold")
            ax3.axhline(2.0, color="darkred", linestyle="--", label="z=2 threshold")
            ax3.set_xlabel("Feature index")
            ax3.set_ylabel("|mean shift| / pooled std")
            ax3.set_title(f"Feature Mean Shift: {name_a} vs {name_b}")
            ax3.legend()
            fig2.tight_layout()
            out2 = plots_dir / f"{plot_prefix}_feature_shift.png"
            fig2.savefig(out2)
            plt.close(fig2)
            saved_plots["feature_shift"] = out2

        print(f"  Split comparison plots saved to: {plots_dir}")

    return {
        "label_shift_score": label_shift_score,
        "feature_shift_score": feature_shift_score,
        "composite_shift_score": composite,
        "js_divergence": js,
        "dist_a": dist_a,
        "dist_b": dist_b,
        "feature_shift_z": feature_shift,
        "plots": saved_plots,
    }


def run_temporal_quality_audit(
    df: pd.DataFrame,
    plots_dir: Path | None = None,
    target_col: str = "Target_x",
    node_col: str = "Compno_x",
    ts_col: str = "TimeStamp",
    label_window: int = 10,
    weights: dict[str, float] | None = None,
) -> dict:
    """Run all five quality dimensions and print a formatted report.

    Parameters
    ----------
    df          : The preprocessed dataframe (processedData.csv).
                  Target_x should already be discretised into integer classes.
    plots_dir   : If provided, saves diagnostic plots there.
    weights     : Per-dimension score weights. Defaults to equal weighting.

    Returns
    -------
    dict with keys: scores, composite_score, details, plots
    """
    if weights is None:
        weights = {
            "Node Coverage":        0.20,
            "Timestamp Regularity": 0.15,
            "Label Drift":          0.30,
            "Feature Drift":        0.20,
            "Temporal Autocorr.":   0.15,
        }

    print("\n" + "=" * 60)
    print("         TEMPORAL DATA QUALITY AUDIT")
    print("=" * 60)

    cov   = score_node_coverage(df, node_col=node_col)
    reg   = score_timestamp_regularity(df, ts_col=ts_col)
    drift = score_label_drift(df, target_col=target_col, node_col=node_col, window=label_window)
    feat  = score_feature_drift(df)
    auto  = score_temporal_autocorrelation(df, target_col=target_col, node_col=node_col)

    scores = {
        "Node Coverage":        cov["score"],
        "Timestamp Regularity": reg["score"],
        "Label Drift":          drift["score"],
        "Feature Drift":        feat["score"],
        "Temporal Autocorr.":   auto["score"],
    }

    w_arr = np.array([weights[k] for k in scores])
    s_arr = np.array([scores[k] for k in scores])
    composite = float((w_arr * s_arr).sum() / w_arr.sum())

    def _grade(s):
        if s >= 0.85: return "Excellent"
        if s >= 0.70: return "Good"
        if s >= 0.55: return "Fair"
        if s >= 0.40: return "Poor"
        return "Very Poor"

    print(f"\n{'Dimension':<26} {'Score':>6}  {'Grade':<12}  Notes")
    print("-" * 72)
    print(f"{'Node Coverage':<26} {cov['score']:>6.3f}  {_grade(cov['score']):<12}  "
          f"min={cov['min_coverage']:.2f}, mean={cov['mean_coverage']:.2f}")
    print(f"{'Timestamp Regularity':<26} {reg['score']:>6.3f}  {_grade(reg['score']):<12}  "
          f"mean gap={reg.get('mean_gap_days', 0):.1f}d, CV={reg.get('cv_gap', 0):.3f}")
    print(f"{'Label Drift':<26} {drift['score']:>6.3f}  {_grade(drift['score']):<12}  "
          f"mean JS={drift['mean_js_divergence']:.3f}, max JS={drift['max_js_divergence']:.3f}")
    print(f"{'Feature Drift':<26} {feat['score']:>6.3f}  {_grade(feat['score']):<12}  "
          f"mean norm drift={feat['mean_normalised_drift']:.3f}")
    print(f"{'Temporal Autocorr.':<26} {auto['score']:>6.3f}  {_grade(auto['score']):<12}  "
          f"mean Jaccard={auto['mean_jaccard_overlap']:.3f}")
    print("-" * 72)
    print(f"{'COMPOSITE SCORE':<26} {composite:>6.3f}  {_grade(composite):<12}")
    print("=" * 60)

    # Interpretation notes
    print("\nKey observations:")
    if drift["mean_js_divergence"] > 0.15:
        print(f"  [!] High label drift (JS={drift['mean_js_divergence']:.3f}) — class distribution")
        print(f"      changes significantly across time, causing train/test skew.")
        print(f"      Consider temporal cross-validation or concept-drift-aware training.")
    if feat["mean_normalised_drift"] > 0.5:
        print(f"  [!] High feature drift — input distributions shift over time.")
        print(f"      Consider rolling normalisation or batch normalisation in the model.")
    if reg["score"] < 0.6:
        print(f"  [!] Irregular timestamps (CV={reg.get('cv_gap', 0):.3f}) — gaps vary widely.")
        print(f"      Consider time-aware positional encoding in the model.")
    if cov["min_coverage"] < 0.7:
        print(f"  [!] Some timestamps have low node coverage ({cov['min_coverage']:.0%}).")
        print(f"      Missing nodes may cause inconsistent graph structure.")
    if auto["score"] < 0.5:
        print(f"  [!] Low temporal autocorrelation — adjacent snapshots share few dominant classes.")
        print(f"      Temporal memory (LSTM/GRU layers) may not help much.")

    saved_plots = {}
    if plots_dir is not None:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        saved_plots["label_drift"]   = _plot_label_drift(drift, plots_dir)
        saved_plots["feature_drift"] = _plot_feature_drift(feat, plots_dir)
        saved_plots["node_coverage"] = _plot_coverage(cov, plots_dir)
        saved_plots["radar"]         = _plot_summary_radar(scores, plots_dir)
        print(f"\nQuality plots saved to: {plots_dir}")

    return {
        "scores": scores,
        "composite_score": composite,
        "details": {
            "node_coverage": cov,
            "timestamp_regularity": reg,
            "label_drift": drift,
            "feature_drift": feat,
            "temporal_autocorr": auto,
        },
        "plots": saved_plots,
    }
