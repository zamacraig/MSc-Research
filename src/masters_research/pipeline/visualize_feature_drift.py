"""
Visualize Feature Drift in temporal data.

This script generates detailed plots to understand why the Feature Drift
score in the temporal quality audit is poor (0.185).

Run from project root:
    python -m src.masters_research.pipeline.visualize_feature_drift
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_data(base_path: Path) -> pd.DataFrame:
    """Load the processed data."""
    data_path = base_path / "Dataset" / "TransientData" / "processedData.csv"
    if not data_path.exists():
        # Fallback to alternate location
        data_path = base_path / "Dataset" / "TransientData" / "final_df.csv"
    print(f"Loading data from {data_path}")
    return pd.read_csv(data_path)


def infer_feature_cols(df: pd.DataFrame) -> list[str]:
    """Infer feature columns (numeric, not ID/target)."""
    exclude = {"TimeStamp", "Compno_x", "Compno_y", "Target_x", "Target_y", 
               "ComputerNo", "Label", "Target", "month", "month_sin", "month_cos"}
    feature_cols = [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        and not c.endswith("_y")
    ]
    return feature_cols


def plot_feature_distributions_over_time(df: pd.DataFrame, feature_cols: list[str], plots_dir: Path):
    """
    Create boxplots showing how each feature's distribution changes across time.
    """
    timestamps = sorted(df["TimeStamp"].unique())
    n_ts = len(timestamps)
    
    # Sample timestamps if too many
    if n_ts > 20:
        sample_idx = np.linspace(0, n_ts - 1, 20, dtype=int)
        sampled_ts = [timestamps[i] for i in sample_idx]
    else:
        sampled_ts = timestamps
    
    df_sampled = df[df["TimeStamp"].isin(sampled_ts)]
    
    n_features = len(feature_cols)
    fig, axes = plt.subplots(n_features, 1, figsize=(14, 3 * n_features), squeeze=False)
    
    for i, col in enumerate(feature_cols):
        ax = axes[i, 0]
        # Create position mapping for box plot
        df_sampled["ts_pos"] = df_sampled["TimeStamp"].map({ts: j for j, ts in enumerate(sampled_ts)})
        
        data_for_box = [df_sampled[df_sampled["TimeStamp"] == ts][col].dropna().values for ts in sampled_ts]
        bp = ax.boxplot(data_for_box, positions=range(len(sampled_ts)), widths=0.6, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.6)
        
        ax.set_title(f"{col} Distribution Over Time", fontsize=10)
        ax.set_xlabel("Timestamp (sampled)")
        ax.set_ylabel("Value")
        ax.set_xticks(range(len(sampled_ts)))
        ax.set_xticklabels([str(ts)[:10] for ts in sampled_ts], rotation=45, ha="right", fontsize=7)
    
    fig.suptitle("Feature Distributions Across Time (showing drift)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = plots_dir / "feature_drift_boxplots.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_feature_heatmap(df: pd.DataFrame, feature_cols: list[str], plots_dir: Path):
    """
    Create a heatmap of z-scored feature means over time.
    Shows which features drift the most and when.
    """
    ts_means = df.groupby("TimeStamp")[feature_cols].mean()
    
    # Z-score each feature (column-wise)
    ts_means_z = (ts_means - ts_means.mean()) / (ts_means.std() + 1e-8)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(
        ts_means_z.T, 
        cmap="RdBu_r", 
        center=0,
        xticklabels=10,  # Show every 10th label
        yticklabels=True,
        ax=ax,
        cbar_kws={"label": "Z-score (deviation from mean)"}
    )
    ax.set_title("Feature Drift Heatmap (Z-scored means per timestamp)", fontsize=12)
    ax.set_xlabel("Timestamp Index")
    ax.set_ylabel("Feature")
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=8)
    
    fig.tight_layout()
    out = plots_dir / "feature_drift_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_rolling_statistics(df: pd.DataFrame, feature_cols: list[str], plots_dir: Path):
    """
    Show rolling mean and std for each feature to highlight drift patterns.
    """
    ts_means = df.groupby("TimeStamp")[feature_cols].mean()
    ts_std = df.groupby("TimeStamp")[feature_cols].std()
    
    n_features = len(feature_cols)
    fig, axes = plt.subplots(n_features, 1, figsize=(14, 2.5 * n_features), squeeze=False)
    
    window = max(5, len(ts_means) // 10)  # Rolling window
    
    for i, col in enumerate(feature_cols):
        ax = axes[i, 0]
        
        # Raw means
        ax.plot(ts_means.index, ts_means[col], alpha=0.4, label="Per-timestamp mean", color="gray")
        
        # Rolling mean
        rolling_mean = ts_means[col].rolling(window=window, min_periods=1).mean()
        ax.plot(ts_means.index, rolling_mean, label=f"Rolling mean (w={window})", color="blue", linewidth=2)
        
        # Rolling std as shaded region
        rolling_std = ts_std[col].rolling(window=window, min_periods=1).mean()
        ax.fill_between(
            ts_means.index,
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            alpha=0.2, color="blue", label="±1 std"
        )
        
        # Mark regions of high drift (where rolling std is high)
        overall_std = ts_std[col].mean()
        high_drift = rolling_std > 1.5 * overall_std
        if high_drift.any():
            ax.fill_between(
                ts_means.index, ax.get_ylim()[0], ax.get_ylim()[1],
                where=high_drift, alpha=0.1, color="red", label="High variance region"
            )
        
        ax.set_title(f"{col}", fontsize=10)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right", fontsize=7)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
    
    fig.suptitle("Rolling Statistics per Feature (detecting drift patterns)", fontsize=12, y=1.01)
    fig.tight_layout()
    out = plots_dir / "feature_drift_rolling_stats.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_drift_magnitude_ranking(df: pd.DataFrame, feature_cols: list[str], plots_dir: Path):
    """
    Bar chart ranking features by their drift magnitude.
    """
    ts_means = df.groupby("TimeStamp")[feature_cols].mean()
    overall_std = df[feature_cols].std().replace(0, 1)
    
    # Normalized drift = std of timestamp means / overall std
    normalised_drift = (ts_means.std() / overall_std).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["crimson" if v > 0.5 else "darkorange" if v > 0.3 else "steelblue" for v in normalised_drift.values]
    bars = ax.barh(range(len(normalised_drift)), normalised_drift.values, color=colors)
    ax.set_yticks(range(len(normalised_drift)))
    ax.set_yticklabels(normalised_drift.index, fontsize=9)
    ax.set_xlabel("Normalized Drift (std of ts-means / overall std)")
    ax.set_title("Feature Drift Magnitude Ranking\n(Higher = more drift over time)", fontsize=11)
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.7, label="High drift threshold")
    ax.axvline(0.3, color="orange", linestyle="--", alpha=0.7, label="Moderate drift threshold")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, normalised_drift.values)):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=8)
    
    fig.tight_layout()
    out = plots_dir / "feature_drift_ranking.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    
    return normalised_drift


def plot_pairwise_feature_correlation_drift(df: pd.DataFrame, feature_cols: list[str], plots_dir: Path):
    """
    Show how feature correlations change over time (correlation stability).
    """
    timestamps = sorted(df["TimeStamp"].unique())
    n_ts = len(timestamps)
    
    # Split into early vs late half
    mid = n_ts // 2
    early_ts = timestamps[:mid]
    late_ts = timestamps[mid:]
    
    df_early = df[df["TimeStamp"].isin(early_ts)]
    df_late = df[df["TimeStamp"].isin(late_ts)]
    
    corr_early = df_early[feature_cols].corr()
    corr_late = df_late[feature_cols].corr()
    corr_diff = corr_late - corr_early
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.heatmap(corr_early, cmap="RdBu_r", center=0, ax=axes[0], 
                xticklabels=True, yticklabels=True, square=True, vmin=-1, vmax=1)
    axes[0].set_title(f"Early Correlations\n(timestamps 1-{mid})", fontsize=10)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(axes[0].yaxis.get_majorticklabels(), fontsize=7)
    
    sns.heatmap(corr_late, cmap="RdBu_r", center=0, ax=axes[1],
                xticklabels=True, yticklabels=True, square=True, vmin=-1, vmax=1)
    axes[1].set_title(f"Late Correlations\n(timestamps {mid+1}-{n_ts})", fontsize=10)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(axes[1].yaxis.get_majorticklabels(), fontsize=7)
    
    sns.heatmap(corr_diff, cmap="PiYG", center=0, ax=axes[2],
                xticklabels=True, yticklabels=True, square=True, vmin=-0.5, vmax=0.5)
    axes[2].set_title("Correlation Change (Late - Early)\n(non-zero = relationship drift)", fontsize=10)
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(axes[2].yaxis.get_majorticklabels(), fontsize=7)
    
    fig.suptitle("Feature Correlation Stability Over Time", fontsize=12, y=1.02)
    fig.tight_layout()
    out = plots_dir / "feature_drift_correlation_stability.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def generate_drift_summary(df: pd.DataFrame, feature_cols: list[str], normalised_drift: pd.Series, plots_dir: Path):
    """
    Generate a text summary of feature drift findings.
    """
    summary = []
    summary.append("=" * 60)
    summary.append("FEATURE DRIFT ANALYSIS SUMMARY")
    summary.append("=" * 60)
    summary.append(f"\nDataset: {len(df)} rows, {len(df['TimeStamp'].unique())} timestamps")
    summary.append(f"Features analyzed: {len(feature_cols)}")
    
    mean_drift = normalised_drift.mean()
    summary.append(f"\nOverall mean normalized drift: {mean_drift:.4f}")
    summary.append(f"Feature Drift Score: {max(0, 1 - mean_drift):.4f}")
    
    if mean_drift > 0.5:
        summary.append("\n⚠️  SEVERE feature drift detected!")
    elif mean_drift > 0.3:
        summary.append("\n⚠️  HIGH feature drift detected")
    else:
        summary.append("\n✓ Feature drift is moderate or low")
    
    summary.append("\n--- Top Drifting Features ---")
    for feat, val in normalised_drift.head(5).items():
        status = "🔴" if val > 0.5 else "🟠" if val > 0.3 else "🟢"
        summary.append(f"  {status} {feat}: {val:.4f}")
    
    summary.append("\n--- Most Stable Features ---")
    for feat, val in normalised_drift.tail(3).items():
        summary.append(f"  🟢 {feat}: {val:.4f}")
    
    summary.append("\n--- Recommendations ---")
    high_drift_feats = normalised_drift[normalised_drift > 0.5].index.tolist()
    if high_drift_feats:
        summary.append(f"  1. High-drift features ({', '.join(high_drift_feats[:3])}) may need:")
        summary.append("     - Rolling/window normalization")
        summary.append("     - Per-timestamp standardization")
        summary.append("     - Input BatchNorm in the model (now added)")
    summary.append("  2. Consider seasonal decomposition for periodic features")
    summary.append("  3. Model's InputNorm layer should help stabilize these distributions")
    
    summary.append("\n" + "=" * 60)
    
    summary_text = "\n".join(summary)
    print(summary_text)
    
    out = plots_dir / "feature_drift_summary.txt"
    out.write_text(summary_text)
    print(f"\nSaved summary: {out}")


def main():
    # Find project root by looking for configs/default.json
    script_path = Path(__file__).resolve()
    base_path = script_path.parents[3]  # src/masters_research/pipeline -> project root
    
    # Verify we found the right path
    if not (base_path / "configs" / "default.json").exists():
        # Try alternate: maybe script is run from project root
        base_path = Path.cwd()
        if not (base_path / "configs" / "default.json").exists():
            print(f"Could not find project root. Tried: {script_path.parents[3]} and {Path.cwd()}")
            return
    
    plots_dir = base_path / "src" / "masters_research" / "artifacts" / "plots" / "feature_drift"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("FEATURE DRIFT VISUALIZATION")
    print("=" * 60)
    
    df = load_data(base_path)
    feature_cols = infer_feature_cols(df)
    print(f"Found {len(feature_cols)} feature columns: {feature_cols}")
    
    print("\nGenerating visualizations...")
    
    # Generate all plots
    plot_feature_heatmap(df, feature_cols, plots_dir)
    normalised_drift = plot_drift_magnitude_ranking(df, feature_cols, plots_dir)
    plot_rolling_statistics(df, feature_cols, plots_dir)
    plot_feature_distributions_over_time(df, feature_cols, plots_dir)
    plot_pairwise_feature_correlation_drift(df, feature_cols, plots_dir)
    
    # Generate summary
    generate_drift_summary(df, feature_cols, normalised_drift, plots_dir)
    
    print(f"\n✓ All visualizations saved to: {plots_dir}")


if __name__ == "__main__":
    main()
