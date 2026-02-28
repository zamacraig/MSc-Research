"""
Trace and visualize the data pipeline to understand how features transform
at each stage before reaching the model.

This helps understand WHY feature drift is measured before vs after scaling.

Run from project root:
    python src/masters_research/pipeline/trace_data_pipeline.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


def load_raw_data(base_path: Path) -> pd.DataFrame:
    """Load the raw/semi-processed CSV that train.py loads."""
    # This is what config data_path points to
    data_path = base_path / "Dataset" / "TransientData" / "processedData.csv"
    if not data_path.exists():
        data_path = base_path / "Dataset" / "TransientData" / "final_df.csv"
    print(f"[Stage 1] Loading raw data from: {data_path}")
    return pd.read_csv(data_path)


def apply_minmax_scaling(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply the same MinMaxScaler that create_temporal_dataset uses."""
    df = df.copy()
    
    features_to_scale = [
        col for col in df.columns
        if "_x" in col and "Compno" not in col and "Target" not in col and "TimeStamp" not in col
    ]
    
    scaler = MinMaxScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    print(f"[Stage 2] MinMaxScaler applied to: {features_to_scale}")
    return df, features_to_scale


def compute_drift_score(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Compute normalized drift score for each feature."""
    ts_means = df.groupby("TimeStamp")[feature_cols].mean()
    overall_std = df[feature_cols].std().replace(0, 1)
    normalised_drift = (ts_means.std() / overall_std).fillna(0)
    return {
        "per_feature": normalised_drift.to_dict(),
        "mean_drift": float(normalised_drift.mean()),
        "ts_means": ts_means,
    }


def plot_pipeline_comparison(
    raw_df: pd.DataFrame,
    scaled_df: pd.DataFrame,
    feature_cols: list[str],
    plots_dir: Path
):
    """
    Create side-by-side comparison of raw vs scaled feature distributions over time.
    """
    n_features = len(feature_cols)
    fig, axes = plt.subplots(n_features, 2, figsize=(16, 3 * n_features))
    
    for i, col in enumerate(feature_cols):
        # Raw data (left)
        ts_means_raw = raw_df.groupby("TimeStamp")[col].mean()
        axes[i, 0].plot(range(len(ts_means_raw)), ts_means_raw.values, 
                        color="steelblue", linewidth=1.5, alpha=0.8)
        axes[i, 0].fill_between(range(len(ts_means_raw)), ts_means_raw.values, alpha=0.3)
        axes[i, 0].set_title(f"{col}\n[RAW - before MinMaxScaler]", fontsize=9)
        axes[i, 0].set_ylabel("Mean value")
        if i == 0:
            axes[i, 0].annotate("STAGE 1: Raw CSV Data", xy=(0.5, 1.15), 
                               xycoords='axes fraction', ha='center', fontsize=11, fontweight='bold')
        
        # Scaled data (right)  
        ts_means_scaled = scaled_df.groupby("TimeStamp")[col].mean()
        axes[i, 1].plot(range(len(ts_means_scaled)), ts_means_scaled.values,
                        color="crimson", linewidth=1.5, alpha=0.8)
        axes[i, 1].fill_between(range(len(ts_means_scaled)), ts_means_scaled.values, alpha=0.3, color="crimson")
        axes[i, 1].set_title(f"{col}\n[SCALED - after MinMaxScaler]", fontsize=9)
        axes[i, 1].set_ylabel("Mean value (0-1)")
        axes[i, 1].set_ylim(-0.05, 1.05)
        if i == 0:
            axes[i, 1].annotate("STAGE 2: What Model Sees", xy=(0.5, 1.15),
                               xycoords='axes fraction', ha='center', fontsize=11, fontweight='bold')
    
    axes[-1, 0].set_xlabel("Timestamp Index")
    axes[-1, 1].set_xlabel("Timestamp Index")
    
    fig.suptitle("Data Pipeline: Feature Drift at Each Stage\n"
                 "(Seasonal patterns persist AFTER scaling because MinMaxScaler is global)", 
                 fontsize=12, y=1.02)
    fig.tight_layout()
    
    out = plots_dir / "pipeline_raw_vs_scaled.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_drift_comparison(raw_drift: dict, scaled_drift: dict, feature_cols: list[str], plots_dir: Path):
    """
    Bar chart comparing drift scores before vs after scaling.
    """
    features = list(feature_cols)
    raw_scores = [raw_drift["per_feature"].get(f, 0) for f in features]
    scaled_scores = [scaled_drift["per_feature"].get(f, 0) for f in features]
    
    x = np.arange(len(features))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, raw_scores, width, label='Before Scaling (Raw)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, scaled_scores, width, label='After Scaling (Model Input)', color='crimson', alpha=0.8)
    
    ax.set_ylabel('Normalized Drift Score')
    ax.set_title('Feature Drift: Raw Data vs Model Input\n(MinMaxScaler does NOT reduce temporal drift)')
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_x", "") for f in features], rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.7, label='High drift threshold')
    
    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=7)
    
    fig.tight_layout()
    out = plots_dir / "pipeline_drift_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def create_pipeline_diagram(plots_dir: Path):
    """
    Create a visual pipeline diagram showing data flow.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(8, 9.5, "DATA PIPELINE FLOW", ha='center', fontsize=16, fontweight='bold')
    
    # Boxes with stages
    boxes = [
        # (x, y, width, height, text, color, details)
        (0.5, 7, 3, 1.5, "STAGE 1\nRaw CSV", "#E3F2FD", 
         "processedData.csv\n• Weather measurements\n• High variance across seasons"),
        
        (5, 7, 3.5, 1.5, "STAGE 2\nMinMaxScaler", "#FFF3E0",
         "create_temporal_dataset()\n• Scales to 0-1 range\n• GLOBAL fit (all timestamps)\n• Drift patterns PRESERVED"),
        
        (10, 7, 3.5, 1.5, "STAGE 3\nMonth Encoding", "#E8F5E9",
         "sin/cos encoding added\n• month_sin, month_cos\n• Explicit seasonal signal"),
        
        (5, 3.5, 3.5, 1.5, "STAGE 4\nModel InputNorm", "#FCE4EC",
         "BatchNorm1d on inputs\n• LEARNS per-feature stats\n• Adapts to drift DURING training"),
        
        (10, 3.5, 3.5, 1.5, "STAGE 5\nGCN + GRU", "#F3E5F5",
         "Model forward pass\n• Spatial convolutions\n• Temporal GRU state"),
    ]
    
    for (x, y, w, h, title, color, details) in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.3, title, ha='center', va='top', fontsize=11, fontweight='bold')
        ax.text(x + w/2, y + h/2 - 0.2, details, ha='center', va='center', fontsize=8, 
                linespacing=1.3, family='monospace')
    
    # Arrows
    ax.annotate('', xy=(5, 7.75), xytext=(3.5, 7.75),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(10, 7.75), xytext=(8.5, 7.75),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(6.75, 5), xytext=(6.75, 7),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(10, 4.25), xytext=(8.5, 4.25),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Problem annotation
    ax.add_patch(plt.Rectangle((0.5, 1), 15, 2, facecolor='#FFEBEE', edgecolor='#C62828', linewidth=2, linestyle='--'))
    ax.text(8, 2.6, "⚠️ THE PROBLEM: WHY FEATURE DRIFT PERSISTS", ha='center', fontsize=12, fontweight='bold', color='#C62828')
    ax.text(8, 1.5, 
            "MinMaxScaler uses GLOBAL min/max across ALL timestamps.\n"
            "This maps values to 0-1 but PRESERVES the temporal pattern (seasonality).\n"
            "The model receives features that still drift over time → InputNorm is critical.",
            ha='center', fontsize=10, family='monospace', linespacing=1.5)
    
    fig.tight_layout()
    out = plots_dir / "pipeline_diagram.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def print_summary(raw_drift: dict, scaled_drift: dict, feature_cols: list[str]):
    """Print a summary comparing drift at each stage."""
    print("\n" + "=" * 70)
    print("DATA PIPELINE DRIFT ANALYSIS")
    print("=" * 70)
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                    DRIFT SCORES BY PIPELINE STAGE                   │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│ {'Feature':<35} {'Raw':<12} {'Scaled':<12} {'Change':<10} │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    
    for col in feature_cols:
        raw = raw_drift["per_feature"].get(col, 0)
        scaled = scaled_drift["per_feature"].get(col, 0)
        change = scaled - raw
        change_str = f"{change:+.3f}" if abs(change) > 0.001 else "~0"
        print(f"│ {col:<35} {raw:<12.3f} {scaled:<12.3f} {change_str:<10} │")
    
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│ {'MEAN DRIFT':<35} {raw_drift['mean_drift']:<12.3f} {scaled_drift['mean_drift']:<12.3f}            │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    print("\n📊 KEY INSIGHT:")
    print("   MinMaxScaler scales features to 0-1 but does NOT reduce drift.")
    print("   The seasonal patterns in weather data are preserved after scaling.")
    print("   This is why InputNorm (BatchNorm on raw model inputs) is needed —")
    print("   it learns running statistics DURING training to adapt to drift.")
    
    print("\n🔧 SOLUTION APPLIED:")
    print("   1. InputNorm layer added to model (normalizes per-batch)")
    print("   2. Month sin/cos encoding (explicit seasonal signal)")
    print("   3. GRU temporal layer (learns from sequential context)")
    print("=" * 70)


def main():
    # Find project root
    script_path = Path(__file__).resolve()
    base_path = script_path.parents[3]
    
    if not (base_path / "configs" / "default.json").exists():
        base_path = Path.cwd()
    
    plots_dir = base_path / "src" / "masters_research" / "artifacts" / "plots" / "pipeline_trace"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("DATA PIPELINE TRACE")
    print("=" * 70)
    
    # Stage 1: Load raw data
    raw_df = load_raw_data(base_path)
    
    # Get feature columns
    feature_cols = [
        col for col in raw_df.columns
        if "_x" in col and "Compno" not in col and "Target" not in col and "TimeStamp" not in col
    ]
    print(f"   Features: {feature_cols}")
    
    # Compute drift on raw data
    raw_drift = compute_drift_score(raw_df, feature_cols)
    print(f"   Raw mean drift: {raw_drift['mean_drift']:.4f}")
    
    # Stage 2: Apply MinMaxScaler (same as create_temporal_dataset)
    scaled_df, _ = apply_minmax_scaling(raw_df)
    
    # Compute drift on scaled data
    scaled_drift = compute_drift_score(scaled_df, feature_cols)
    print(f"   Scaled mean drift: {scaled_drift['mean_drift']:.4f}")
    
    # Generate visualizations
    print("\n[Generating visualizations...]")
    create_pipeline_diagram(plots_dir)
    plot_pipeline_comparison(raw_df, scaled_df, feature_cols, plots_dir)
    plot_drift_comparison(raw_drift, scaled_drift, feature_cols, plots_dir)
    
    # Print summary
    print_summary(raw_drift, scaled_drift, feature_cols)
    
    print(f"\n✓ All visualizations saved to: {plots_dir}")


if __name__ == "__main__":
    main()
