import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def analyze_cleaned(cleaned_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze cleaned dataset: compute per-node availability, per-timestamp node counts,
    and imputation metrics.

    Returns:
    - availability_df: per-node availability summary
    - ts_metrics_df: per-timestamp metrics (node count, imputed count, imputation rate)
    """
    if "TimeStamp" not in cleaned_df.columns or "Compno" not in cleaned_df.columns:
        raise ValueError("cleaned_df must contain 'TimeStamp' and 'Compno' columns")

    df = cleaned_df.copy()
    if not np.issubdtype(df["TimeStamp"].dtype, np.datetime64):
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")

    # Per-timestamp metrics
    ts_group = df.groupby("TimeStamp")
    node_count = ts_group["Compno"].nunique()
    imputed_count = ts_group["Imputed"].sum() if "Imputed" in df.columns else pd.Series(0, index=node_count.index)
    total_rows = ts_group.size()
    imputation_rate = (imputed_count / total_rows).fillna(0)

    ts_metrics_df = pd.DataFrame({
        "TimeStamp": node_count.index,
        "NodeCount": node_count.values,
        "ImputedCount": imputed_count.values,
        "TotalRows": total_rows.values,
        "ImputationRate": imputation_rate.values,
    }).sort_values("TimeStamp")

    # Per-node availability
    total_ts = df["TimeStamp"].nunique()
    presence = df.groupby(["Compno", "TimeStamp"]).size().reset_index().drop(columns=0)
    availability = presence.groupby("Compno")["TimeStamp"].nunique()
    availability_rate = (availability / total_ts * 100).round(2)

    availability_df = pd.DataFrame({
        "Compno": availability.index,
        "TimestampsPresent": availability.values,
        "TotalTimestamps": total_ts,
        "AvailabilityPercent": availability_rate.values,
    }).sort_values("AvailabilityPercent", ascending=False)

    return availability_df, ts_metrics_df


def plot_post_cleanup(ts_metrics_df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Plot nodes per timestamp
    plt.figure(figsize=(14, 6))
    plt.bar(ts_metrics_df["TimeStamp"], ts_metrics_df["NodeCount"], width=20)
    plt.title("Nodes per Timestamp (Post-Pruning)")
    plt.xlabel("Timestamp")
    plt.ylabel("Number of Nodes")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    path_nodes = os.path.join(output_dir, "nodes_per_timestamp_cleaned.png")
    plt.tight_layout()
    plt.savefig(path_nodes)
    plt.close()

    # Plot imputed count per timestamp
    plt.figure(figsize=(14, 4))
    plt.plot(ts_metrics_df["TimeStamp"], ts_metrics_df["ImputedCount"], color="tab:orange")
    plt.title("Imputed Rows per Timestamp")
    plt.xlabel("Timestamp")
    plt.ylabel("Imputed Rows")
    plt.grid(True, linestyle='--', alpha=0.5)
    path_imp = os.path.join(output_dir, "imputed_rows_per_timestamp.png")
    plt.tight_layout()
    plt.savefig(path_imp)
    plt.close()

    # Plot imputation rate per timestamp
    plt.figure(figsize=(14, 4))
    plt.plot(ts_metrics_df["TimeStamp"], ts_metrics_df["ImputationRate"], color="tab:green")
    plt.title("Imputation Rate per Timestamp")
    plt.xlabel("Timestamp")
    plt.ylabel("Imputation Rate")
    plt.grid(True, linestyle='--', alpha=0.5)
    path_rate = os.path.join(output_dir, "imputation_rate_per_timestamp.png")
    plt.tight_layout()
    plt.savefig(path_rate)
    plt.close()

    return {
        "nodes_per_timestamp": path_nodes,
        "imputed_rows_per_timestamp": path_imp,
        "imputation_rate_per_timestamp": path_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze dataset after pruning and feature propagation.")
    parser.add_argument("--base_path", type=str, default="../..", help="Base path to project root.")
    parser.add_argument("--cleaned_name", type=str, default="cleaned_df.csv", help="Cleaned dataset filename in TransientData.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save plots; defaults to script folder.")
    parser.add_argument("--save_availability", type=str, default="availability_summary.csv", help="Filename for availability summary CSV in TransientData.")
    parser.add_argument("--save_ts_metrics", type=str, default="timestamp_metrics.csv", help="Filename for timestamp metrics CSV in TransientData.")
    args = parser.parse_args()

    transient_dir = os.path.abspath(os.path.join(args.base_path, "Dataset", "TransientData"))
    cleaned_path = os.path.join(transient_dir, args.cleaned_name)

    print("--- Loading cleaned dataset ---")
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned dataset not found: {cleaned_path}")

    cleaned_df = pd.read_csv(cleaned_path)

    availability_df, ts_metrics_df = analyze_cleaned(cleaned_df)

    # Save CSV summaries
    avail_path = os.path.join(transient_dir, args.save_availability)
    ts_metrics_path = os.path.join(transient_dir, args.save_ts_metrics)
    availability_df.to_csv(avail_path, index=False)
    ts_metrics_df.to_csv(ts_metrics_path, index=False)

    # Save plots
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    plot_paths = plot_post_cleanup(ts_metrics_df, out_dir)

    print("--- Post-Cleanup Analysis Summary ---")
    print(f"Kept timestamps: {ts_metrics_df['TimeStamp'].nunique()}")
    print(f"Mean nodes/timestamp: {ts_metrics_df['NodeCount'].mean():.2f} | min: {ts_metrics_df['NodeCount'].min()} | max: {ts_metrics_df['NodeCount'].max()}")
    total_imputed = int(ts_metrics_df['ImputedCount'].sum())
    print(f"Total imputed rows: {total_imputed}")
    print(f"Availability saved to: {avail_path}")
    print(f"Timestamp metrics saved to: {ts_metrics_path}")
    print("Plots:")
    for k, p in plot_paths.items():
        print(f" - {k}: {p}")


if __name__ == "__main__":
    main()
