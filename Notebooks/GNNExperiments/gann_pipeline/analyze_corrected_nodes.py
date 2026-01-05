import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Set, Dict

BASE_FEATURES = [
    'AverageMaximumTemperature',
    'AverageMinimunTemperature',
    'AverageMaximumRelativeHumidity',
    'AverageMinimumRelativeHumidity',
    'TotalRainfall',
    'AverageTotalRadiation',
    'AverageWindSpeed',
    'AverageTotalRelativeEvaporation',
    'Target',
]


def load_corrected_df(base_path: str, filename: str) -> pd.DataFrame:
    transient_dir = os.path.abspath(os.path.join(base_path, 'Dataset', 'TransientData'))
    path = os.path.join(transient_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"corrected_df not found: {path}")
    df = pd.read_csv(path)
    return df


def to_long_node_features(df: pd.DataFrame) -> pd.DataFrame:
    # Build long format from x-side
    x_cols = {f"{b}_x": b for b in BASE_FEATURES}
    x_select = ['TimeStamp', 'Compno_x'] + list(x_cols.keys())
    df_x = df[x_select].rename(columns={'Compno_x': 'Node', **x_cols})
    # Build long format from y-side
    y_cols = {f"{b}_y": b for b in BASE_FEATURES}
    y_select = ['TimeStamp', 'Compno_y'] + list(y_cols.keys())
    df_y = df[y_select].rename(columns={'Compno_y': 'Node', **y_cols})
    # Concatenate and average per (TimeStamp, Node)
    long_df = pd.concat([df_x, df_y], axis=0, ignore_index=True)
    # Coerce types
    long_df['Node'] = pd.to_numeric(long_df['Node'], errors='coerce').astype('Int64')
    long_df['TimeStamp'] = pd.to_datetime(long_df['TimeStamp'], errors='coerce')
    grouped = long_df.groupby(['TimeStamp', 'Node'], as_index=False).mean()
    return grouped


def analyze_nodes(df: pd.DataFrame, required_nodes: Set[int] | None = None) -> Dict[str, pd.DataFrame]:
    # Node set per timestamp and missing nodes
    ts_groups = df.groupby('TimeStamp')
    node_counts = ts_groups['Node'].nunique().rename('NodeCount')
    ts_metrics = pd.DataFrame({'TimeStamp': node_counts.index, 'NodeCount': node_counts.values})

    if required_nodes:
        missing_info = []
        for ts, ts_df in ts_groups:
            present = set(ts_df['Node'].dropna().astype(int).unique())
            missing = sorted(list(required_nodes - present))
            missing_info.append({'TimeStamp': ts, 'MissingCount': len(missing), 'MissingNodes': missing})
        miss_df = pd.DataFrame(missing_info)
        ts_metrics = ts_metrics.merge(miss_df, on='TimeStamp', how='left')
    else:
        ts_metrics['MissingCount'] = np.nan
        ts_metrics['MissingNodes'] = [[] for _ in range(len(ts_metrics))]

    # Per-node availability
    total_ts = df['TimeStamp'].nunique()
    availability = df.groupby('Node')['TimeStamp'].nunique().rename('TimestampsPresent')
    availability_df = availability.reset_index()
    availability_df['TotalTimestamps'] = total_ts
    availability_df['AvailabilityPercent'] = (availability_df['TimestampsPresent'] / total_ts * 100).round(2)
    availability_df = availability_df.sort_values('AvailabilityPercent', ascending=False)

    # Feature NaN summary across long df
    nan_summary_rows = []
    for b in BASE_FEATURES:
        nan_count = int(df[b].isna().sum())
        nan_rate = nan_count / len(df) if len(df) else 0.0
        nan_summary_rows.append({'Feature': b, 'NaNCount': nan_count, 'NaNRate': nan_rate})
    nan_summary_df = pd.DataFrame(nan_summary_rows)

    return {
        'ts_metrics': ts_metrics.sort_values('TimeStamp'),
        'availability': availability_df,
        'nan_summary': nan_summary_df.sort_values('NaNRate', ascending=False),
    }


def plot_summaries(ts_metrics: pd.DataFrame, nan_summary: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    # Nodes per timestamp
    plt.figure(figsize=(14, 6))
    plt.bar(ts_metrics['TimeStamp'], ts_metrics['NodeCount'], width=20)
    plt.title('Nodes per Timestamp (Corrected)')
    plt.xlabel('Timestamp')
    plt.ylabel('Number of Nodes')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    p_nodes = os.path.join(output_dir, 'nodes_per_timestamp_corrected.png')
    plt.tight_layout()
    plt.savefig(p_nodes)
    plt.close()

    # Missing count per timestamp if present
    if 'MissingCount' in ts_metrics.columns and ts_metrics['MissingCount'].notna().any():
        plt.figure(figsize=(14, 4))
        plt.plot(ts_metrics['TimeStamp'], ts_metrics['MissingCount'], color='tab:red')
        plt.title('Missing Nodes per Timestamp (Corrected)')
        plt.xlabel('Timestamp')
        plt.ylabel('Missing Nodes')
        plt.grid(True, linestyle='--', alpha=0.5)
        p_missing = os.path.join(output_dir, 'missing_nodes_per_timestamp_corrected.png')
        plt.tight_layout()
        plt.savefig(p_missing)
        plt.close()
    else:
        p_missing = None

    # Feature NaN rates bar
    plt.figure(figsize=(10, 5))
    plt.bar(nan_summary['Feature'], nan_summary['NaNRate'])
    plt.title('Feature NaN Rate (Corrected Long)')
    plt.xlabel('Feature')
    plt.ylabel('NaN Rate')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    p_nan = os.path.join(output_dir, 'feature_nan_rate_corrected.png')
    plt.tight_layout()
    plt.savefig(p_nan)
    plt.close()

    return {
        'nodes_per_timestamp': p_nodes,
        'missing_nodes_per_timestamp': p_missing,
        'feature_nan_rate': p_nan,
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze nodes after graph-based neighbor-only cleaning.')
    parser.add_argument('--base_path', type=str, default='../../../', help='Base path to project root.')
    parser.add_argument('--corrected_name', type=str, default='corrected_df.csv', help='Corrected dataset filename.')
    parser.add_argument('--required_nodes', type=int, nargs='*', default=[30688,30142,30173,30191,30389,20272,30692,20197,30899], help='Canonical node set.')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save plots.')
    parser.add_argument('--save_availability', type=str, default='corrected_availability_summary.csv', help='Availability CSV filename.')
    parser.add_argument('--save_ts_metrics', type=str, default='corrected_timestamp_metrics.csv', help='Timestamp metrics CSV filename.')
    parser.add_argument('--save_nan_summary', type=str, default='corrected_feature_nan_summary.csv', help='Feature NaN summary CSV filename.')
    args = parser.parse_args()

    df_pairs = load_corrected_df(args.base_path, args.corrected_name)

    # Build long per-node feature table
    long_df = to_long_node_features(df_pairs)

    # Analyze
    results = analyze_nodes(long_df, set(args.required_nodes))

    # Save CSVs
    transient_dir = os.path.abspath(os.path.join(args.base_path, 'Dataset', 'TransientData'))
    os.makedirs(transient_dir, exist_ok=True)
    avail_path = os.path.join(transient_dir, args.save_availability)
    ts_metrics_path = os.path.join(transient_dir, args.save_ts_metrics)
    nan_summary_path = os.path.join(transient_dir, args.save_nan_summary)
    results['availability'].to_csv(avail_path, index=False)
    results['ts_metrics'].to_csv(ts_metrics_path, index=False)
    results['nan_summary'].to_csv(nan_summary_path, index=False)

    # Save plots next to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    plot_paths = plot_summaries(results['ts_metrics'], results['nan_summary'], out_dir)

    print('--- Post-Cleaning Node Analysis ---')
    print(f"Kept timestamps: {results['ts_metrics']['TimeStamp'].nunique()}")
    print(f"Mean nodes/timestamp: {results['ts_metrics']['NodeCount'].mean():.2f} | min: {results['ts_metrics']['NodeCount'].min()} | max: {results['ts_metrics']['NodeCount'].max()}")
    print(f"Top availability (first 5):\n{results['availability'].head(5).to_string(index=False)}")
    print(f"Feature NaN summary saved to: {nan_summary_path}")
    print('Plots:')
    for k, p in plot_paths.items():
        if p:
            print(f" - {k}: {p}")
    print(f"Availability saved: {avail_path}")
    print(f"Timestamp metrics saved: {ts_metrics_path}")


if __name__ == '__main__':
    main()
