import os
import argparse
import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Set, Dict


def propagate_features(graph: nx.Graph) -> nx.Graph:
    """
    Strict neighbor-only propagation:
    For any node with missing features (None), average features of neighbors
    (that have features) and assign the mean. If no neighbors have features,
    leave as None (no fallback).
    """
    for node in graph.nodes():
        feat = graph.nodes[node].get("feature", None)
        if feat is None:
            neighbors = list(graph.neighbors(node))
            neighbor_features = [graph.nodes[n].get("feature") for n in neighbors if graph.nodes[n].get("feature") is not None]
            if neighbor_features:
                graph.nodes[node]["feature"] = np.mean(neighbor_features, axis=0)
    return graph


def load_edges(edges_path: str) -> pd.DataFrame:
    edges = pd.read_csv(edges_path)
    return edges


def get_distance(edges_df: pd.DataFrame, n1: int, n2: int) -> float:
    row = edges_df[((edges_df.get('CompNo1') == n1) & (edges_df.get('CompNo2') == n2)) |
                   ((edges_df.get('CompNo1') == n2) & (edges_df.get('CompNo2') == n1))]
    if not row.empty:
        return float(row.iloc[0].get('distance_km', np.nan))
    return float('nan')


def build_feature_array(row: pd.Series, suffix: str) -> np.ndarray | None:
    cols = [
        f"AverageMaximumTemperature_{suffix}",
        f"AverageMinimunTemperature_{suffix}",
        f"AverageMaximumRelativeHumidity_{suffix}",
        f"AverageMinimumRelativeHumidity_{suffix}",
        f"TotalRainfall_{suffix}",
        f"AverageTotalRadiation_{suffix}",
        f"AverageWindSpeed_{suffix}",
        f"AverageTotalRelativeEvaporation_{suffix}",
        f"Target_{suffix}",
    ]
    values = [row.get(c) for c in cols]
    # If the key feature (first one) is NaN, treat as missing
    if pd.isna(values[0]):
        return None
    return np.array(values, dtype=float)


def process_timestamp(timestamp_data: pd.DataFrame, edges_df: pd.DataFrame, required_nodes: Set[int],
                     impute_threshold_count: int | None = None) -> nx.Graph:
    G = nx.Graph()

    # Add nodes and edges from existing pairs, restricted to required_nodes
    for _, row in timestamp_data.iterrows():
        x_node = int(row['Compno_x'])
        y_node = int(row['Compno_y'])
        if x_node not in required_nodes or y_node not in required_nodes:
            continue
        x_feat = build_feature_array(row, 'x')
        y_feat = build_feature_array(row, 'y')

        if x_node not in G:
            G.add_node(x_node, feature=x_feat)
        else:
            # Prefer existing feature if present, otherwise update
            if G.nodes[x_node].get('feature') is None and x_feat is not None:
                G.nodes[x_node]['feature'] = x_feat
        if y_node not in G:
            G.add_node(y_node, feature=y_feat)
        else:
            if G.nodes[y_node].get('feature') is None and y_feat is not None:
                G.nodes[y_node]['feature'] = y_feat

        dist = row.get('distance_km', np.nan)
        G.add_edge(x_node, y_node, distance=dist)
        G.add_edge(y_node, x_node, distance=dist)

    # Ensure required nodes exist (missing => feature None)
    for node in required_nodes:
        if node not in G:
            G.add_node(node, feature=None)

    # For missing nodes, connect to all other required nodes with known edge distances
    for node in required_nodes:
        # Missing if feature is None
        if G.nodes[node].get('feature') is None:
            for other in required_nodes:
                if other == node:
                    continue
                if other not in G:
                    G.add_node(other, feature=None)
                dist = get_distance(edges_df, node, other)
                G.add_edge(node, other, distance=dist)
                G.add_edge(other, node, distance=dist)

    # Apply strict neighbor propagation only if present nodes among required_nodes are below threshold
    if impute_threshold_count is not None:
        present_required = [n for n in required_nodes if G.nodes[n].get('feature') is not None]
        if len(present_required) < impute_threshold_count:
            G = propagate_features(G)
    else:
        # No gating provided, always propagate
        G = propagate_features(G)
    return G


def graph_to_rows(G: nx.Graph, timestamp) -> List[Dict]:
    # Build a nodes DataFrame-like access
    rows: List[Dict] = []
    # For each edge, create two rows (x,y) and (y,x)
    for n1, n2, data in G.edges(data=True):
        feat1 = G.nodes[n1].get('feature')
        feat2 = G.nodes[n2].get('feature')
        # If any features remain None, keep NaNs
        def feat_val(f, idx):
            return (None if f is None else float(f[idx]))
        rows.append({
            'TimeStamp': timestamp,
            'Compno_x': n1,
            'Compno_y': n2,
            'distance_km': data.get('distance', None),
            'AverageMaximumTemperature_x': feat_val(feat1, 0),
            'AverageMinimunTemperature_x': feat_val(feat1, 1),
            'AverageMaximumRelativeHumidity_x': feat_val(feat1, 2),
            'AverageMinimumRelativeHumidity_x': feat_val(feat1, 3),
            'TotalRainfall_x': feat_val(feat1, 4),
            'AverageTotalRadiation_x': feat_val(feat1, 5),
            'AverageWindSpeed_x': feat_val(feat1, 6),
            'AverageTotalRelativeEvaporation_x': feat_val(feat1, 7),
            'Target_x': feat_val(feat1, 8),
            'AverageMaximumTemperature_y': feat_val(feat2, 0),
            'AverageMinimunTemperature_y': feat_val(feat2, 1),
            'AverageMaximumRelativeHumidity_y': feat_val(feat2, 2),
            'AverageMinimumRelativeHumidity_y': feat_val(feat2, 3),
            'TotalRainfall_y': feat_val(feat2, 4),
            'AverageTotalRadiation_y': feat_val(feat2, 5),
            'AverageWindSpeed_y': feat_val(feat2, 6),
            'AverageTotalRelativeEvaporation_y': feat_val(feat2, 7),
            'Target_y': feat_val(feat2, 8),
        })
        rows.append({
            'TimeStamp': timestamp,
            'Compno_x': n2,
            'Compno_y': n1,
            'distance_km': data.get('distance', None),
            'AverageMaximumTemperature_x': feat_val(feat2, 0),
            'AverageMinimunTemperature_x': feat_val(feat2, 1),
            'AverageMaximumRelativeHumidity_x': feat_val(feat2, 2),
            'AverageMinimumRelativeHumidity_x': feat_val(feat2, 3),
            'TotalRainfall_x': feat_val(feat2, 4),
            'AverageTotalRadiation_x': feat_val(feat2, 5),
            'AverageWindSpeed_x': feat_val(feat2, 6),
            'AverageTotalRelativeEvaporation_x': feat_val(feat2, 7),
            'Target_x': feat_val(feat2, 8),
            'AverageMaximumTemperature_y': feat_val(feat1, 0),
            'AverageMinimunTemperature_y': feat_val(feat1, 1),
            'AverageMaximumRelativeHumidity_y': feat_val(feat1, 2),
            'AverageMinimumRelativeHumidity_y': feat_val(feat1, 3),
            'TotalRainfall_y': feat_val(feat1, 4),
            'AverageTotalRadiation_y': feat_val(feat1, 5),
            'AverageWindSpeed_y': feat_val(feat1, 6),
            'AverageTotalRelativeEvaporation_y': feat_val(feat1, 7),
            'Target_y': feat_val(feat1, 8),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Graph-based neighbor-only feature propagation per timestamp (no fallbacks).")
    parser.add_argument("--base_path", type=str, default="../..", help="Base path to project root.")
    parser.add_argument("--final_df_name", type=str, default="final_df.csv", help="Input edge-pair dataset filename in TransientData.")
    parser.add_argument("--timestamp_filter_name", type=str, default="final_df1.csv", help="Optional CSV with TimeStamp column to filter timestamps.")
    parser.add_argument("--edges_name", type=str, default="edges_df.csv", help="Edges CSV providing distance_km and CompNo1/CompNo2.")
    parser.add_argument("--output_name", type=str, default="corrected_df.csv", help="Output filename for propagated dataset.")
    parser.add_argument("--max_nodes", type=int, default=None, help="Maximum number of canonical nodes to enforce per timestamp (size of required node set).")
    parser.add_argument("--impute_percent", type=float, default=1.0, help="Percentage of --max_nodes presence below which to run neighbor-only imputation. Default 1.0 (always propagate).")
    args = parser.parse_args()

    transient_dir = os.path.abspath(os.path.join(args.base_path, "Dataset", "TransientData"))
    final_df_path = os.path.join(transient_dir, args.final_df_name)
    edges_path = os.path.join(transient_dir, args.edges_name)
    filter_path = os.path.join(transient_dir, args.timestamp_filter_name)

    df = pd.read_csv(final_df_path)
    edges_df = load_edges(edges_path)

    # Optional timestamp filter
    if os.path.exists(filter_path):
        filter_df = pd.read_csv(filter_path)
        if 'TimeStamp' in filter_df.columns:
            df = df[df['TimeStamp'].isin(filter_df['TimeStamp'].unique())]

    # Infer required nodes from data and cap to --max_nodes if given
    all_nodes = pd.concat([
        pd.to_numeric(df['Compno_x'], errors='coerce').dropna().astype(int),
        pd.to_numeric(df['Compno_y'], errors='coerce').dropna().astype(int)
    ])
    # Rank nodes by global frequency
    freq = all_nodes.value_counts()
    if args.max_nodes is not None and args.max_nodes > 0:
        required_nodes = set(freq.head(args.max_nodes).index.tolist())
    else:
        required_nodes = set(freq.index.tolist())

    corrected_rows: List[Dict] = []
    impute_flags: List[Dict] = []
    timestamps = list(df['TimeStamp'].unique())
    for ts in timestamps:
        ts_df = df[df['TimeStamp'] == ts]
        # Compute present count among required nodes for gating
        present_nodes = set(pd.to_numeric(ts_df['Compno_x'], errors='coerce').dropna().astype(int).unique()).union(
            set(pd.to_numeric(ts_df['Compno_y'], errors='coerce').dropna().astype(int).unique()))
        present_required_count = len(present_nodes.intersection(required_nodes))
        threshold_count = None
        if args.max_nodes is not None and args.max_nodes > 0:
            threshold_count = int(args.max_nodes * float(args.impute_percent))
        elif len(required_nodes) > 0:
            threshold_count = int(len(required_nodes) * float(args.impute_percent))

        G = process_timestamp(ts_df, edges_df, required_nodes, threshold_count)
        did_impute = (threshold_count is None) or (present_required_count < threshold_count)
        impute_flags.append({'TimeStamp': ts, 'PresentRequired': present_required_count, 'Threshold': threshold_count, 'Imputed': did_impute})
        corrected_rows.extend(graph_to_rows(G, ts))

    corrected_df = pd.DataFrame(corrected_rows)

    # Sort for consistency
    corrected_df = corrected_df.sort_values(by=['TimeStamp', 'Compno_x', 'Compno_y']).reset_index(drop=True)

    # Save output
    out_path = os.path.join(transient_dir, args.output_name)
    corrected_df.to_csv(out_path, index=False)

    # Save imputation flags summary next to output
    flags_path = os.path.join(transient_dir, os.path.splitext(args.output_name)[0] + "_impute_flags.csv")
    pd.DataFrame(impute_flags).sort_values('TimeStamp').to_csv(flags_path, index=False)

    print("--- Graph Propagation Summary ---")
    print(f"Timestamps processed: {len(timestamps)}")
    print(f"Required nodes: {len(required_nodes)}")
    print(f"Rows written: {len(corrected_df)}")
    print(f"Saved to: {out_path}")
    print(f"Imputation flags saved: {flags_path}")


if __name__ == "__main__":
    main()
