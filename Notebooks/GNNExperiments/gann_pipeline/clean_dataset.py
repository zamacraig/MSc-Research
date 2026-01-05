import os
import argparse
import pandas as pd
import numpy as np
from typing import Tuple, Set, List, Dict

from data_loader import load_and_preprocess_data


def prune_and_impute(
    attribute_df: pd.DataFrame,
    node_df: pd.DataFrame,
    max_missing: int = 2,
    max_nodes: int | None = None,
    prune_percent: float | None = None,
    comparator: str = "gt",
    propagation: str = "neighbor_mean",
    edges_path: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prune timestamps where missing nodes exceed `max_missing` and impute
    missing nodes for remaining timestamps using per-timestamp numeric means.

    Returns:
    - cleaned_df: DataFrame with pruned timestamps and imputed rows added.
    - pruned_report: DataFrame with timestamps and missing counts that were pruned.
    """
    # Ensure expected columns exist
    required_cols = {"TimeStamp", "Compno"}
    missing_required = required_cols - set(attribute_df.columns)
    if missing_required:
        raise ValueError(f"attribute_df missing required columns: {missing_required}")

    if "ComputerNo" not in node_df.columns:
        raise ValueError("node_df missing required column: 'ComputerNo'")

    # Normalize dtypes
    attribute_df = attribute_df.copy()
    # Coerce to datetime if needed
    if not np.issubdtype(attribute_df["TimeStamp"].dtype, np.datetime64):
        attribute_df["TimeStamp"] = pd.to_datetime(attribute_df["TimeStamp"], errors="coerce")
    # Coerce Compno to int where possible
    attribute_df["Compno"] = pd.to_numeric(attribute_df["Compno"], errors="coerce").astype("Int64")

    # Build canonical node set from metadata
    all_node_ids: Set[int] = set(pd.to_numeric(node_df["ComputerNo"], errors="coerce").dropna().astype(int).unique())
    if not all_node_ids:
        raise ValueError("No node IDs found in node_df['ComputerNo'] after coercion.")

    # Optionally constrain canonical node set to max_nodes using most frequent nodes in attribute_df
    canonical_ids: List[int] = sorted(list(all_node_ids))
    if max_nodes is not None:
        if max_nodes <= 0:
            raise ValueError("max_nodes must be a positive integer.")
        if max_nodes < len(all_node_ids):
            freq_series = (
                attribute_df[attribute_df["Compno"].isin(all_node_ids)]["Compno"].value_counts()
            )
            top_ids = freq_series.index.tolist()[:max_nodes]
            if len(top_ids) < max_nodes:
                remaining = [nid for nid in canonical_ids if nid not in top_ids]
                top_ids.extend(remaining[: max_nodes - len(top_ids)])
            canonical_ids = sorted(top_ids)
        else:
            canonical_ids = sorted(list(all_node_ids))
    canonical_set: Set[int] = set(canonical_ids)

    # Determine prune threshold count
    if prune_percent is not None:
        if not (0 <= prune_percent <= 1):
            raise ValueError("prune_percent must be between 0 and 1.")
        base_nodes = max_nodes if max_nodes is not None else len(canonical_set)
        threshold_count = int(np.floor(base_nodes * prune_percent))
    else:
        threshold_count = max_missing

    # Build adjacency from edges for neighbor-based propagation (if available)
    adjacency: Dict[int, Set[int]] = {nid: set() for nid in canonical_set}
    if propagation == "neighbor_mean" and edges_path is not None and os.path.exists(edges_path):
        try:
            edges_df = pd.read_csv(edges_path)
            candidate_cols = [("source", "target"), ("src", "dst"), ("u", "v"), ("from", "to"),
                              ("Compno_src", "Compno_dst"), ("Compno_from", "Compno_to")]
            src_col = dst_col = None
            for s, d in candidate_cols:
                if s in edges_df.columns and d in edges_df.columns:
                    src_col, dst_col = s, d
                    break
            if src_col is None:
                cols = edges_df.columns.tolist()
                if len(cols) >= 2:
                    src_col, dst_col = cols[0], cols[1]
            if src_col and dst_col:
                src = pd.to_numeric(edges_df[src_col], errors="coerce").dropna().astype(int)
                dst = pd.to_numeric(edges_df[dst_col], errors="coerce").dropna().astype(int)
                for s_id, d_id in zip(src, dst):
                    if s_id in canonical_set and d_id in canonical_set:
                        adjacency[s_id].add(d_id)
                        adjacency[d_id].add(s_id)
        except Exception:
            adjacency = {nid: set() for nid in canonical_set}

    # Group by timestamp
    grouped = attribute_df.groupby("TimeStamp", sort=True)

    kept_frames: List[pd.DataFrame] = []
    pruned_rows: List[dict] = []

    # Identify numeric feature columns (exclude identifier)
    numeric_cols = attribute_df.select_dtypes(include=[np.number]).columns.tolist()
    if "Compno" in numeric_cols:
        numeric_cols.remove("Compno")

    for ts, ts_df in grouped:
        # Filter out rows with node IDs not in the canonical node set
        ts_df_filtered = ts_df[ts_df["Compno"].isin(canonical_set)].copy()

        present_nodes: Set[int] = set(pd.to_numeric(ts_df_filtered["Compno"], errors="coerce").dropna().astype(int).unique())
        missing_nodes: Set[int] = canonical_set - present_nodes

        missing_count = len(missing_nodes)
        # Comparator-driven prune decision
        do_prune = False
        if comparator == "gt":
            do_prune = missing_count > threshold_count
        elif comparator == "ge":
            do_prune = missing_count >= threshold_count
        elif comparator == "lt":
            do_prune = missing_count < threshold_count
        elif comparator == "le":
            do_prune = missing_count <= threshold_count
        else:
            raise ValueError("comparator must be one of: gt, ge, lt, le")

        if do_prune:
            pruned_rows.append({"TimeStamp": ts, "missing_count": missing_count})
            continue

        # Keep existing rows for this timestamp (filtered to canonical nodes)
        ts_df_clean = ts_df_filtered.copy()
        # Mark original rows as not imputed
        ts_df_clean["Imputed"] = False
        # Ensure one row per (TimeStamp, Compno)
        ts_df_clean = ts_df_clean.drop_duplicates(subset=["TimeStamp", "Compno"], keep="first")

        # Helper: neighbor-only propagation for a missing node (strict, no fallback)
        def propagate_means_for_node(node_id: int) -> pd.Series:
            if not numeric_cols:
                return pd.Series(dtype=float)
            neigh = list(adjacency.get(node_id, set()))
            present_neigh = [n for n in neigh if n in present_nodes]
            if present_neigh:
                return ts_df_clean[ts_df_clean["Compno"].isin(present_neigh)][numeric_cols].mean()
            # No neighbors with features -> leave as NaN for all numeric columns
            return pd.Series({col: np.nan for col in numeric_cols})

        # Create rows for each missing node with propagated numeric features
        if missing_nodes:
            imputed_count = len(missing_nodes)
            imputed_block = pd.DataFrame(index=range(imputed_count))
            imputed_block["TimeStamp"] = ts
            imputed_block["Compno"] = list(missing_nodes)
            imputed_block["Imputed"] = True

            # Compute propagated features per missing node
            propagated_means: Dict[int, pd.Series] = {}
            for node_id in missing_nodes:
                propagated_means[node_id] = propagate_means_for_node(node_id)
            for col in numeric_cols:
                imputed_block[col] = [propagated_means[nid].get(col, np.nan) for nid in imputed_block["Compno"]]

            # Preserve non-numeric columns as NaN in imputed rows
            non_numeric_cols = [c for c in attribute_df.columns if c not in (numeric_cols + ["TimeStamp", "Compno"])]
            for col in non_numeric_cols:
                imputed_block[col] = np.nan

            # Append imputed rows
            ts_df_clean = pd.concat([ts_df_clean, imputed_block], axis=0, ignore_index=True)

        # Final guard: enforce exactly one row per node per timestamp
        ts_df_clean = ts_df_clean.drop_duplicates(subset=["TimeStamp", "Compno"], keep="first")

        kept_frames.append(ts_df_clean)

    cleaned_df = pd.concat(kept_frames, axis=0, ignore_index=True) if kept_frames else attribute_df.iloc[0:0].copy()
    if "Imputed" not in cleaned_df.columns:
        cleaned_df["Imputed"] = False
    pruned_report = pd.DataFrame(pruned_rows).sort_values("TimeStamp") if pruned_rows else pd.DataFrame(columns=["TimeStamp", "missing_count"]) 

    return cleaned_df, pruned_report


def main():
    parser = argparse.ArgumentParser(description="Prune timestamps with missing nodes based on percentage or count, enforce canonical node set, and impute remaining gaps.")
    parser.add_argument("--base_path", type=str, default="../..", help="Base path to project root used by data loader.")
    parser.add_argument("--max_missing", type=int, default=2, help="Fallback: prune timestamps with more than this number of missing nodes (used if --prune_percent not provided).")
    parser.add_argument("--max_nodes", type=int, default=None, help="Target canonical node set size; if smaller than available, uses the most frequent nodes.")
    parser.add_argument("--prune_percent", type=float, default=None, help="Prune based on percentage of max nodes (e.g., 0.25 means threshold=floor(max_nodes*0.25)).")
    parser.add_argument("--comparator", type=str, choices=["gt", "ge", "lt", "le"], default="gt", help="Comparator for pruning decision between missing_count and threshold_count.")
    parser.add_argument("--propagation", type=str, choices=["neighbor_mean", "station_mean", "station_last", "global_mean"], default="neighbor_mean", help="Strategy to propagate features for missing nodes.")
    parser.add_argument("--edges_path", type=str, default=None, help="Path to edges CSV for neighbor-based propagation; default resolves to Dataset/TransientData/edges_df.csv.")
    parser.add_argument("--output_name", type=str, default="cleaned_df.csv", help="Filename for the cleaned dataset output.")
    parser.add_argument("--report_name", type=str, default="pruned_timestamps.csv", help="Filename for the pruned timestamps report.")
    args = parser.parse_args()

    print("--- Loading data for cleanup ---")
    attribute_df, node_df = load_and_preprocess_data(base_path=args.base_path)

    print(f"Total timestamps before cleanup: {attribute_df['TimeStamp'].nunique()}")

    # Resolve default edges path if not provided
    edges_path = args.edges_path or os.path.abspath(os.path.join(args.base_path, "Dataset", "TransientData", "edges_df.csv"))

    cleaned_df, pruned_report = prune_and_impute(
        attribute_df,
        node_df,
        max_missing=args.max_missing,
        max_nodes=args.max_nodes,
        prune_percent=args.prune_percent,
        comparator=args.comparator,
        propagation=args.propagation,
        edges_path=edges_path,
    )

    # Prepare output paths under Dataset/TransientData
    transient_dir = os.path.abspath(os.path.join(args.base_path, "Dataset", "TransientData"))
    os.makedirs(transient_dir, exist_ok=True)

    cleaned_path = os.path.join(transient_dir, args.output_name)
    report_path = os.path.join(transient_dir, args.report_name)

    cleaned_df.to_csv(cleaned_path, index=False)
    pruned_report.to_csv(report_path, index=False)

    total_pruned = len(pruned_report)
    total_imputed = int(cleaned_df["Imputed"].sum()) if "Imputed" in cleaned_df.columns else 0

    print("--- Cleanup Summary ---")
    canonical_size = cleaned_df["Compno"].nunique() if len(cleaned_df) else (args.max_nodes or len(set(pd.to_numeric(node_df["ComputerNo"], errors="coerce").dropna().astype(int).unique())))
    # threshold_count is defined in prune_and_impute; recompute here for logging
    base_nodes = args.max_nodes if args.max_nodes is not None else canonical_size
    threshold_count = int(np.floor(base_nodes * args.prune_percent)) if args.prune_percent is not None else args.max_missing
    print(f"Canonical node set size: {canonical_size}")
    print(f"Prune threshold count: {threshold_count}")
    print(f"Comparator: {args.comparator}")
    print(f"Pruned timestamps: {total_pruned}")
    if total_pruned:
        first_last = pruned_report["TimeStamp"].agg(["min", "max"]).to_dict()
        print(f"Pruned range: {first_last}")
    print(f"Cleaned rows: {len(cleaned_df)}")
    print(f"Report rows: {len(pruned_report)}")
    print(f"Imputed rows: {total_imputed}")
    print(f"Saved cleaned dataset to: {cleaned_path}")
    print(f"Saved pruned timestamp report to: {report_path}")


if __name__ == "__main__":
    main()
