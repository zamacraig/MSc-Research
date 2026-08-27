"""
Builds the canonical modelling dataset (processedData.csv) from the raw
station CSVs: load -> prune timestamps / propagate missing-node features ->
construct the graph edge list.

This is the up-to-date replacement for the ad hoc, hand-run steps that used
to live in Notebooks/GNNExperiments/gann_pipeline/ (load_and_preprocess_data
-> clean_dataset.prune_and_impute -> graph_constructor.create_graph_data).
train.py and the reporting scripts (eda_raw_data.py, trace_data_pipeline.py,
visualize_feature_drift.py, learning_curve.py, mcnemar_report.py) all read
the output of this script from Dataset/TransientData/processedData.csv.

Run from project root:
    python src/masters_research/pipeline/build_dataset.py
"""

import argparse
from pathlib import Path

from masters_research.data.cleaning import prune_and_impute
from masters_research.data.graph import build_static_edges, create_graph_data
from masters_research.data.loader import load_and_preprocess_data


def build_dataset(
    base_path: Path,
    skip_pruning: bool = False,
    max_missing: int = 2,
    max_nodes: int | None = None,
    prune_percent: float | None = None,
    comparator: str = "gt",
    propagation: str = "neighbor_mean",
):
    attribute_df, node_df = load_and_preprocess_data(base_path=str(base_path))

    imputed_report = None
    if skip_pruning:
        cleaned_df = attribute_df
        pruned_report = None
    else:
        edges_df = build_static_edges(node_df)
        cleaned_df, pruned_report = prune_and_impute(
            attribute_df,
            node_df,
            max_missing=max_missing,
            max_nodes=max_nodes,
            prune_percent=prune_percent,
            comparator=comparator,
            propagation=propagation,
            edges_df=edges_df,
        )
        n_pruned = len(pruned_report)
        n_imputed = int(cleaned_df["Imputed"].sum()) if "Imputed" in cleaned_df.columns else 0
        print(f"Pruned {n_pruned} timestamp(s); imputed {n_imputed} missing-node row(s).")

        # create_temporal_dataset() treats every "_x"-suffixed column as a feature to
        # scale and average into the target (see dataset.py's features_to_scale). The
        # boolean Imputed flag must not survive graph construction as Imputed_x/_y or
        # it gets folded into MinMax scaling and the composite quality target itself.
        # Keep the flag as its own report instead of carrying it into processedData.csv.
        imputed_report = cleaned_df.loc[cleaned_df["Imputed"], ["TimeStamp", "Compno"]].reset_index(drop=True)
        cleaned_df = cleaned_df.drop(columns=["Imputed"])

    final_df = create_graph_data(cleaned_df, node_df)
    return final_df, pruned_report, imputed_report


def main():
    parser = argparse.ArgumentParser(description="Build processedData.csv from the raw station CSVs.")
    parser.add_argument("--base_path", type=str, default=None, help="Base path to project root (contains Dataset/). Defaults to the repo root.")
    parser.add_argument("--skip_pruning", action="store_true", help="Skip timestamp pruning / feature propagation and build the graph directly from the raw attribute data.")
    parser.add_argument("--max_missing", type=int, default=2, help="Fallback: prune timestamps with more than this number of missing nodes (used if --prune_percent not provided).")
    parser.add_argument("--max_nodes", type=int, default=None, help="Target canonical node set size; if smaller than available, uses the most frequent nodes.")
    parser.add_argument("--prune_percent", type=float, default=None, help="Prune based on percentage of max nodes (e.g., 0.25 means threshold=floor(max_nodes*0.25)).")
    parser.add_argument("--comparator", type=str, choices=["gt", "ge", "lt", "le"], default="gt", help="Comparator for pruning decision between missing_count and threshold_count.")
    parser.add_argument("--propagation", type=str, choices=["neighbor_mean", "station_mean", "station_last", "global_mean"], default="neighbor_mean", help="Strategy to propagate features for missing nodes.")
    parser.add_argument("--output_name", type=str, default="processedData.csv", help="Filename for the built dataset, written under Dataset/TransientData/.")
    parser.add_argument("--report_name", type=str, default="pruned_timestamps.csv", help="Filename for the pruned-timestamps report, written under Dataset/TransientData/.")
    parser.add_argument("--imputed_report_name", type=str, default="imputed_rows.csv", help="Filename for the imputed-rows report, written under Dataset/TransientData/.")
    args = parser.parse_args()

    base_path = Path(args.base_path) if args.base_path else Path(__file__).resolve().parents[3]

    final_df, pruned_report, imputed_report = build_dataset(
        base_path,
        skip_pruning=args.skip_pruning,
        max_missing=args.max_missing,
        max_nodes=args.max_nodes,
        prune_percent=args.prune_percent,
        comparator=args.comparator,
        propagation=args.propagation,
    )

    transient_dir = base_path / "Dataset" / "TransientData"
    transient_dir.mkdir(parents=True, exist_ok=True)

    output_path = transient_dir / args.output_name
    final_df.to_csv(output_path, index=False)
    print(f"Saved built dataset to: {output_path} ({len(final_df)} rows)")

    if pruned_report is not None:
        report_path = transient_dir / args.report_name
        pruned_report.to_csv(report_path, index=False)
        print(f"Saved pruned timestamp report to: {report_path}")

    if imputed_report is not None:
        imputed_report_path = transient_dir / args.imputed_report_name
        imputed_report.to_csv(imputed_report_path, index=False)
        print(f"Saved imputed rows report to: {imputed_report_path}")


if __name__ == "__main__":
    main()
