import torch
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch_geometric_temporal.signal import DynamicGraphTemporalSignalBatch
import numpy as np


def _apply_seasonal_normalization(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Normalize features relative to their monthly (seasonal) statistics.
    
    This converts raw values to z-scores within each month, so that:
    - A hot summer day (35°C in July) → z ≈ 0 (normal for July)
    - A cold winter day (10°C in January) → z ≈ 0 (normal for January)
    
    This removes the seasonal component from the features, leaving only
    the residual variation that's meaningful for prediction.
    """
    df = df.copy()
    
    # Extract month from timestamp
    df["_month"] = pd.to_datetime(df["TimeStamp"]).dt.month
    
    # Compute per-month mean and std for each feature
    monthly_stats = df.groupby("_month")[features].agg(["mean", "std"])
    
    # Z-score normalize each feature within its month
    for feat in features:
        month_mean = df["_month"].map(monthly_stats[(feat, "mean")])
        month_std = df["_month"].map(monthly_stats[(feat, "std")]).replace(0, 1)
        df[feat] = (df[feat] - month_mean) / month_std
    
    # Drop temporary column
    df = df.drop(columns=["_month"])
    
    return df


def _add_temporal_lag_features(df: pd.DataFrame, lag_features: list[str], lags: list[int] = [1], rolling_windows: list[int] = [7]) -> pd.DataFrame:
    """
    Add temporal lag and rolling mean features for each node.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with TimeStamp and Compno_x columns
    lag_features : list[str]
        List of feature columns to create lags for
    lags : list[int]
        List of lag periods (default [1] = previous day)
    rolling_windows : list[int]
        List of rolling window sizes for mean (default [7] = 7-day average)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added lag and rolling features
    """
    df = df.copy()
    df = df.sort_values(["Compno_x", "TimeStamp"])
    
    new_features = []
    
    for feat in lag_features:
        # Lag features (previous day values)
        for lag in lags:
            lag_col = f"{feat}_lag{lag}"
            df[lag_col] = df.groupby("Compno_x")[feat].shift(lag)
            new_features.append(lag_col)
        
        # Rolling mean features (e.g., 7-day average)
        for window in rolling_windows:
            roll_col = f"{feat}_roll{window}"
            df[roll_col] = df.groupby("Compno_x")[feat].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            new_features.append(roll_col)
        
        # Difference from previous day (captures trend/change)
        diff_col = f"{feat}_diff"
        df[diff_col] = df.groupby("Compno_x")[feat].diff()
        new_features.append(diff_col)
    
    # Fill NaN values created by lag/diff with 0 (first timestamps)
    for col in new_features:
        df[col] = df[col].fillna(0)
    
    print(f"Added {len(new_features)} temporal lag features:")
    for feat in new_features[:5]:
        print(f"  - {feat}")
    if len(new_features) > 5:
        print(f"  ... and {len(new_features) - 5} more")
    
    return df


def create_temporal_dataset(
    final_df: pd.DataFrame,
    seasonal_normalization: bool = True,
    add_lag_features: bool = True,
    n_classes: int = 5,
) -> DynamicGraphTemporalSignalBatch:
    """
    Creates a temporal dataset from the final processed dataframe.
    
    Parameters
    ----------
    final_df : pd.DataFrame
        The preprocessed dataframe with node features and edges.
    seasonal_normalization : bool, default True
        If True, normalize features relative to their monthly statistics
        (removes seasonal drift). If False, use global MinMaxScaler.
    add_lag_features : bool, default True
        If True, add lag-1, 7-day rolling mean, and diff features for
        key weather variables to capture temporal autocorrelation.
    n_classes : int, default 5
        Number of classes for target discretization.
    """
    print("Creating temporal dataset...")

    # Add temporal lag features BEFORE normalization
    if add_lag_features:
        # Key features that benefit from lag information
        lag_feature_cols = [
            "AverageMaximumTemperature_x",
            "AverageTotalRadiation_x",  # GHI is highly autocorrelated
            "TotalRainfall_x",
        ]
        # Only add lags for features that exist
        lag_feature_cols = [f for f in lag_feature_cols if f in final_df.columns]
        if lag_feature_cols:
            final_df = _add_temporal_lag_features(final_df, lag_feature_cols)

    # Label encode Compno columns to be 0-indexed
    le = LabelEncoder()
    final_df["Compno_x_encoded"] = le.fit_transform(final_df["Compno_x"])
    final_df["Compno_y_encoded"] = le.transform(final_df["Compno_y"])
    print("Label encoding complete.")

    # Identify features to normalize
    features_to_scale = [
        col
        for col in final_df.columns
        if "_x" in col and "Compno" not in col and "Target" not in col and "TimeStamp" not in col
    ]
    print("\n" + "=" * 50)
    print(" " * 18 + "DATASET FEATURES")
    print("=" * 50)
    print("Features being scaled:")
    for feature in features_to_scale:
        print(f"- {feature}")
    print("=" * 50)

    # Apply normalization based on configuration
    if seasonal_normalization:
        print("Applying SEASONAL normalization (per-month z-scores)...")
        print("  This removes seasonal drift by normalizing relative to monthly baselines.")
        final_df = _apply_seasonal_normalization(final_df, features_to_scale)
        # After seasonal z-scoring, rescale to 0-1 for consistency
        scaler = MinMaxScaler()
        final_df[features_to_scale] = scaler.fit_transform(final_df[features_to_scale])
        print("Seasonal normalization + MinMax rescaling complete.")
    else:
        print("Applying GLOBAL normalization (MinMaxScaler across all timestamps)...")
        print("  Warning: Seasonal drift patterns will be preserved in the features.")
        scaler = MinMaxScaler()
        final_df[features_to_scale] = scaler.fit_transform(final_df[features_to_scale])
        print("Feature scaling complete.")

    # Recompute Target_x as the mean of the NOW-normalized feature columns.
    # This ensures the target reflects a fair composite, regardless of what was
    # stored in the source CSV (which may have used raw unnormalized values).
    final_df["Target_x"] = final_df[features_to_scale].mean(axis=1)
    print("Target_x recomputed from normalized features.")

    # Discretize target variable into n_classes classes.
    # pd.qcut is tried first (equal-frequency); if duplicate bin edges reduce the
    # number of unique classes below n_classes, pd.cut (equal-width) is used as a fallback
    # so that exactly n_classes non-empty classes are always produced.
    print(f"Discretizing into {n_classes} classes...")
    raw_bins = pd.qcut(final_df["Target_x"], q=n_classes, labels=False, duplicates="drop")
    actual_classes = raw_bins.nunique()
    if actual_classes < n_classes:
        print(
            f"  WARNING: pd.qcut produced only {actual_classes} bins (duplicate bin edges). "
            f"Falling back to equal-width pd.cut to guarantee {n_classes} classes."
        )
        raw_bins = pd.cut(final_df["Target_x"], bins=n_classes, labels=False)

    # Relabel to consecutive integers 0..k-1 (guards against any gaps left by either method)
    label_map = {old: new for new, old in enumerate(sorted(raw_bins.dropna().unique()))}
    final_df["Target_x"] = raw_bins.map(label_map)

    # Report the resulting distribution
    dist = final_df["Target_x"].value_counts().sort_index()
    total = dist.sum()
    print("Target discretization complete. Distribution:")
    for cls, cnt in dist.items():
        print(f"  Class {int(cls)}: {cnt} samples ({100.0 * cnt / total:.1f}%)")

    # Timestamps
    timestamps = final_df["TimeStamp"].unique()

    # Initialize lists to store data for each timestamp
    edge_indices_list = []
    edge_weights_list = []
    node_features_list = []
    targets_list = []
    all_batches = []
    timestamps_list = []

    print("\n" + "=" * 50)
    print(" " * 18 + "DATASET COUNTS")
    print("=" * 50)
    print(f"Number of unique timestamps: {len(timestamps)}")
    print(f"Total number of nodes: {len(final_df['Compno_x_encoded'].unique())}")
    print(f"Total number of edges (time-stamped): {len(final_df)}")
    print("=" * 50)

    for timestamp in timestamps:
        df_timestamp = final_df[final_df["TimeStamp"] == timestamp]

        # Convert timestamp string to numerical (Unix timestamp)
        if isinstance(timestamp, str):
            try:
                dt_object = datetime.strptime(timestamp, "%Y-%m-%d")
                timestamp_numeric = dt_object.timestamp()
            except ValueError:
                raise ValueError(f"Invalid timestamp format: {timestamp}")
        else:
            timestamp_numeric = float(timestamp)

        # Dynamically get all feature columns (includes base + lag features)
        # Base features: temperature, humidity, rainfall, radiation, wind, evaporation
        # Lag features: *_lag1, *_roll7, *_diff (if add_lag_features=True)
        base_feature_cols = [
            "AverageMaximumTemperature_x",
            "AverageMinimunTemperature_x",
            "AverageMaximumRelativeHumidity_x",
            "AverageMinimumRelativeHumidity_x",
            "TotalRainfall_x",
            "AverageTotalRadiation_x",
            "AverageWindSpeed_x",
            "AverageTotalRelativeEvaporation_x",
        ]
        
        # Find any lag/rolling/diff features that exist in the dataframe
        lag_feature_cols = [
            col for col in df_timestamp.columns
            if "_lag" in col or "_roll" in col or "_diff" in col
        ]
        
        all_feature_cols = base_feature_cols + lag_feature_cols

        # Node features - deduplicate by Compno_x ONLY, then select features
        # This ensures we get one row per node even when lag features differ slightly across edges
        node_features = (
            df_timestamp[["Compno_x"] + all_feature_cols]
            .drop_duplicates(subset=["Compno_x"])
            .sort_values(by="Compno_x")
        )
        x = torch.tensor(
            node_features[all_feature_cols].values,
            dtype=torch.float,
        )

        # Cyclical month encoding (sin + cos) — gives the model explicit seasonal context
        # to address the high Feature Drift score.  Using sin/cos avoids the ordinal
        # discontinuity between December (month 12) and January (month 1).
        #   January  →  sin=0.00, cos=1.00
        #   April    →  sin=0.87, cos=0.50
        #   July     →  sin=0.87, cos=-0.50  (opposite to January — hottest month)
        #   October  →  sin=-0.87, cos=-0.50
        ts_dt = pd.Timestamp(timestamp) if not isinstance(timestamp, pd.Timestamp) else timestamp
        month_rad = 2.0 * np.pi * (ts_dt.month - 1) / 12.0
        month_enc = torch.tensor(
            [[float(np.sin(month_rad)), float(np.cos(month_rad))]] * x.size(0),
            dtype=torch.float,
        )
        x = torch.cat([x, month_enc], dim=1)   # [num_nodes, base_features + lag_features + 2]

        # Edge indices
        edge_indices = torch.tensor(df_timestamp[["Compno_x", "Compno_y"]].values.T, dtype=torch.long)

        # Edge weights: INVERSE distance so closer stations have HIGHER influence
        # Using exp(-distance) for smooth decay; epsilon prevents division by zero
        distances = df_timestamp["distance_km"].values
        edge_weights = torch.tensor(np.exp(-distances), dtype=torch.float)

        # Targets - must match node_features extraction (one per node)
        targets_df = (
            df_timestamp[["Compno_x", "Target_x"]]
            .drop_duplicates(subset=["Compno_x"])
            .sort_values(by="Compno_x")
        )
        targets_np = np.array(targets_df["Target_x"].values, dtype=np.float64)
        targets_tensor = torch.tensor(targets_np, dtype=torch.float)
        
        # Sanity check: node features and targets must have same count
        assert x.size(0) == targets_tensor.size(0), (
            f"Node/target mismatch at timestamp {timestamp}: "
            f"{x.size(0)} nodes vs {targets_tensor.size(0)} targets"
        )

        # Generate batches (assuming each timestamp is a separate batch)
        batch = torch.tensor([i for i in range(len(df_timestamp))], dtype=torch.long)

        snapshot_timestamp = torch.FloatTensor([timestamp_numeric])

        # Append to lists
        node_features_list.append(x)
        edge_indices_list.append(edge_indices)
        edge_weights_list.append(edge_weights)
        targets_list.append(targets_tensor)
        all_batches.append(batch)
        timestamps_list.append(snapshot_timestamp)

    # Ensure the length consistency across all lists
    assert (
        len(edge_indices_list)
        == len(edge_weights_list)
        == len(node_features_list)
        == len(targets_list)
        == len(all_batches)
        == len(timestamps_list)
    ), "Temporal dimension inconsistency."

    print("Temporal dataset creation complete.")

    return DynamicGraphTemporalSignalBatch(
        edge_indices=edge_indices_list,
        edge_weights=edge_weights_list,
        features=node_features_list,
        targets=targets_list,
        batches=all_batches,
        timestamps=timestamps_list,
    )


def stratified_temporal_signal_split(dataset: DynamicGraphTemporalSignalBatch, train_ratio: float = 0.8, num_classes: int | None = None):
    """
    Split snapshots into train/val while preserving class proportions.

    Uses interleaved sampling within class-based stratification to maintain both
    temporal patterns and class balance across splits.

    Algorithm:
    1. Compute signature vector for each snapshot (class counts as proportion)
    2. Use k-means to cluster snapshots by class signature similarity
    3. Within each cluster, apply interleaved sampling to select val snapshots
    4. This ensures val set mirrors train set class distribution

    Returns (train_dataset, val_dataset).
    """
    total = dataset.snapshot_count
    val_count = max(1, int(round(total * (1.0 - float(train_ratio)))))

    # Compute per-snapshot class counts
    snapshot_class_counts = []
    for i in range(total):
        snap = dataset[i]
        lbls = snap.y.cpu().numpy().astype(int)
        counts = np.bincount(lbls, minlength=num_classes if num_classes else 0)
        snapshot_class_counts.append(counts)

    # Infer number of classes if not provided
    if num_classes is None:
        num_classes = max(len(c) for c in snapshot_class_counts) if snapshot_class_counts else 1

    # Pad counts to num_classes if needed
    snapshot_class_counts = [
        np.pad(c, (0, max(0, num_classes - len(c)))) for c in snapshot_class_counts
    ]

    # Convert to proportions (signature vectors)
    signatures = []
    for counts in snapshot_class_counts:
        total_nodes = counts.sum()
        if total_nodes > 0:
            signatures.append(counts / total_nodes)
        else:
            signatures.append(counts.astype(float))
    signatures = np.array(signatures)  # (num_snapshots, num_classes)

    # Compute overall target class distribution from all snapshots
    all_counts = np.sum(snapshot_class_counts, axis=0)
    target_dist = all_counts / all_counts.sum() if all_counts.sum() > 0 else np.ones(num_classes) / num_classes

    # Strategy: select val snapshots that together approximate the target distribution
    # while maintaining temporal spread
    
    # Step 1: Interleaved base selection (ensures temporal coverage)
    step = total / val_count
    base_val_idxs = sorted({min(int(i * step), total - 1) for i in range(val_count)})

    # Step 2: Compute class distribution of base selection
    def compute_dist(idxs):
        counts = np.sum([snapshot_class_counts[i] for i in idxs], axis=0)
        return counts / counts.sum() if counts.sum() > 0 else np.ones(num_classes) / num_classes

    # Step 3: Iteratively swap to improve class balance (greedy refinement)
    val_idxs = list(base_val_idxs)
    remaining = [i for i in range(total) if i not in set(val_idxs)]
    
    # JS divergence helper
    def js_divergence(p, q):
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)
        m = 0.5 * (p + q)
        return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

    max_swaps = min(50, total)  # limit iterations
    for _ in range(max_swaps):
        current_dist = compute_dist(val_idxs)
        current_js = js_divergence(current_dist, target_dist)
        
        # Try swapping each val snapshot with each remaining snapshot
        best_swap = None
        best_js = current_js
        
        for vi, val_i in enumerate(val_idxs):
            for rem_i in remaining:
                # Candidate swap
                candidate = val_idxs.copy()
                candidate[vi] = rem_i
                cand_dist = compute_dist(candidate)
                cand_js = js_divergence(cand_dist, target_dist)
                
                if cand_js < best_js - 1e-6:  # must improve significantly
                    best_js = cand_js
                    best_swap = (vi, val_i, rem_i)
        
        if best_swap is None:
            break  # no improvement possible
        
        vi, val_i, rem_i = best_swap
        val_idxs[vi] = rem_i
        remaining.remove(rem_i)
        remaining.append(val_i)

    val_set = set(val_idxs)
    train_idxs = [i for i in range(total) if i not in val_set]
    val_idxs = sorted(val_idxs)  # maintain temporal order

    # Construct new datasets by indexing
    def subset(idxs):
        return DynamicGraphTemporalSignalBatch(
            edge_indices=[dataset.edge_indices[i] for i in idxs],
            edge_weights=[dataset.edge_weights[i] for i in idxs],
            features=[dataset.features[i] for i in idxs],
            targets=[dataset.targets[i] for i in idxs],
            batches=[dataset.batches[i] for i in idxs],
            timestamps=[dataset.timestamps[i] for i in idxs],
        )

    return subset(train_idxs), subset(val_idxs)


def interleaved_temporal_split(
    dataset: DynamicGraphTemporalSignalBatch, train_ratio: float = 0.8
):
    """
    Systematic (interleaved) outer split that preserves the full temporal class distribution.

    Every ~(1/test_ratio)-th snapshot is assigned to the test set so that every season
    is proportionally represented in both partitions.  Temporal ordering within each
    partition is preserved.

    This prevents the seasonal label drift that occurs when the last N% of chronological
    snapshots are used as the test set.
    """
    total = dataset.snapshot_count
    test_count = max(1, int(round(total * (1.0 - train_ratio))))
    step = total / test_count  # float step so positions spread evenly
    test_idxs: list[int] = sorted({min(int(i * step), total - 1) for i in range(test_count)})
    # If rounding / set-deduplication left us short, fill from the middle of the series
    if len(test_idxs) < test_count:
        remaining = [i for i in range(total) if i not in set(test_idxs)]
        mid = len(remaining) // 2
        fill = remaining[mid: mid + (test_count - len(test_idxs))]
        test_idxs = sorted(test_idxs + fill)

    test_set   = set(test_idxs)
    train_idxs = [i for i in range(total) if i not in test_set]

    def _subset(idxs):
        return DynamicGraphTemporalSignalBatch(
            edge_indices=[dataset.edge_indices[i] for i in idxs],
            edge_weights=[dataset.edge_weights[i] for i in idxs],
            features=[dataset.features[i] for i in idxs],
            targets=[dataset.targets[i] for i in idxs],
            batches=[dataset.batches[i] for i in idxs],
            timestamps=[dataset.timestamps[i] for i in idxs],
        )

    return _subset(train_idxs), _subset(test_idxs)


def walk_forward_cv_split(dataset: DynamicGraphTemporalSignalBatch, n_folds: int = 5):
    """
    Expanding-window walk-forward cross-validation generator.

    Divides the T snapshots into (n_folds + 1) equal segments of size
    seg = T // (n_folds + 1).  For fold k (1-indexed):

        Training window:   snapshots [0,  (k) * seg)
        Validation window: snapshots [(k) * seg,  min((k+1) * seg, T)]

    The last fold's validation window runs to the end of the dataset.

    Yields (train_dataset, val_dataset, fold_number) for each valid fold.
    """
    total = dataset.snapshot_count
    seg   = max(1, total // (n_folds + 1))

    def _subset(idxs):
        return DynamicGraphTemporalSignalBatch(
            edge_indices=[dataset.edge_indices[i] for i in idxs],
            edge_weights=[dataset.edge_weights[i] for i in idxs],
            features=[dataset.features[i] for i in idxs],
            targets=[dataset.targets[i] for i in idxs],
            batches=[dataset.batches[i] for i in idxs],
            timestamps=[dataset.timestamps[i] for i in idxs],
        )

    for k in range(n_folds):
        train_end = seg * (k + 1)
        val_start = train_end
        val_end   = min(val_start + seg, total) if k < n_folds - 1 else total

        if train_end >= total or val_start >= total or val_end <= val_start:
            print(f"  Walk-forward CV: fold {k + 1} skipped (insufficient snapshots).")
            continue

        yield _subset(list(range(train_end))), _subset(list(range(val_start, val_end))), k + 1
