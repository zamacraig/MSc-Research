import torch
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch_geometric_temporal.signal import DynamicGraphTemporalSignalBatch
import numpy as np


def create_temporal_dataset(final_df: pd.DataFrame) -> DynamicGraphTemporalSignalBatch:
    """
    Creates a temporal dataset from the final processed dataframe.
    """
    print("Creating temporal dataset...")

    # Label encode Compno columns to be 0-indexed
    le = LabelEncoder()
    final_df["Compno_x_encoded"] = le.fit_transform(final_df["Compno_x"])
    final_df["Compno_y_encoded"] = le.transform(final_df["Compno_y"])
    print("Label encoding complete.")

    # Feature Scaling
    scaler = MinMaxScaler()

    # Scale numerical features
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

    final_df[features_to_scale] = scaler.fit_transform(final_df[features_to_scale])
    print("Feature scaling complete.")

    # Discretize target variable
    final_df["Target_x"] = pd.qcut(final_df["Target_x"], q=5, labels=False, duplicates="drop")
    print("Target discretization complete.")

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

        # Node features
        node_features = (
            df_timestamp[
                [
                    "Compno_x",
                    "AverageMaximumTemperature_x",
                    "AverageMinimunTemperature_x",
                    "AverageMaximumRelativeHumidity_x",
                    "AverageMinimumRelativeHumidity_x",
                    "TotalRainfall_x",
                    "AverageTotalRadiation_x",
                    "AverageWindSpeed_x",
                    "AverageTotalRelativeEvaporation_x",
                ]
            ]
            .drop_duplicates()
            .sort_values(by="Compno_x")
        )
        x = torch.tensor(
            node_features[
                [
                    "AverageMaximumTemperature_x",
                    "AverageMinimunTemperature_x",
                    "AverageMaximumRelativeHumidity_x",
                    "AverageMinimumRelativeHumidity_x",
                    "TotalRainfall_x",
                    "AverageTotalRadiation_x",
                    "AverageWindSpeed_x",
                    "AverageTotalRelativeEvaporation_x",
                ]
            ].values,
            dtype=torch.float,
        )

        # Edge indices
        edge_indices = torch.tensor(df_timestamp[["Compno_x", "Compno_y"]].values.T, dtype=torch.long)

        # Edge weights
        edge_weights = torch.tensor(df_timestamp["distance_km"].values, dtype=torch.float)

        # Targets
        targets = df_timestamp[["Compno_x", "Target_x"]].drop_duplicates()["Target_x"].values
        targets_np = np.array(targets, dtype=np.float64)
        targets_tensor = torch.tensor(targets_np, dtype=torch.float)

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
    Split snapshots into train/test ensuring the test set covers all classes.

    - Greedy selection adds snapshots that cover uncovered classes first.
    - Fills remaining test slots from leftover snapshots.

    Returns (train_dataset, test_dataset).
    """
    # Snapshot-level labels
    snapshot_labels = []
    for i in range(dataset.snapshot_count):
        snap = dataset[i]
        lbls = set(np.unique(snap.y.cpu().numpy()))
        snapshot_labels.append(lbls)

    # Infer number of classes if not provided
    if num_classes is None:
        all_classes = sorted(set().union(*snapshot_labels))
        num_classes = int(max(all_classes) + 1) if all_classes else 1

    total = dataset.snapshot_count
    test_count = max(1, int(round(total * (1.0 - float(train_ratio)))) )

    # Greedy cover of classes
    uncovered = set(range(num_classes))
    candidates = list(range(total))
    test_idxs = []
    while uncovered and len(test_idxs) < test_count and candidates:
        # pick snapshot that covers most uncovered classes
        best_i = None
        best_gain = -1
        for i in candidates:
            gain = len(snapshot_labels[i] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i is None:
            break
        test_idxs.append(best_i)
        uncovered -= (snapshot_labels[best_i] & uncovered)
        candidates.remove(best_i)

    # Fill remaining test slots
    for i in candidates:
        if len(test_idxs) >= test_count:
            break
        test_idxs.append(i)

    test_set = set(test_idxs)
    train_idxs = [i for i in range(total) if i not in test_set]

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

    return subset(train_idxs), subset(test_idxs)
