import pandas as pd
import numpy as np


def haversine(lat1, lon1, lat2, lon2, alt1, alt2):
    """
    Calculate the Haversine distance between two points on Earth, including altitude.
    """
    R = 6371  # Radius of the Earth in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    horizontal_distance = R * c

    # Altitudes are in meters, convert to km for consistent units
    vertical_distance = (alt2 - alt1) / 1000.0

    distance = np.sqrt(horizontal_distance ** 2 + vertical_distance ** 2)
    return distance


def build_static_edges(node_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the complete (self-loop-free) station-pair edge list with Haversine
    distances, independent of any timestamp. Used both as the first stage of
    `create_graph_data` and as the neighbor source for feature propagation in
    `masters_research.data.cleaning.prune_and_impute`.

    Returns a DataFrame with columns: CompNo1, CompNo2, distance_km.
    """
    df1 = node_df[["ComputerNo", "Latitude", "Longitude", "Altitude"]].copy()
    df1["key"] = 1

    # Cross join to get all pairs of nodes
    edges_df = pd.merge(df1, df1, on="key").drop("key", axis=1)
    edges_df.rename(
        columns={
            "ComputerNo_x": "CompNo1",
            "Latitude_x": "Lat1",
            "Longitude_x": "Long1",
            "Altitude_x": "Alt1",
            "ComputerNo_y": "CompNo2",
            "Latitude_y": "Lat2",
            "Longitude_y": "Long2",
            "Altitude_y": "Alt2",
        },
        inplace=True,
    )

    edges_df["distance_km"] = edges_df.apply(
        lambda row: haversine(row["Lat1"], row["Long1"], row["Lat2"], row["Long2"], row["Alt1"], row["Alt2"]),
        axis=1,
    )

    # Filter out self-loops (distance = 0)
    return edges_df[edges_df["distance_km"] != 0][["CompNo1", "CompNo2", "distance_km"]]


def create_graph_data(attribute_df: pd.DataFrame, node_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the graph data from preprocessed dataframes.

    Args:
        attribute_df (pd.DataFrame): The preprocessed attribute data.
        node_df (pd.DataFrame): The preprocessed node data.

    Returns:
        pd.DataFrame: The final DataFrame with graph data for each timestamp.
    """
    print("\n--- Starting Graph Construction ---")
    print("[1/6] Creating all possible edges between nodes...")
    print("[2/6] Calculating Haversine distance for each edge...")
    print("[3/6] Filtering out self-loops...")
    edges_df = build_static_edges(node_df)

    # Sort attributes by timestamp
    attribute_df = attribute_df.sort_values(by="TimeStamp")

    # Prepare attributes for merging
    attr_x = attribute_df.add_suffix("_x")
    attr_y = attribute_df.add_suffix("_y")

    # Create all node pairs for each timestamp
    print("[4/6] Creating time-stamped edges...")
    time_stamped_edges = (
        attribute_df.groupby("TimeStamp")[["Compno"]].apply(lambda x: pd.merge(x.assign(key=1), x.assign(key=1), on="key"))
        .drop("key", axis=1)
        .reset_index()
    )

    # Merge with attributes and edge distances
    print("[5/6] Merging attributes and edge distances...")
    final_df = (
        time_stamped_edges.merge(attr_x, left_on=["TimeStamp", "Compno_x"], right_on=["TimeStamp_x", "Compno_x"])
        .merge(attr_y, left_on=["TimeStamp", "Compno_y"], right_on=["TimeStamp_y", "Compno_y"])
        .merge(edges_df, left_on=["Compno_x", "Compno_y"], right_on=["CompNo1", "CompNo2"])
    )

    # Clean up final dataframe
    print("[6/6] Cleaning up final dataframe...")
    final_df = final_df.drop(
        columns=[
            "TimeStamp_x",
            "TimeStamp_y",
            "CompNo1",
            "CompNo2",
            "level_1",
            "Year_x",
            "Month_x",
            "Year_y",
            "Month_y",
        ]
    )
    final_df = final_df.sort_values(by=["TimeStamp", "Compno_x", "Compno_y"]).reset_index(drop=True)
    final_df.dropna(inplace=True)

    print("--- Graph Construction Finished ---\n")
    return final_df
