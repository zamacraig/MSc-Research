import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_data(base_path: str = "."):
    """
    Loads and preprocesses the weather station data.

    Args:
        base_path (str): The base path to the project root (contains Dataset/).

    Returns:
        tuple: A tuple containing the preprocessed attribute_df and node_df.
    """
    print("\n--- Starting Data Loading and Preprocessing ---")

    base_path = str(Path(base_path).resolve())
    dataset_path = os.path.join(base_path, "Dataset", "OriginalDataset")

    # Load station metadata
    print("[1/8] Loading station metadata...")
    stations_path = os.path.join(dataset_path, "StationsData.csv")
    node_df = pd.read_csv(
        stations_path,
        sep=";",
        skiprows=[0],
        header=None,
        names=["ComputerNo", "StationName", "Latitude", "Longitude", "Altitude"],
    )

    # Load weather data for all stations
    print("[2/8] Loading weather data for all stations...")
    data_files = [
        "St_BrandvleiB.csv",
        "St_Kakamas.csv",
        "St_Luckhof.csv",
        "St_Olifantshoek.csv",
        "St_Prieska.csv",
        "St_Vaalharts.csv",
        "St_VanWyksvleiB.csv",
        "St_BerkleyWest.csv",
    ]

    frames = []
    for file in data_files:
        file_path = os.path.join(dataset_path, file)
        df = pd.read_csv(
            file_path,
            sep=";",
            skiprows=[0],
            header=None,
            names=[
                "Compno",
                "Year",
                "Month",
                "AverageMaximumTemperature",
                "AverageMinimunTemperature",
                "AverageMaximumRelativeHumidity",
                "AverageMinimumRelativeHumidity",
                "TotalRainfall",
                "AverageTotalRadiation",
                "AverageWindSpeed",
                "AverageTotalRelativeEvaporation",
            ],
        )
        frames.append(df)

    attribute_df = pd.concat(frames)

    # Data Cleaning
    print("[3/8] Cleaning data (replacing commas and tildes)...")
    attribute_df = attribute_df.replace(",", ".", regex=True)
    attribute_df = attribute_df.replace("~", "0", regex=True)
    node_df = node_df.replace(",", ".", regex=True)
    node_df = node_df.replace("~", "0", regex=True)

    # Convert to numeric
    print("[4/8] Converting data to numeric types...")
    attribute_df = attribute_df.apply(pd.to_numeric, errors="coerce")
    node_df[["ComputerNo", "Latitude", "Longitude", "Altitude"]] = node_df[
        ["ComputerNo", "Latitude", "Longitude", "Altitude"]
    ].apply(pd.to_numeric, errors="coerce")

    # One metadata row per station, and only stations that have an observation
    # file. StationsData.csv lists identifiers (e.g. the superseded Brandvlei A /
    # Van Wyksvlei A records) and repeats at least one station; keeping them would
    # add nodes that never report (filled entirely by imputation) and duplicate
    # every edge touching the repeated station.
    node_df = node_df.drop_duplicates(subset="ComputerNo", keep="first")
    with_data = attribute_df["Compno"].dropna().astype(int).unique()
    n_meta = len(node_df)
    node_df = node_df[node_df["ComputerNo"].isin(with_data)].reset_index(drop=True)
    print(f"      Station metadata: {n_meta} unique IDs, {len(node_df)} with observation files kept.")

    # Normalize feature columns before computing the target so that every feature
    # contributes equally to the composite mean (not dominated by high-magnitude
    # columns such as Humidity or Rainfall).
    print("[5/8] Normalizing feature columns (MinMaxScaler) and calculating target column...")
    feature_cols = [
        "AverageMaximumTemperature",
        "AverageMinimunTemperature",
        "AverageMaximumRelativeHumidity",
        "AverageMinimumRelativeHumidity",
        "TotalRainfall",
        "AverageTotalRadiation",
        "AverageWindSpeed",
        "AverageTotalRelativeEvaporation",
    ]
    scaler = MinMaxScaler()
    attribute_df[feature_cols] = scaler.fit_transform(attribute_df[feature_cols])
    # Target is the mean of the 8 normalized features — each contributes equally.
    attribute_df["Target"] = attribute_df[feature_cols].mean(axis=1)

    # Fix null values
    print("[6/8] Handling null values (forward-fill for TotalRainfall)...")
    attribute_df["TotalRainfall"] = attribute_df["TotalRainfall"].fillna(method="ffill")

    # Create TimeStamp
    print("[7/8] Creating TimeStamp...")
    attribute_df["TimeStamp"] = pd.to_datetime(attribute_df[["Year", "Month"]].assign(Day=1))

    # Drop rows with NaN values that might have been created during conversion
    print("[8/8] Dropping rows with NaN values...")
    attribute_df.dropna(inplace=True)
    node_df.dropna(inplace=True)

    print("--- Data Loading and Preprocessing Finished ---\n")
    return attribute_df, node_df


if __name__ == "__main__":
    # Example usage from project root
    attr_df, nodes_df = load_and_preprocess_data(base_path=str(Path(__file__).resolve().parents[3]))
    print("Attribute DataFrame head:")
    print(attr_df.head())
    print("\nNode DataFrame head:")
    print(nodes_df.head())
