
import pandas as pd
import numpy as np
import os

def load_and_preprocess_data(base_path="."):
    """
    Loads and preprocesses the weather station data.

    Args:
        base_path (str): The base path to the data directory.

    Returns:
        tuple: A tuple containing the preprocessed attribute_df and node_df.
    """
    print("\n--- Starting Data Loading and Preprocessing ---")
    
    # Construct the correct path to the dataset folder
    dataset_path = os.path.join(base_path, "Dataset", "OriginalDataset")

    # Load station metadata
    print("[1/8] Loading station metadata...")
    node_df = pd.read_csv(os.path.join("../", dataset_path, "StationsData.csv"), sep=";", skiprows=[0], header=None,
                         names=["ComputerNo", "StationName", "Latitude", "Longitude", "Altitude"])

    # Load weather data for all stations
    print("[2/8] Loading weather data for all stations...")
    data_files = [
        "St_BrandvleiB.csv", "St_Kakamas.csv", "St_Luckhof.csv",
        "St_Olifantshoek.csv", "St_Prieska.csv", "St_Vaalharts.csv",
        "St_VanWyksvleiB.csv", "St_BerkleyWest.csv"
    ]
    
    frames = []
    for file in data_files:
        file_path = os.path.join("../", dataset_path, file)
        df = pd.read_csv(file_path, sep=";", skiprows=[0], header=None,
                         names=["Compno", "Year", "Month", "AverageMaximumTemperature", "AverageMinimunTemperature", 
                                "AverageMaximumRelativeHumidity", "AverageMinimumRelativeHumidity", "TotalRainfall", 
                                "AverageTotalRadiation", "AverageWindSpeed", "AverageTotalRelativeEvaporation"])
        frames.append(df)
        
    attribute_df = pd.concat(frames)

    # Data Cleaning
    print("[3/8] Cleaning data (replacing commas and tildes)...")
    attribute_df = attribute_df.replace(',', '.', regex=True)
    attribute_df = attribute_df.replace('~', '0', regex=True)
    node_df = node_df.replace(',', '.', regex=True)
    node_df = node_df.replace('~', '0', regex=True)

    # Convert to numeric
    print("[4/8] Converting data to numeric types...")
    attribute_df = attribute_df.apply(pd.to_numeric, errors='coerce')
    node_df[['ComputerNo', 'Latitude', 'Longitude', 'Altitude']] = node_df[['ComputerNo', 'Latitude', 'Longitude', 'Altitude']].apply(pd.to_numeric, errors='coerce')

    # Calculate target column
    print("[5/8] Calculating target column...")
    attribute_df["Target"] = attribute_df.drop(['Compno', 'Year', 'Month'], axis=1).mean(axis=1)

    # Fix null values
    print("[6/8] Handling null values (forward-fill for TotalRainfall)...")
    attribute_df['TotalRainfall'] = attribute_df['TotalRainfall'].fillna(method='ffill')
    
    # Create TimeStamp
    print("[7/8] Creating TimeStamp...")
    attribute_df['TimeStamp'] = pd.to_datetime(attribute_df[['Year', 'Month']].assign(Day=1))
    
    # Drop rows with NaN values that might have been created during conversion
    print("[8/8] Dropping rows with NaN values...")
    attribute_df.dropna(inplace=True)
    node_df.dropna(inplace=True)

    print("--- Data Loading and Preprocessing Finished ---\n")
    return attribute_df, node_df

if __name__ == '__main__':
    # Example of how to use the function, assuming the script is run from the project root
    attribute_df, node_df = load_and_preprocess_data(base_path=".")
    print("Attribute DataFrame head:")
    print(attribute_df.head())
    print("\nNode DataFrame head:")
    print(node_df.head())
