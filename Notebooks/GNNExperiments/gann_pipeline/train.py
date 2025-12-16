
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from TemporalDataLoaderClass import TemporalDataLoader
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal, temporal_signal_split, DynamicGraphTemporalSignalBatch
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt

from model import TemporalSpatialNodeClassifier

def create_temporal_dataset(final_df):
    """
    Creates a temporal dataset from the final processed dataframe.
    """
    print("Creating temporal dataset...")
    
    # Label encode Compno columns to be 0-indexed
    le = LabelEncoder()
    final_df['Compno_x_encoded'] = le.fit_transform(final_df['Compno_x'])
    final_df['Compno_y_encoded'] = le.transform(final_df['Compno_y'])
    print("Label encoding complete.")

    # --- Feature Scaling and Target Discretization ---
    # As pointed out in the analysis, this should be done after splitting to avoid data leakage.
    # For this conversion script, we'll keep it similar to the notebook for now.
    scaler = MinMaxScaler()
    
    # Scale numerical features
    features_to_scale = [col for col in final_df.columns if '_x' in col and 'Compno' not in col and 'Target' not in col and 'TimeStamp' not in col]
    print("\n" + "="*50)
    print(" " * 18 + "DATASET FEATURES")
    print("="*50)
    print("Features being scaled:")
    for feature in features_to_scale:
        print(f"- {feature}")
    print("="*50)

    final_df[features_to_scale] = scaler.fit_transform(final_df[features_to_scale])
    print("Feature scaling complete.")

    # Discretize target variable
    final_df['Target_x'] = pd.qcut(final_df['Target_x'], q=5, labels=False, duplicates='drop')
    print("Target discretization complete.")

    # Assuming 'final_df' has a 'timestamp' column
    timestamps = final_df['TimeStamp'].unique()

    # Initialize lists to store data for each timestamp
    edge_indices_list = []
    edge_weights_list = []
    node_features_list = []
    targets_list = []
    all_batches = []
    timestamps_list = []

    print("\n" + "="*50)
    print(" " * 18 + "DATASET COUNTS")
    print("="*50)
    print(f"Number of unique timestamps: {len(timestamps)}")
    print(f"Total number of nodes: {len(final_df['Compno_x_encoded'].unique())}")
    print(f"Total number of edges (time-stamped): {len(final_df)}")
    print("="*50)


    for timestamp in timestamps:
        df_timestamp = final_df[final_df['TimeStamp'] == timestamp]
    
        # Convert timestamp string to numerical (Unix timestamp)
        if isinstance(timestamp, str):
            try:
                dt_object = datetime.strptime(timestamp, "%Y-%m-%d")
                timestamp_numeric = dt_object.timestamp()
            except ValueError:
                raise ValueError(f"Invalid timestamp format: {timestamp}")
        else:
            timestamp_numeric = float(timestamp)  # If it's already a number
        
        # Node features
        node_features = df_timestamp[['Compno_x', 'AverageMaximumTemperature_x', 'AverageMinimunTemperature_x', 
                                    'AverageMaximumRelativeHumidity_x', 'AverageMinimumRelativeHumidity_x', 
                                    'TotalRainfall_x', 'AverageTotalRadiation_x', 'AverageWindSpeed_x', 
                                    'AverageTotalRelativeEvaporation_x']].drop_duplicates()
        node_features = node_features.sort_values(by='Compno_x')
        x = torch.tensor(node_features[['AverageMaximumTemperature_x', 'AverageMinimunTemperature_x', 
                                        'AverageMaximumRelativeHumidity_x', 'AverageMinimumRelativeHumidity_x', 
                                        'TotalRainfall_x', 'AverageTotalRadiation_x', 'AverageWindSpeed_x', 
                                        'AverageTotalRelativeEvaporation_x']].values, dtype=torch.float)
        
        # Edge indices
        edge_indices = torch.tensor(df_timestamp[['Compno_x', 'Compno_y']].values.T, dtype=torch.long)
        
        # Edge weights
        edge_weights = torch.tensor(df_timestamp['distance_km'].values, dtype=torch.float)
        
        #edge_indices = [torch.randint(0, n_nodes, (2, n_edges), dtype=torch.long) for _ in df_timestamps] 
        #edge_weights = [torch.rand(n_edges) for _ in timestamps]
        
        # Targets
        targets = df_timestamp[['Compno_x', 'Target_x']].drop_duplicates()['Target_x'].values
        targets_np = np.array(targets, dtype=np.float)
        targets_tensor = torch.tensor(targets_np, dtype=torch.float)
        
        # Generate batches (assuming each timestamp is a separate batch)
        batch = torch.tensor([i for i in range(len(df_timestamp))], dtype=torch.long)
        
        snapshot_timestamp = torch.FloatTensor([timestamp])
        
        # Append to lists
        node_features_list.append(x)
        edge_indices_list.append(edge_indices)
        edge_weights_list.append(edge_weights)
        targets_list.append(targets_tensor)
        all_batches.append(batch)
        timestamps_list.append(snapshot_timestamp)   # Keep the timestamp

    # Ensure the length consistency across all lists
    assert len(edge_indices_list) == len(edge_weights_list) == len(node_features_list) == len(targets_list) == len(all_batches) == len(timestamps_list), "Temporal dimension inconsistency."

    print("Temporal dataset creation complete.")
    
    return DynamicGraphTemporalSignalBatch(
        edge_indices=edge_indices_list,
        edge_weights=edge_weights_list,
        features=node_features_list,
        targets=targets_list,
        batches=all_batches,
        timestamps=timestamps_list
    )


def train(model, optimizer, criterion, train_loader):
    model.train()
    total_loss = 0
    print("Starting training epoch...")
    loader_len = len(list(train_loader))
    for i, snapshot in enumerate(train_loader):
        #print(f"  Training on snapshot {i+1}/{loader_len}")
        optimizer.zero_grad()
        out = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        
        targets_long = snapshot.y.clone().to(torch.long)
        loss = criterion(out, targets_long)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("Training epoch finished.")
    return total_loss / loader_len if loader_len > 0 else total_loss

def test(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    print("Starting testing...")
    loader_len = len(list(loader))
    with torch.no_grad():
        for i, snapshot in enumerate(loader):
            #print(f"  Testing on snapshot {i+1}/{loader_len}")
            out = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(snapshot.y.cpu().numpy())
    print("Testing finished.")
    return np.array(all_preds), np.array(all_labels)

import matplotlib.pyplot as plt

def main():
    """
    Main function to run the training and evaluation pipeline.
    """
    try:
        from data_loader import load_and_preprocess_data
        from graph_constructor import create_graph_data

        # Load and create graph data
        print("Loading and preprocessing data...")
        attribute_df, node_df = load_and_preprocess_data(base_path="../..")
        print("Data loading and preprocessing complete.")

        # --- Filter timestamps based on missing nodes ---
        print("\nFiltering timestamps...")
        all_node_ids = set(node_df['ComputerNo'].unique())
        nodes_per_timestamp = attribute_df.groupby('TimeStamp')['Compno'].nunique()
        
        # Timestamps with 2 or less missing nodes
        total_nodes = len(all_node_ids)
        timestamps_to_keep = nodes_per_timestamp[total_nodes - nodes_per_timestamp <= 2].index
        
        original_timestamp_count = len(attribute_df['TimeStamp'].unique())
        
        attribute_df = attribute_df[attribute_df['TimeStamp'].isin(timestamps_to_keep)]
        
        filtered_timestamp_count = len(attribute_df['TimeStamp'].unique())
        
        print(f"Filtered timestamps from {original_timestamp_count} to {filtered_timestamp_count}.")
        print(f"Keeping timestamps with {total_nodes - 2} or more nodes.")
        # -----------------------------------------

        print("\nCreating graph data...")
        #final_df = create_graph_data(attribute_df, node_df)
        final_df = pd.read_csv('../../../Dataset/TransientData/corrected1_df.csv')    
        print("Graph data creation complete.")

        # --- Generate and Save Plot ---
        print("\nGenerating plot of nodes per timestamp...")
        nodes_per_timestamp = final_df.groupby('TimeStamp')['Compno_x'].nunique()
        
        plt.figure(figsize=(15, 7))
        nodes_per_timestamp.plot(kind='bar')
        plt.title('Number of Nodes per Timestamp')
        plt.xlabel('Timestamp')
        plt.ylabel('Number of Nodes')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_path = 'nodes_per_timestamp.png'
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}\n")
        # --------------------------------

        
        # Create dataset
        dataset = create_temporal_dataset(final_df)
        print("\n" + "="*50)
        print(" " * 18 + "DATASET INFO")
        print("="*50)
        print(dataset)
        
        print("\nSplitting dataset into training and testing sets...")
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
        print("Dataset splitting complete.")

        # Create a custom data loader
        train_loader = TemporalDataLoader(train_dataset)
        test_loader = TemporalDataLoader(test_dataset)

        # --- Model, Optimizer, Criterion ---
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
        model = TemporalSpatialNodeClassifier(in_channels=8, hidden_channels=16, num_classes=5).to(device)
        
        print("\n" + "="*50)
        print(" " * 15 + "MODEL ARCHITECTURE")
        print("="*50)
        print(model)
        print("="*50 + "\n")

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        # --- Training Loop ---
        print("Starting model training...")
        for epoch in range(1, 201):
            print(f"\nEpoch {epoch}/200")
            loss = train(model, optimizer, criterion, train_dataset)
            
            preds, labels = test(model, test_dataset)
            accuracy = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='weighted')
            print(f"Epoch {epoch:03d} | Train Loss: {loss:.4f} | Test Acc: {accuracy:.4f} | Test F1: {f1:.4f}")
        print("Model training finished.")
                
        # --- Save Model ---
        model_save_path = "temporal_spatial_classifier_state.pth"
        print(f"Saving model to {model_save_path}...")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    except Exception as e:
        import traceback
        print("An error occurred during the training process:")
        traceback.print_exc()

if __name__ == '__main__':
    main()
