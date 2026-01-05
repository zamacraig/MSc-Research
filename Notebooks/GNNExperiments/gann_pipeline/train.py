
import json
from pathlib import Path
import torch
import torch.optim as optim
from datetime import datetime
import itertools
import csv
import time
import os
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


def train(model, optimizer, criterion, train_loader, grad_clip=None):
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
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
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

def load_config():
    """Load config.json located next to this script and return as dict."""
    config_path = Path(__file__).with_name('config.json')
    cfg = {}
    if config_path.exists():
        try:
            with config_path.open('r') as fh:
                cfg = json.load(fh)
        except Exception:
            cfg = {}
    return cfg


def main():
    """Main function to run the training and evaluation pipeline.
    Configuration is loaded directly from `config.json` located next to this script.
    """
    cfg = load_config()
    try:
        from data_loader import load_and_preprocess_data
        from graph_constructor import create_graph_data

        # Build parameter grid from config (list-valued entries become grid dimensions)
        def generate_param_grid(cfg_in):
            list_keys = [k for k, v in cfg_in.items() if isinstance(v, list) and len(v) > 0]
            if not list_keys:
                return [cfg_in.copy()]
            pools = [cfg_in[k] for k in list_keys]
            grids = []
            for vals in itertools.product(*pools):
                c = cfg_in.copy()
                for k, v in zip(list_keys, vals):
                    c[k] = v
                grids.append(c)
            return grids

        grid = generate_param_grid(cfg)
        if not grid:
            grid = [cfg]

        # Output directory and results file
        save_dir = Path(cfg.get('save_dir', Path(__file__).parent))
        save_dir.mkdir(parents=True, exist_ok=True)
        results_path = save_dir / 'grid_results.csv'
        header = ['run_id', 'params', 'accuracy', 'f1', 'model_path', 'best_model_path', 'best_f1', 'duration_seconds']

        # reproducibility seed will be set per-run below

        # Load data once (heavy operations)
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
        data_path = cfg.get('data_path', '../../../Dataset/TransientData/corrected1_df.csv')
        final_df = pd.read_csv(data_path)
        print("Graph data creation complete.")

        # --- Generate and Save Plot ---
        if cfg.get('plot_nodes_per_timestamp', True):
            print("\nGenerating plot of nodes per timestamp...")
            nodes_per_timestamp = final_df.groupby('TimeStamp')['Compno_x'].nunique()
            plt.figure(figsize=(15, 7))
            nodes_per_timestamp.plot(kind='bar')
            plt.title('Number of Nodes per Timestamp')
            plt.xlabel('Timestamp')
            plt.ylabel('Number of Nodes')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_path = save_dir / 'nodes_per_timestamp.png'
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}\n")
        # --------------------------------

        
        # Create dataset
        dataset = create_temporal_dataset(final_df)
        print("\n" + "="*50)
        print(" " * 18 + "DATASET INFO")
        print("="*50)
        print(dataset)
        
        # Run grid of experiments (if config contains list-valued hyperparameters)
        for run_idx, cfg_run in enumerate(grid, start=1):
            run_prefix = cfg.get('run_name', 'run')
            run_id = f"{run_prefix}{run_idx}_{int(time.time())}"
            print(f"\n=== Starting grid run {run_idx}/{len(grid)} ({run_id}) ===")

            # set per-run seed
            seed = int(cfg_run.get('seed', cfg.get('seed', 42)))
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Split dataset according to this run's train ratio
            print("\nSplitting dataset into training and testing sets...")
            train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=cfg_run.get('train_ratio', cfg.get('train_ratio', 0.8)))
            print("Dataset splitting complete.")

            train_loader = TemporalDataLoader(train_dataset)
            test_loader = TemporalDataLoader(test_dataset)

            # Device selection
            cfg_device = cfg_run.get('device', cfg.get('device', None))
            if cfg_device == 'cpu':
                device = torch.device('cpu')
            elif cfg_device == 'cuda':
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"\nUsing device: {device}")

            # Model
            model = TemporalSpatialNodeClassifier(
                in_channels=int(cfg_run.get('in_channels', cfg.get('in_channels', 8))),
                hidden_channels=int(cfg_run.get('hidden_channels', cfg.get('hidden_channels', 16))),
                num_classes=int(cfg_run.get('num_classes', cfg.get('num_classes', 5)))
            ).to(device)
            print("\n" + "="*50)
            print(" " * 15 + "MODEL ARCHITECTURE")
            print("="*50)
            print(model)
            print("="*50 + "\n")

            opt_name = str(cfg_run.get('optimizer', cfg.get('optimizer', 'adam'))).lower()
            if opt_name == 'sgd':
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=float(cfg_run.get('lr', cfg.get('lr', 0.001))),
                    momentum=float(cfg_run.get('momentum', cfg.get('momentum', 0.9))),
                    weight_decay=float(cfg_run.get('weight_decay', cfg.get('weight_decay', 5e-4)))
                )
            else:
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=float(cfg_run.get('lr', cfg.get('lr', 0.001))),
                    weight_decay=float(cfg_run.get('weight_decay', cfg.get('weight_decay', 5e-4)))
                )
            criterion = nn.CrossEntropyLoss()

            # Optional scheduler
            scheduler_cfg = cfg_run.get('scheduler', cfg.get('scheduler', None))
            scheduler = None
            if isinstance(scheduler_cfg, dict) and scheduler_cfg.get('name') == 'StepLR':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=int(scheduler_cfg.get('step_size', 10)),
                    gamma=float(scheduler_cfg.get('gamma', 0.5))
                )

            # Training loop
            print("Starting model training...")
            start_time = time.time()
            best_f1 = -1.0
            best_model_path = None
            eval_interval = int(cfg_run.get('eval_interval', cfg.get('eval_interval', 1)))
            patience = cfg_run.get('early_stopping_patience', cfg.get('early_stopping_patience', None))
            min_delta = float(cfg_run.get('early_stopping_min_delta', cfg.get('early_stopping_min_delta', 0.0)))
            grad_clip = cfg_run.get('grad_clip', cfg.get('grad_clip', None))

            for epoch in range(1, int(cfg_run.get('epochs', cfg.get('epochs', 200))) + 1):
                print(f"\nEpoch {epoch}/{int(cfg_run.get('epochs', cfg.get('epochs', 200)))}")
                loss = train(model, optimizer, criterion, train_loader, grad_clip=grad_clip)

                # Evaluate per interval
                do_eval = (epoch % eval_interval == 0) or (epoch == int(cfg_run.get('epochs', cfg.get('epochs', 200))))
                if do_eval:
                    preds, labels = test(model, test_loader)
                    accuracy = accuracy_score(labels, preds)
                    f1 = f1_score(labels, preds, average='weighted')
                    print(f"Epoch {epoch:03d} | Train Loss: {loss:.4f} | Test Acc: {accuracy:.4f} | Test F1: {f1:.4f}")

                    # Track best and optionally save
                    if f1 > best_f1 + min_delta:
                        best_f1 = f1
                        if cfg_run.get('save_best', cfg.get('save_best', True)):
                            best_model_path = save_dir / f"best_{run_id}.pth"
                            torch.save(model.state_dict(), best_model_path)
                            print(f"Saved best model to {best_model_path}")

                    # Early stopping check
                    if patience is not None:
                        # if not improved, reduce patience
                        # We use a simple counter: reset when improved, else decrement
                        # To implement, store counter in cfg_run scoped variable
                        pass

                if scheduler is not None:
                    scheduler.step()
            duration = time.time() - start_time
            print(f"Model training finished in {duration:.1f}s.")

            # Final evaluation and save
            final_preds, final_labels = test(model, test_loader)
            final_acc = float(accuracy_score(final_labels, final_preds))
            final_f1 = float(f1_score(final_labels, final_preds, average='weighted'))

            model_save_path = save_dir / f"temporal_spatial_classifier_state_{run_id}.pth"
            print(f"Saving model to {model_save_path}...")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

            # Append results to CSV
            row = {
                'run_id': run_id,
                'params': json.dumps(cfg_run),
                'accuracy': final_acc,
                'f1': final_f1,
                'model_path': str(model_save_path),
                'best_model_path': str(best_model_path) if best_model_path else '',
                'best_f1': best_f1 if best_f1 >= 0 else final_f1,
                'duration_seconds': round(duration, 3)
            }
            write_header = not results_path.exists()
            with results_path.open('a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=header)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            print(f"Run {run_id} results saved to {results_path}")

    except Exception as e:
        import traceback
        print("An error occurred during the training process:")
        traceback.print_exc()

if __name__ == '__main__':
    main()
