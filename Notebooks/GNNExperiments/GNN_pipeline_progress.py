#!pip install torch_geometric_temporal
#!pip install networkx

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence, Union
from torch_geometric.data import Data
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import TGCN
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import DynamicGraphTemporalSignalBatch
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric_temporal.signal import temporal_signal_split, DynamicGraphTemporalSignalBatch
from torch_geometric.nn import GCNConv, global_mean_pool
import datetime
from datetime import datetime
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


warnings.simplefilter("ignore", DeprecationWarning)

# Set display option to show all rows 
pd.set_option('display.max_rows', None)

# Set display option to show all columns 
pd.set_option('display.max_columns', None)

corrected_df = pd.read_csv('corrected_df.csv')

#final_df = pd.read_csv('corrected_df.csv')
final_df = pd.read_csv('corrected1_df.csv')

def convert_timestamps_to_float_tensor(timestamps):
    new_timestamps = []
    for timestamp in timestamps:
        if isinstance(timestamp, torch.Tensor):
            new_timestamps.append(timestamp.float())
        elif isinstance(timestamp, (list, tuple)):
            new_timestamps.append(torch.FloatTensor(timestamp))
        elif isinstance(timestamp, str):
            try:
                # Convert timestamp string to Unix time
                dt_object = datetime.strptime(timestamp, "%Y-%m-%d")
                unix_timestamp = dt_object.timestamp()
                new_timestamps.append(torch.FloatTensor([unix_timestamp]))
            except ValueError:
                raise ValueError(f"Invalid timestamp format: {timestamp}")
        else:
            raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")
    
    return new_timestamps

def convert_targets_to_float_tensor(targets):
    new_targets = []
    for target in targets:
        if isinstance(target, torch.Tensor):
            new_targets.append(target.float())
        elif isinstance(target, (list, tuple)):
            new_targets.append(torch.FloatTensor(target))
        elif isinstance(target, str):
            # Convert string target to float tensor; assuming it is comma-separated numbers
            float_target = torch.FloatTensor([float(x) for x in target.split(',') if x.replace('.', '', 1).isdigit()])
            new_targets.append(float_target)
        else:
            raise ValueError(f"Unsupported target type: {type(target)}")
    return new_targets

def convert_batches_to_long_tensor(batches):
    new_batches = []
    for batch in batches:
        if isinstance(batch, torch.Tensor):
            new_batches.append(batch.long())
        elif isinstance(batch, (list, tuple)):
            new_batches.append(torch.LongTensor(batch))
        elif isinstance(batch, str):
            # Convert string batch to long tensor; assuming it is comma-separated integers
            long_batch = torch.LongTensor([int(x) for x in batch.split(',') if x.isdigit()])
            new_batches.append(long_batch)
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")
    return new_batches

def convert_edge_weights_to_float_tensor(edge_weights):
    new_edge_weights = []
    for weight in edge_weights:
        if isinstance(weight, torch.Tensor):
            new_edge_weights.append(weight.float())
        elif isinstance(weight, (list, tuple)):
            new_edge_weights.append(torch.FloatTensor(weight))
        elif isinstance(weight, str):
            # Convert string edge weight to float tensor; assuming it is comma-separated numbers
            float_weight = torch.FloatTensor([float(x) for x in weight.split(',') if x.replace('.', '', 1).isdigit()])
            new_edge_weights.append(float_weight)
        else:
            raise ValueError(f"Unsupported edge weight type: {type(weight)}")
    return new_edge_weights

def convert_features_to_float_tensor(features):
    new_features = []
    for feature in features:
        try:
            if isinstance(feature, torch.Tensor):
                new_features.append(feature.float())
            elif isinstance(feature, (list, tuple)):
                new_features.append(torch.FloatTensor(feature))
            elif isinstance(feature, str):
                # Convert string feature to float tensor; assuming it is comma-separated numbers
                float_feature = torch.FloatTensor([float(x) for x in feature.split(',') if x.replace('.', '', 1).isdigit()])
                new_features.append(float_feature)
            else:
                new_features.append(torch.FloatTensor(feature.to_numpy()))
        except ValueError:
            print(f"Warning: Skipping non-numeric feature: {feature}")
    return new_features

def convert_edges_to_long_tensor(edge_indices):
    new_edge_indices = []
    for edge in edge_indices:
        if isinstance(edge, torch.Tensor):
            new_edge_indices.append(edge.long())
        elif isinstance(edge, (list, tuple)):
            new_edge_indices.append(torch.LongTensor(edge))
        elif isinstance(edge, str):
            # Convert string edge index to long tensor; assuming it is comma-separated integers
            long_edge = torch.LongTensor([int(x) for x in edge.split(',') if x.isdigit()])
            new_edge_indices.append(long_edge)
        else:
            raise ValueError(f"Unsupported edge index type: {type(edge)}")
    return new_edge_indices


# Assuming 'final_df' has a 'timestamp' column
timestamps = final_df['TimeStamp'].unique()

# Initialize lists to store data for each timestamp
edge_indices_list = []
edge_weights_list = []
node_features_list = []
targets_list = []
all_batches = []
timestamps_list = []

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

# Create a DynamicGraphTemporalSignal object
dataset = DynamicGraphTemporalSignalBatch(
    edge_indices=edge_indices_list,
    edge_weights=edge_weights_list,
    features=node_features_list,
    targets=targets_list,
    batches=all_batches,
    timestamps=timestamps_list
)

# Custom DataLoader for DynamicGraphTemporalSignalBatch
class TemporalDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.snapshot_count = dataset.snapshot_count

    def __len__(self):
        return self.snapshot_count

    def __iter__(self):
        for i in range(len(self)):
            yield self.dataset[i]

# Split dataset into training and test sets
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

# Create a custom data loader
train_loader = TemporalDataLoader(train_dataset)
test_loader = TemporalDataLoader(test_dataset)


class TemporalSpatialNodeClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(TemporalSpatialNodeClassifier, self).__init__()
        
        # Spatial graph convolutions
        self.spatial_conv1 = GCNConv(in_channels, hidden_channels)
        self.spatial_conv2 = GCNConv(hidden_channels, hidden_channels)

        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.6)
        
        # Temporal processing layers
        self.temporal_fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.temporal_fc2 = nn.Linear(hidden_channels, hidden_channels)

         # **Fix: Ensure classifier is properly initialized**
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight, batch=None):
        #print('Before: ')
        #print(x)
        x = torch.stack((x))  # Convert list of tensors into `[num_nodes, num_features]`

        #print('\n\nAfter:')
        #print(x)
        
        edge_index = torch.stack(edge_index, dim=0)  # Shape should be `[2, num_edges]`
        edge_weight = torch.tensor(edge_weight).to(torch.float32)  # Ensure `[num_edges]` shape

        #print('\n\nDebbugging')
        #print(f"x type: {type(x)}, x length: {len(x) if isinstance(x, list) else 'N/A'}")
        #print(f"x shape: {x.shape}, dtype: {x.dtype}")
        #print(f"edge_index shape: {edge_index.shape}, dtype: {edge_index.dtype}")
        #print(f"edge_weight shape: {edge_weight.shape}, dtype: {edge_weight.dtype}")

        
        # Spatial processing
        x = F.relu(self.spatial_conv1(x, edge_index, edge_weight))
        x = self.dropout1(x)
        x = F.relu(self.spatial_conv2(x, edge_index, edge_weight))
        #print("Feature Shape after spatial:", x.shape)
        
        # Temporal processing
        x = F.relu(self.temporal_fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.temporal_fc2(x))
        
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        out = self.classifier(x)
        out = F.log_softmax(out, dim=1) 
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

node_features_size=8
hidden_channels=8
num_classes=5

param_grid = {
    "hidden_channels": [8, 16, 32],
    "learning_rate": [0.001, 0.01, 0.1],
    "dropout": [0.3, 0.5, 0.6]
}

class GraphModelWrapper:
    def __init__(self, in_channels=8, hidden_channels=16, num_classes=5, learning_rate=0.001, dropout=0.5):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dropout = dropout

        # Initialize Model
        self.model = TemporalSpatialNodeClassifier(in_channels, hidden_channels, num_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, train_X, train_y, epochs=130):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0  # Track cumulative loss
            
            for batch_X, batch_y in zip(train_X, train_y):
                
                #print()
                #print(batch_X)
                #print()
                
                # Process input features
                features, edge_indices, edge_weights, targets, batch, timestamps = batch_X
                targets = batch_y.to(torch.long)

                # Convert inputs to correct tensor formats
                edge_indices = convert_edges_to_long_tensor(edge_indices)
                edge_weights = convert_edge_weights_to_float_tensor(edge_weights)
                features = convert_features_to_float_tensor(features)
                targets = targets.to(torch.long)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(features, edge_indices, edge_weights)
                loss = self.criterion(output, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
        # Print epoch loss
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}")
        
        return self

    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                features, edge_indices, edge_weights , targets, _, _ = batch
                output = self.model(features, edge_indices, edge_weights)
                predictions.append(output.argmax(dim=1).cpu().numpy())
        
        return np.concatenate(predictions, axis=0)

    def get_params(self, deep=True):
        """ Allows GridSearchCV to fetch hyperparameters """
        return {
            "in_channels": self.in_channels,
            "hidden_channels": self.hidden_channels,
            "num_classes": self.num_classes,
            "learning_rate": self.learning_rate,
            "dropout": self.dropout
        }

    def set_params(self, **params):
        """ Allows GridSearchCV to modify hyperparameters dynamically """
        for param, value in params.items():
            setattr(self, param, value)
        return self
    

# Initialize Grid Search
grid_search = GridSearchCV(
    estimator=GraphModelWrapper(in_channels=8, hidden_channels=8, num_classes=5, learning_rate=0.001, dropout=0.6),
    param_grid=param_grid,
    scoring="accuracy",
    cv=2,  # Cross-validation with 3 folds,
    verbose=3
)

train_X = []
train_y = []

for dynamic_graph_batch in train_loader:
    features, edge_indices, edge_weights, targets, batch, timestamps = dynamic_graph_batch

    train_X.append((features[1], edge_indices[1], edge_weights[1], targets[1], batch[1], timestamps[1])) 
    train_y.append(targets[1])

train_X = tuple(train_X)  
train_y = [batch.y for batch in train_loader]  # Ensure proper indexing [batch[3] for batch in train_loader]  # Extract targets from each batch

# Run hyperparameter tuning
grid_search.fit(train_X, train_y)

best_model = grid_search.best_estimator_
print('\n\nFinal Results')
print("Best Model:", best_model)

print("Best Hyperparameters:", grid_search.best_params_)
