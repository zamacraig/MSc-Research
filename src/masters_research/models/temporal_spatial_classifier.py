import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class TemporalSpatialNodeClassifier(nn.Module):
    """
    A Temporal-Spatial Graph Neural Network for node classification.
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int):
        super(TemporalSpatialNodeClassifier, self).__init__()

        # Spatial graph convolutions
        self.spatial_conv1 = GCNConv(in_channels, hidden_channels)
        self.spatial_conv2 = GCNConv(hidden_channels, hidden_channels)

        # Layer Normalization for spatial processing
        self.layer_norm_spatial1 = nn.LayerNorm(hidden_channels)
        self.layer_norm_spatial2 = nn.LayerNorm(hidden_channels)

        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)

        # Temporal processing layers (using Linear layers as a simple form of temporal processing)
        self.temporal_fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.temporal_fc2 = nn.Linear(hidden_channels, hidden_channels)

        # Layer Normalization for temporal processing
        self.layer_norm_temporal1 = nn.LayerNorm(hidden_channels)
        self.layer_norm_temporal2 = nn.LayerNorm(hidden_channels)

        # Final classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight, batch=None):
        """
        Forward pass for the model.
        """
        # Spatial processing
        x = F.relu(self.layer_norm_spatial1(self.spatial_conv1(x, edge_index, edge_weight)))
        x = self.dropout1(x)
        x = F.relu(self.layer_norm_spatial2(self.spatial_conv2(x, edge_index, edge_weight)))

        # Temporal processing
        x = F.relu(self.layer_norm_temporal1(self.temporal_fc1(x)))
        x = self.dropout2(x)
        x = F.relu(self.layer_norm_temporal2(self.temporal_fc2(x)))

        # Global pooling if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)

        # Classification
        out = self.classifier(x)
        out = F.log_softmax(out, dim=1)
        return out


if __name__ == "__main__":
    # Example model instantiation
    model = TemporalSpatialNodeClassifier(in_channels=8, hidden_channels=16, num_classes=5)
    print("Model architecture:")
    print(model)
