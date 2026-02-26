import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class TemporalSpatialAttentionNodeClassifier(nn.Module):
    """
    A Temporal-Spatial Graph Neural Network using attention (GAT) for node classification.
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, heads: int = 4):
        super(TemporalSpatialAttentionNodeClassifier, self).__init__()

        # Spatial graph attention convolutions
        self.attn_conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=False)
        self.attn_conv2 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)

        # Layer Normalization for spatial processing
        self.layer_norm_spatial1 = nn.LayerNorm(hidden_channels)
        self.layer_norm_spatial2 = nn.LayerNorm(hidden_channels)

        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)

        # Temporal processing layers (simple linear temporal mixing)
        self.temporal_fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.temporal_fc2 = nn.Linear(hidden_channels, hidden_channels)

        # Layer Normalization for temporal processing
        self.layer_norm_temporal1 = nn.LayerNorm(hidden_channels)
        self.layer_norm_temporal2 = nn.LayerNorm(hidden_channels)

        # Final classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # Spatial processing with attention
        x = F.relu(self.layer_norm_spatial1(self.attn_conv1(x, edge_index)))
        x = self.dropout1(x)
        x = F.relu(self.layer_norm_spatial2(self.attn_conv2(x, edge_index)))

        # Temporal processing
        x = F.relu(self.layer_norm_temporal1(self.temporal_fc1(x)))
        x = self.dropout2(x)
        x = F.relu(self.layer_norm_temporal2(self.temporal_fc2(x)))

        if batch is not None:
            x = global_mean_pool(x, batch)

        out = self.classifier(x)
        out = F.log_softmax(out, dim=1)
        return out


if __name__ == "__main__":
    model = TemporalSpatialAttentionNodeClassifier(in_channels=8, hidden_channels=16, num_classes=5, heads=4)
    print(model)
