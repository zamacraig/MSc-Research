import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class TemporalSpatialAttentionNodeClassifier(nn.Module):
    """
    Temporal-Spatial GNN using Graph Attention (GAT) + GRU.

    Architecture per snapshot
    ─────────────────────────
    InputNorm  →  GATConv  →  BatchNorm1d  →  ReLU  →  Dropout
               →  GATConv  →  BatchNorm1d  →  ReLU
               →  GRU  (carries hidden state across snapshot sequence)
               →  Dropout  →  Linear  →  log_softmax

    Parameters
    ----------
    in_channels     : node feature size (8 weather features + 2 month sin/cos = 10)
    hidden_channels : width of all hidden layers and the GRU state
    num_classes     : number of target classes
    heads           : GAT attention heads (default 4)
    gru_layers      : stacked GRU layers (default 1)
    dropout         : dropout probability (default 0.3)
    input_norm      : whether to apply BatchNorm to input features (default True)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        heads: int = 4,
        gru_layers: int = 1,
        dropout: float = 0.3,
        input_norm: bool = True,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self._gru_layers = gru_layers
        self._use_input_norm = input_norm

        # --- Input Normalization (addresses Feature Drift) ---
        if input_norm:
            self.input_norm = nn.BatchNorm1d(in_channels)

        # --- Spatial (GAT) ---
        self.attn_conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=False)
        self.attn_conv2 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # --- Temporal (GRU) ---
        self.gru = nn.GRU(
            hidden_channels, hidden_channels,
            num_layers=gru_layers, batch_first=False,
        )

        # --- Classifier ---
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight=None, hidden=None, batch=None):
        """
        Returns (log_probs [num_nodes, num_classes], hidden_out [gru_layers, num_nodes, H]).
        Pass hidden=None to start a new sequence (zeros initialisation).
        """
        # Input normalization (handles Feature Drift)
        if self._use_input_norm:
            x = self.input_norm(x)

        # Spatial with skip connection (residual)
        h1 = F.relu(self.bn1(self.attn_conv1(x, edge_index)))
        h1 = self.dropout1(h1)
        h2 = F.relu(self.bn2(self.attn_conv2(h1, edge_index)))
        x = h1 + h2  # Skip connection: prevents information loss in deeper layers

        # Temporal
        num_nodes = x.size(0)
        if hidden is None:
            hidden = x.new_zeros(self._gru_layers, num_nodes, self.hidden_channels)
        x_gru, hidden_out = self.gru(x.unsqueeze(0), hidden)
        x = self.dropout2(x_gru.squeeze(0))

        if batch is not None:
            x = global_mean_pool(x, batch)

        return F.log_softmax(self.classifier(x), dim=1), hidden_out


if __name__ == "__main__":
    model = TemporalSpatialAttentionNodeClassifier(in_channels=10, hidden_channels=16, num_classes=5, heads=4)
    print(model)
