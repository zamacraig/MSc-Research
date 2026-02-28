import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class TemporalSpatialNodeClassifier(nn.Module):
    """
    Temporal-Spatial GNN for node-level classification.

    Architecture per snapshot
    ─────────────────────────
    InputNorm  →  GCNConv  →  BatchNorm1d  →  ReLU  →  Dropout
               →  GCNConv  →  BatchNorm1d  →  ReLU
               →  GRU  (carries hidden state across snapshot sequence)
               →  Dropout  →  Linear  →  log_softmax

    Improvements over the original:
    • InputNorm (BatchNorm1d on raw features) — directly addresses input-level
      feature drift by normalising the raw node features before any GCN layer.
    • BatchNorm1d after each GCN layer — normalises per-feature across nodes
      within each snapshot, handling intermediate-level distribution shifts.
    • GRU temporal layer — hidden state carries context from each processed
      snapshot to the next, addressing Temporal Autocorrelation gap.

    Parameters
    ----------
    in_channels     : node feature size (8 weather features + 2 month sin/cos = 10)
    hidden_channels : width of all hidden layers and the GRU state
    num_classes     : number of target classes
    gru_layers      : stacked GRU layers (default 1)
    dropout         : dropout probability (default 0.3)
    input_norm      : whether to apply BatchNorm to input features (default True)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        gru_layers: int = 1,
        dropout: float = 0.3,
        input_norm: bool = True,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self._gru_layers = gru_layers
        self._use_input_norm = input_norm

        # --- Input Normalization (addresses Feature Drift) ---
        # Normalises raw features across nodes per snapshot, stabilising inputs
        # when feature distributions shift over time.
        if input_norm:
            self.input_norm = nn.BatchNorm1d(in_channels)

        # --- Spatial (GCN) ---
        self.spatial_conv1 = GCNConv(in_channels, hidden_channels)
        self.spatial_conv2 = GCNConv(hidden_channels, hidden_channels)

        # BatchNorm1d: normalises per-feature across the node dimension within each
        # snapshot, making the model robust to snapshot-level distribution shifts.
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # --- Temporal (GRU) ---
        # Treats every node as an independent sequence element.
        # Hidden state shape: [gru_layers, num_nodes, hidden_channels]
        self.gru = nn.GRU(
            hidden_channels, hidden_channels,
            num_layers=gru_layers, batch_first=False,
        )

        # --- Classifier ---
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight=None, hidden=None, batch=None):
        """
        Parameters
        ----------
        x           : node features          [num_nodes, in_channels]
        edge_index  : edge connectivity      [2, num_edges]
        edge_weight : edge weights           [num_edges]  (optional)
        hidden      : GRU hidden state       [gru_layers, num_nodes, hidden_channels]
                      Pass None to start a new sequence (initialised with zeros).
        batch       : batch assignment       [num_nodes]  (optional, graph-level pool)

        Returns
        -------
        (log_probs, hidden_out)
            log_probs  : [num_nodes, num_classes]
            hidden_out : [gru_layers, num_nodes, hidden_channels]
        """
        # Input normalization (handles Feature Drift)
        if self._use_input_norm:
            x = self.input_norm(x)

        # Spatial with skip connection (residual)
        h1 = F.relu(self.bn1(self.spatial_conv1(x, edge_index, edge_weight)))
        h1 = self.dropout1(h1)
        h2 = F.relu(self.bn2(self.spatial_conv2(h1, edge_index, edge_weight)))
        x = h1 + h2  # Skip connection: prevents information loss in deeper layers

        # Temporal — unsqueeze seq_len=1 so GRU sees [1, num_nodes, H]
        num_nodes = x.size(0)
        if hidden is None:
            hidden = x.new_zeros(self._gru_layers, num_nodes, self.hidden_channels)
        x_gru, hidden_out = self.gru(x.unsqueeze(0), hidden)   # [1, N, H]
        x = self.dropout2(x_gru.squeeze(0))                     # [N, H]

        if batch is not None:
            x = global_mean_pool(x, batch)

        return F.log_softmax(self.classifier(x), dim=1), hidden_out


if __name__ == "__main__":
    model = TemporalSpatialNodeClassifier(in_channels=10, hidden_channels=16, num_classes=5)
    print("Model architecture:")
    print(model)
