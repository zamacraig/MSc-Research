import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class SimpleGCNClassifier(nn.Module):
    """
    Simple GCN-based node classifier WITHOUT temporal (GRU) component.
    
    This model processes each snapshot independently, avoiding the hidden state
    accumulation that may cause mode collapse in the GRU-based model.
    
    Architecture per snapshot
    ─────────────────────────
    InputNorm  →  GCNConv  →  BatchNorm1d  →  ReLU  →  Dropout
               →  GCNConv  →  BatchNorm1d  →  ReLU  →  Dropout
               →  GCNConv  →  BatchNorm1d  →  ReLU
               →  Linear  →  log_softmax

    Parameters
    ----------
    in_channels     : node feature size (8 weather features + 4 fourier = 12)
    hidden_channels : width of all hidden layers
    num_classes     : number of target classes
    dropout         : dropout probability (default 0.3)
    input_norm      : whether to apply BatchNorm to input features (default True)
    num_layers      : number of GCN layers (default 3)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        dropout: float = 0.3,
        input_norm: bool = True,
        num_layers: int = 3,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self._use_input_norm = input_norm
        self._num_layers = num_layers

        # --- Input Normalization (addresses Feature Drift) ---
        if input_norm:
            self.input_norm = nn.BatchNorm1d(in_channels)

        # --- GCN layers ---
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # First layer: in_channels -> hidden
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.dropouts.append(nn.Dropout(dropout))
        
        # Middle layers: hidden -> hidden
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Last GCN layer (no dropout after this one, goes to classifier)
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # --- Classifier ---
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight=None, hidden=None, batch=None):
        """
        Parameters
        ----------
        x           : node features          [num_nodes, in_channels]
        edge_index  : edge connectivity      [2, num_edges]
        edge_weight : edge weights           [num_edges]  (optional)
        hidden      : ignored (compatibility with GRU model interface)
        batch       : batch assignment       [num_nodes]  (optional, graph-level pool)

        Returns
        -------
        (log_probs, None)
            log_probs : [num_nodes, num_classes]
            None      : placeholder for hidden state (GRU compatibility)
        """
        # Input normalization (handles Feature Drift)
        if self._use_input_norm:
            x = self.input_norm(x)

        # GCN layers with skip connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_prev = x
            x = conv(x, edge_index, edge_weight)
            x = bn(x)
            x = F.relu(x)
            
            # Skip connection if dimensions match (after first layer)
            if i > 0 and x.size(-1) == x_prev.size(-1):
                x = x + x_prev
            
            # Apply dropout (except last layer)
            if i < len(self.dropouts):
                x = self.dropouts[i](x)

        if batch is not None:
            x = global_mean_pool(x, batch)

        # Return (logits, None) to match GRU model interface
        return F.log_softmax(self.classifier(x), dim=1), None


if __name__ == "__main__":
    model = SimpleGCNClassifier(in_channels=12, hidden_channels=16, num_classes=3)
    print("Model architecture:")
    print(model)
