"""
Temporal Attention GNN - Enhanced temporal modeling with attention over past snapshots.

Instead of treating all past information equally (as a simple GRU does), this model
uses multi-head attention to selectively focus on relevant historical patterns.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


class TemporalAttention(nn.Module):
    """
    Multi-head attention over temporal history.
    
    Given current features and a history buffer of past hidden states,
    computes attention weights to selectively aggregate relevant past information.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        query : torch.Tensor
            Current hidden state [num_nodes, hidden_dim]
        history : torch.Tensor
            Past hidden states [history_len, num_nodes, hidden_dim]
            
        Returns
        -------
        torch.Tensor
            Attended output [num_nodes, hidden_dim]
        """
        if history.size(0) == 0:
            return query
        
        num_nodes = query.size(0)
        history_len = history.size(0)
        
        # Include current in the sequence for self-attention
        # [history_len + 1, num_nodes, hidden_dim]
        full_seq = torch.cat([history, query.unsqueeze(0)], dim=0)
        seq_len = full_seq.size(0)
        
        # Reshape for attention: [num_nodes, seq_len, hidden_dim]
        full_seq = full_seq.permute(1, 0, 2)
        
        # Compute Q, K, V
        Q = self.q_proj(query).view(num_nodes, 1, self.num_heads, self.head_dim)
        K = self.k_proj(full_seq).view(num_nodes, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(full_seq).view(num_nodes, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention: [num_nodes, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)  # [num_nodes, num_heads, 1, head_dim]
        K = K.transpose(1, 2)  # [num_nodes, num_heads, seq_len, head_dim]
        V = V.transpose(1, 2)  # [num_nodes, num_heads, seq_len, head_dim]
        
        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [num_nodes, num_heads, 1, seq_len]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # [num_nodes, num_heads, 1, head_dim]
        out = out.transpose(1, 2).contiguous().view(num_nodes, self.hidden_dim)
        
        return self.out_proj(out)


class TemporalAttentionGNN(nn.Module):
    """
    GNN with Temporal Attention for node classification.
    
    Architecture per snapshot
    ─────────────────────────
    InputNorm → GCNConv → BN → ReLU → Dropout
              → GCNConv → BN → ReLU (+ skip connection)
              → GRU (processes current snapshot)
              → TemporalAttention (attends over history buffer)
              → Dropout → Linear → log_softmax
    
    The key improvement over the base model is the TemporalAttention layer that
    allows the model to selectively focus on relevant past snapshots rather than
    treating all history equally through the GRU.
    
    Parameters
    ----------
    in_channels : int
        Number of input features per node
    hidden_channels : int
        Hidden dimension for all layers
    num_classes : int
        Number of output classes
    history_len : int, default 10
        Number of past snapshots to store in history buffer
    temporal_heads : int, default 4
        Number of attention heads for temporal attention
    gru_layers : int, default 1
        Number of GRU layers
    dropout : float, default 0.3
        Dropout probability
    input_norm : bool, default True
        Whether to apply BatchNorm to input features
    use_gat : bool, default False
        Use GAT instead of GCN for spatial layers
    gat_heads : int, default 4
        Number of GAT attention heads (if use_gat=True)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        history_len: int = 10,
        temporal_heads: int = 4,
        gru_layers: int = 1,
        dropout: float = 0.3,
        input_norm: bool = True,
        use_gat: bool = False,
        gat_heads: int = 4,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.history_len = history_len
        self._gru_layers = gru_layers
        self._use_input_norm = input_norm
        self._use_gat = use_gat
        
        # Input Normalization
        if input_norm:
            self.input_norm = nn.BatchNorm1d(in_channels)
        
        # Spatial layers (GCN or GAT)
        if use_gat:
            self.spatial_conv1 = GATConv(in_channels, hidden_channels, heads=gat_heads, concat=False)
            self.spatial_conv2 = GATConv(hidden_channels, hidden_channels, heads=gat_heads, concat=False)
        else:
            self.spatial_conv1 = GCNConv(in_channels, hidden_channels)
            self.spatial_conv2 = GCNConv(hidden_channels, hidden_channels)
        
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # GRU for sequential processing
        self.gru = nn.GRU(
            hidden_channels, hidden_channels,
            num_layers=gru_layers, batch_first=False,
        )
        
        # Temporal Attention
        self.temporal_attention = TemporalAttention(
            hidden_dim=hidden_channels,
            num_heads=temporal_heads,
            dropout=dropout,
        )
        
        # Layer norm after attention
        self.post_attn_norm = nn.LayerNorm(hidden_channels)
        
        # Classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)
        
        # History storage (NOT a registered buffer to avoid SWA buffer misalignment)
        # Using register_buffer causes issues with AveragedModel.update_parameters()
        # because the buffer list changes size after forward passes
        self._history = None
    
    def reset_history(self):
        """Clear the history buffer. Call at start of new sequence."""
        self._history = None
    
    def forward(self, x, edge_index, edge_weight=None, hidden=None, batch=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Node features [num_nodes, in_channels]
        edge_index : torch.Tensor
            Edge connectivity [2, num_edges]
        edge_weight : torch.Tensor, optional
            Edge weights [num_edges]
        hidden : torch.Tensor, optional
            GRU hidden state [gru_layers, num_nodes, hidden_channels]
        batch : torch.Tensor, optional
            Batch assignment for graph-level pooling
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (log_probs [num_nodes, num_classes], hidden_out [gru_layers, num_nodes, H])
        """
        # Input normalization
        if self._use_input_norm:
            x = self.input_norm(x)
        
        # Spatial with skip connection
        if self._use_gat:
            h1 = F.relu(self.bn1(self.spatial_conv1(x, edge_index)))
            h1 = self.dropout1(h1)
            h2 = F.relu(self.bn2(self.spatial_conv2(h1, edge_index)))
        else:
            h1 = F.relu(self.bn1(self.spatial_conv1(x, edge_index, edge_weight)))
            h1 = self.dropout1(h1)
            h2 = F.relu(self.bn2(self.spatial_conv2(h1, edge_index, edge_weight)))
        
        x = h1 + h2  # Skip connection
        
        # GRU
        num_nodes = x.size(0)
        if hidden is None:
            hidden = x.new_zeros(self._gru_layers, num_nodes, self.hidden_channels)
        
        x_gru, hidden_out = self.gru(x.unsqueeze(0), hidden)
        x = x_gru.squeeze(0)  # [num_nodes, hidden_channels]
        
        # Temporal Attention over history
        # IMPORTANT: Detach history tensors to prevent backprop through past snapshots
        if self._history is None:
            # First snapshot - no history yet
            self._history = x.detach().unsqueeze(0)  # [1, num_nodes, H]
            x_attn = x
        else:
            # Apply temporal attention (history is already detached)
            x_attn = self.temporal_attention(x, self._history)
            
            # Update history buffer (FIFO) - detach current x before storing
            x_detached = x.detach().unsqueeze(0)
            if self._history.size(0) >= self.history_len:
                self._history = torch.cat([self._history[1:], x_detached], dim=0)
            else:
                self._history = torch.cat([self._history, x_detached], dim=0)
        
        # Residual + Norm
        x = self.post_attn_norm(x + x_attn)
        x = self.dropout2(x)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return F.log_softmax(self.classifier(x), dim=1), hidden_out


if __name__ == "__main__":
    print("Testing TemporalAttentionGNN...")
    model = TemporalAttentionGNN(
        in_channels=19,  # 8 base + 9 lag + 2 month
        hidden_channels=32,
        num_classes=5,
        history_len=10,
        temporal_heads=4,
    )
    print(model)
    
    # Simulate 3 snapshots
    num_nodes = 9
    for t in range(3):
        x = torch.randn(num_nodes, 19)
        edge_index = torch.randint(0, num_nodes, (2, 20))
        edge_weight = torch.randn(20)
        
        out, hidden = model(x, edge_index, edge_weight)
        print(f"Snapshot {t}: output shape = {out.shape}, history size = {model._history.shape if model._history is not None else None}")
    
    print("\nModel parameter count:", sum(p.numel() for p in model.parameters()))
