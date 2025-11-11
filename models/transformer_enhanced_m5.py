# transformer_enhanced_m5.py
"""
Enhanced Transformer for M5 Forecasting with:
- Quantile loss for better handling of intermittent demand
- Multi-scale temporal convolutions
- Hierarchical embeddings
- Improved attention mechanism
"""

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder


class PositionalEncoding(nn.Module):
    """Enhanced positional encoding with learnable component."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Standard sinusoidal encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Learnable positional embedding (helps with specific patterns)
        self.learned_pe = nn.Parameter(torch.zeros(max_len, 1, d_model))

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(0)
        x = x + self.pe[:seq_len] + self.learned_pe[:seq_len]
        return self.dropout(x)


class MultiScaleConv1D(nn.Module):
    """Multi-scale 1D convolutions to capture patterns at different timescales."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Fix: Distribute channels properly to avoid dimension mismatch
        # Split into 3 parts, giving remainder to the last conv
        c1 = out_channels // 3
        c2 = out_channels // 3
        c3 = out_channels - c1 - c2  # Takes remainder (e.g., 128-42-42=44)
        
        self.conv_3 = nn.Conv1d(in_channels, c1, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv1d(in_channels, c2, kernel_size=5, padding=2)
        self.conv_7 = nn.Conv1d(in_channels, c3, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        # x: [batch, features, seq_len]
        c3 = self.conv_3(x)
        c5 = self.conv_5(x)
        c7 = self.conv_7(x)
        out = torch.cat([c3, c5, c7], dim=1)
        return F.relu(self.bn(out))


class HierarchicalEmbedding(nn.Module):
    """Hierarchical embedding for store/item IDs."""
    def __init__(self, num_items, num_stores, num_depts, num_cats, embed_dim):
        super().__init__()
        self.item_embed = nn.Embedding(num_items, embed_dim // 4)
        self.store_embed = nn.Embedding(num_stores, embed_dim // 4)
        self.dept_embed = nn.Embedding(num_depts, embed_dim // 4)
        self.cat_embed = nn.Embedding(num_cats, embed_dim // 4)
        
    def forward(self, item_ids, store_ids, dept_ids, cat_ids):
        # All inputs: [batch]
        item_emb = self.item_embed(item_ids)
        store_emb = self.store_embed(store_ids)
        dept_emb = self.dept_embed(dept_ids)
        cat_emb = self.cat_embed(cat_ids)
        return torch.cat([item_emb, store_emb, dept_emb, cat_emb], dim=-1)


class EnhancedForecastingModel(nn.Module):
    """
    Enhanced Transformer with:
    - Quantile regression (3 quantiles: 0.1, 0.5, 0.9)
    - Multi-scale convolutions
    - Hierarchical embeddings
    - Residual connections
    """
    
    def __init__(
        self,
        FEATURE_LAG: int,
        output_dim: int,
        input_dim_seq: int,
        input_dim_static: int,
        embed_size: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        device: str = "cpu",
        # New parameters for hierarchical embeddings
        num_items: int = 3049,
        num_stores: int = 10,
        num_depts: int = 7,
        num_cats: int = 3,
        use_quantile: bool = True
    ):
        super().__init__()
        self.device = torch.device(device)
        self.feature_lag = FEATURE_LAG
        self.use_quantile = use_quantile
        self.output_dim = output_dim
        
        # --- 1. Enhanced Embedding ---
        # Multi-scale Conv1D for sequential features
        self.multiscale_conv = MultiScaleConv1D(input_dim_seq, embed_size)
        
        # Hierarchical embedding for categorical IDs
        hier_embed_dim = embed_size // 2
        self.hierarchical_embed = HierarchicalEmbedding(
            num_items, num_stores, num_depts, num_cats, hier_embed_dim
        )
        
        # Project concatenated embeddings to embed_size
        self.embedding_projection = nn.Linear(embed_size + hier_embed_dim, embed_size)
        
        self.position_encoder = PositionalEncoding(embed_size, dropout)
        
        # --- 2. Enhanced Transformer ---
        encoder_layers = TransformerEncoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # --- 3. Enhanced Regression Head ---
        hidden_size = embed_size * FEATURE_LAG
        
        # Deeper MLP with residual connections
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn2 = nn.BatchNorm1d(hidden_size // 4)
        
        # Output heads
        if use_quantile:
            # Predict 3 quantiles (0.1, 0.5, 0.9) for each timestep
            self.output_head = nn.Linear(hidden_size // 4, output_dim * 3)
        else:
            self.output_head = nn.Linear(hidden_size // 4, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_seq: Tensor, x_static: Tensor, src_key_padding_mask=None):
        """
        Args:
            x_seq: [batch, seq_len, input_dim_seq]
            x_static: [batch, 5] - [item, dept, cat, store, state]
        Returns:
            If quantile: [batch, output_dim * 3] (q10, q50, q90)
            Else: [batch, output_dim]
        """
        batch_size = x_seq.size(0)
        
        # --- 1. Multi-scale Conv Embedding ---
        # [batch, seq_len, features] -> [batch, features, seq_len]
        x_conv = self.multiscale_conv(x_seq.transpose(1, 2))
        # -> [batch, embed_size, seq_len] -> [batch, seq_len, embed_size]
        x_conv = x_conv.transpose(1, 2)
        
        # --- 2. Hierarchical Static Embeddings ---
        # Extract IDs from x_static (assuming order: item, dept, cat, store, state)
        item_ids = x_static[:, 0].long()
        dept_ids = x_static[:, 1].long()
        cat_ids = x_static[:, 2].long()
        store_ids = x_static[:, 3].long()
        
        hier_emb = self.hierarchical_embed(item_ids, store_ids, dept_ids, cat_ids)
        # [batch, hier_embed_dim]
        
        # Broadcast static embedding to all timesteps
        hier_emb_expanded = hier_emb.unsqueeze(1).expand(-1, x_conv.size(1), -1)
        # [batch, seq_len, hier_embed_dim]
        
        # Concatenate and project
        x = torch.cat([x_conv, hier_emb_expanded], dim=-1)
        x = self.embedding_projection(x)
        # [batch, seq_len, embed_size]
        
        # --- 3. Positional Encoding ---
        x = self.position_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # --- 4. Transformer ---
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        # [batch, seq_len, embed_size]
        
        # --- 5. Regression Head ---
        x = x.reshape(batch_size, -1)  # Flatten
        
        # Deep MLP with residuals
        x1 = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x2 = self.dropout(self.relu(self.bn2(self.fc2(x1))))
        
        # Output
        out = self.output_head(x2)
        
        if self.use_quantile:
            # Reshape to [batch, output_dim, 3]
            out = out.view(batch_size, self.output_dim, 3)
        
        return out


# ===== QUANTILE LOSS =====
class QuantileLoss(nn.Module):
    """
    Quantile loss for probabilistic forecasting.
    Better handles intermittent demand and outliers.
    """
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        """
        Args:
            preds: [batch, output_dim, 3] - (q10, q50, q90)
            target: [batch, output_dim]
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, i]
            losses.append(torch.max((q - 1) * errors, q * errors))
        return torch.mean(torch.sum(torch.stack(losses, dim=-1), dim=-1))


# ===== IMPROVED METRICS =====
def compute_metrics(preds, targets, quantile_mode=False):
    """
    Computes multiple retail-relevant metrics.
    
    Args:
        preds: [N, forecast_steps] or [N, forecast_steps, 3] if quantile
        targets: [N, forecast_steps]
    """
    if quantile_mode:
        preds = preds[:, :, 1]  # Use median (q50)
    
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    
    # MAE
    mae = torch.abs(preds_flat - targets_flat).mean().item()
    
    # RMSE
    rmse = torch.sqrt(((preds_flat - targets_flat) ** 2).mean()).item()
    
    # MAPE (avoid division by zero)
    mask = targets_flat != 0
    if mask.sum() > 0:
        mape = (torch.abs((targets_flat[mask] - preds_flat[mask]) / targets_flat[mask])).mean().item() * 100
    else:
        mape = 0.0
    
    # Bias (under/over prediction)
    bias = (preds_flat - targets_flat).mean().item()
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'bias': bias
    }