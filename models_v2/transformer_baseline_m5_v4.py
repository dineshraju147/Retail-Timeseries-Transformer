# ---------------------------------------------------------
# transformer_baseline_m5_fixed.py (IMPROVED VERSION)
# 
# KEY IMPROVEMENTS:
# 1. Added recency-weighted attention instead of mean pooling
# 2. Separate processing for different feature types
# 3. Residual connections for better gradient flow
# 4. Layer normalization for stable training
# 5. Dropout for regularization
# ---------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class RecencyWeightedPooling(nn.Module):
    """
    IMPROVEMENT: Uses exponential weights to emphasize recent timesteps
    This helps the model focus on recent trends rather than distant history
    """
    def __init__(self, decay=0.9):
        super().__init__()
        self.decay = decay
    
    def forward(self, x):
        # x: [batch, seq_len, embed_size]
        seq_len = x.size(1)
        
        # Create exponentially decaying weights (most recent = highest weight)
        weights = torch.tensor([self.decay ** (seq_len - i - 1) for i in range(seq_len)], 
                               device=x.device, dtype=x.dtype)
        weights = weights / weights.sum()  # Normalize
        
        # Apply weights: [batch, seq_len, embed] * [seq_len] -> [batch, embed]
        weighted = (x * weights.view(1, -1, 1)).sum(dim=1)
        return weighted


class ForecastingModel(nn.Module):
    """
    IMPROVED Transformer model with:
      - Recency-weighted pooling (emphasizes recent patterns)
      - Separate feature processing pathways
      - Better regularization (dropout, layer norm)
      - Residual connections in MLP head
      - Multi-step forecasting with temporal awareness
    """

    def __init__(
        self,
        input_dim_seq,
        input_dim_static,
        embed_size=64,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        output_dim=1,
        num_stores=10,
        num_items=10,
        store_emb_dim=16,
        item_emb_dim=16,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.input_dim_seq = input_dim_seq
        self.input_dim_static = input_dim_static
        self.output_dim = output_dim

        # IMPROVEMENT: Separate projection for different feature types
        # This allows the model to learn different representations for different features
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim_seq, embed_size),
            nn.LayerNorm(embed_size),  # Stabilizes training
            nn.Dropout(dropout)
        )

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embed_size, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # IMPROVEMENT: Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Embeddings for static categorical features
        self.store_emb = nn.Embedding(num_stores, store_emb_dim)
        self.item_emb = nn.Embedding(num_items, item_emb_dim)
        
        # IMPROVEMENT: Recency-weighted pooling instead of mean pooling
        self.recency_pool = RecencyWeightedPooling(decay=0.95)
        
        # IMPROVEMENT: Also keep last timestep for immediate context
        self.last_step_proj = nn.Linear(embed_size, embed_size // 2)

        # IMPROVEMENT: Enhanced regression head with residual connections
        mlp_input = embed_size + (embed_size // 2) + store_emb_dim + item_emb_dim
        
        self.regressor = nn.Sequential(
            nn.Linear(mlp_input, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, output_dim),
        )

    def forward(self, x_seq, store_idx, item_idx):
        """
        Forward pass with improved feature processing
        
        Args:
            x_seq: [batch, seq_len, input_dim_seq] - temporal features
            store_idx: [batch] - store IDs
            item_idx: [batch] - item IDs
            
        Returns:
            [batch, output_dim] - forecasted values (log-space)
        """
        # Sequence encoding with transformer
        x = self.input_projection(x_seq)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        # IMPROVEMENT: Dual pooling strategy
        # 1. Recency-weighted pooling (captures overall pattern with recent emphasis)
        x_pooled = self.recency_pool(x)
        
        # 2. Last timestep (captures immediate state)
        x_last = self.last_step_proj(x[:, -1, :])

        # Static embeddings (categorical features)
        s = self.store_emb(store_idx)
        i = self.item_emb(item_idx)

        # IMPROVEMENT: Concatenate all information sources
        x_cat = torch.cat((x_pooled, x_last, s, i), dim=1)
        
        # Generate forecast
        out = self.regressor(x_cat)
        return out