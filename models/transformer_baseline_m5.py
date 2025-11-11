# transformer_baseline_m5.py
"""
transformer_baseline_m5.py

Defines the core Transformer-based architecture for M5 sales forecasting.

Classes:
    PositionalEncoding(nn.Module): Adds positional information to sequential input data.
    ForecastingModel(nn.Module): Transformer-based regression model for sales forecasting,
                                 using both dynamic (time-series) and static (store/item) features.
"""

import math
import torch
from torch import nn, Tensor
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder


class PositionalEncoding(nn.Module):
    """Adds sinusoidal positional encodings to the input sequence embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        Returns:
            Tensor: x with positional encodings added
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ForecastingModel(nn.Module):
    """
    Transformer-based regression model for sales forecasting.
    
    This model uses a Transformer encoder for sequential (dynamic) features 
    and concatenates static features before a final regression head.
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
        conv1d_emb: bool = True,
        conv1d_kernel_size: int = 3,
        device: str = "cpu"
    ):
        super(ForecastingModel, self).__init__()
        self.device = torch.device(device)
        self.feature_lag = FEATURE_LAG
        self.conv1d_emb = conv1d_emb
        
        # --- 1. Embedding Layers ---
        if conv1d_emb:
            # Use 1D Conv for embedding sequential features
            padding = (conv1d_kernel_size - 1) // 2
            self.input_embedding = nn.Conv1d(
                in_channels=input_dim_seq,
                out_channels=embed_size,
                kernel_size=conv1d_kernel_size,
                padding=padding
            )
        else:
            # Use Linear layer for embedding
            self.input_embedding = nn.Linear(input_dim_seq, embed_size)
            
        self.position_encoder = PositionalEncoding(embed_size, dropout)
        
        # --- 2. Transformer Encoder ---
        encoder_layers = TransformerEncoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Expects [batch, seq_len, embed_size]
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # --- 3. Regression Head ---
        # Flattened transformer output + static features
        total_input_size = (embed_size * FEATURE_LAG) + input_dim_static
        
        # Define MLP layers
        hidden_sizes = [total_input_size, total_input_size // 2, total_input_size // 4]
        
        # Add final output layer
        hidden_sizes.append(output_dim) 
        
        self.regression_layers = nn.ModuleList([
            nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            for i in range(len(hidden_sizes) - 1)
        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_seq: Tensor, x_static: Tensor, src_key_padding_mask=None) -> Tensor:
        """
        Forward pass for Transformer model.

        Args:
            x_seq: [batch, seq_len, input_dim_seq] - (sales, sell_price)
            x_static: [batch, input_dim_static] - (store_id_encoded, item_id_encoded)
            src_key_padding_mask: optional mask for padded sequences

        Returns:
            Tensor: Predicted sales [batch, output_dim]
        """
        batch_size = x_seq.size(0)

        # --- 1. Embedding & Positional Encoding ---
        if self.conv1d_emb:
            # [batch, seq_len, features] -> [batch, features, seq_len]
            x = self.input_embedding(x_seq.transpose(1, 2)).transpose(1, 2)
            # [batch, seq_len, embed_size]
        else:
            x = self.input_embedding(x_seq) # [batch, seq_len, embed_size]

        # Transpose for PositionalEncoding: [seq_len, batch, embed_size]
        x_pos = self.position_encoder(x.transpose(0, 1)).transpose(0, 1)
        # x_pos shape: [batch, seq_len, embed_size]

        # --- 2. Transformer Encoder ---
        # (batch_first=True)
        x = self.transformer_encoder(x_pos, src_key_padding_mask=src_key_padding_mask)
        # x shape: [batch, seq_len, embed_size]

        # --- 3. Regression Head ---
        # Flatten and concatenate static features
        x = x.reshape(batch_size, -1) # [batch, seq_len * embed_size]
        x = torch.cat([x, x_static], dim=1) # [batch, (seq_len * embed_size) + static_dim]
        
        # Pass through MLP
        for i, layer in enumerate(self.regression_layers):
            x = layer(x)
            if i < len(self.regression_layers) - 1: # No activation on final output layer
                x = self.relu(x)
                x = self.dropout(x)
                
        return x # Final shape: [batch, output_dim]