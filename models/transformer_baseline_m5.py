"""
transformer_baseline_m5.py

Defines the core Transformer-based architecture for M5 sales forecasting.

Classes:
    PositionalEncoding(nn.Module): Adds positional information to sequential input data.
    ForecastingModel(nn.Module): Transformer-based regression model for sales forecasting,
                                 using both dynamic (time-series) and static (store/item) features.

Usage:
    from transformer_baseline_m5 import ForecastingModel
    model = ForecastingModel(FEATURE_LAG=8, output_dim=1, input_dim_seq=2, input_dim_static=2, ...)
    y_pred = model(x_seq, x_static)

Notes:
    - Sequential features include: [sales, sell_price]
    - Static features include: [store_id_encoded, item_id_encoded]
    - Supports both Conv1D and Linear embedding for sequence input.
    - Position encoding is applied before Transformer encoder layers.
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
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): [seq_len, batch_size, embed_dim]
        Returns:
            Tensor: Same shape as input with positional encoding added.
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ForecastingModel(nn.Module):
    """
    Transformer-based regression model for M5 time series sales forecasting.

    Inputs:
        - x_seq: Sequential features [batch, seq_len, input_dim_seq]
                  (e.g., sales and sell_price)
        - x_static: Static features [batch, input_dim_static]
                    (e.g., store_id_encoded, item_id_encoded)
    Output:
        - y_pred: Forecasted sales [batch, output_dim]
    """

    def __init__(
        self,
        FEATURE_LAG=8,
        output_dim=1,
        input_dim_seq=2,
        input_dim_static=2,
        embed_size=128,
        nhead=8,
        num_layers=6,
        dim_feedforward=512,
        dropout=0.1,
        conv1d_emb=True,
        conv1d_kernel_size=3,
        device="cpu"
    ):
        super().__init__()
        self.device = device
        self.FEATURE_LAG = FEATURE_LAG
        self.embed_size = embed_size
        self.input_dim_seq = input_dim_seq
        self.input_dim_static = input_dim_static
        self.conv1d_emb = conv1d_emb
        self.conv1d_kernel_size = conv1d_kernel_size

        # Embedding Layer (Conv1D or Linear)
        if conv1d_emb:
            if conv1d_kernel_size % 2 == 0:
                raise ValueError("conv1d_kernel_size must be odd to preserve dimensions.")
            padding = conv1d_kernel_size // 2
            self.input_embedding = nn.Conv1d(
                in_channels=input_dim_seq,
                out_channels=embed_size,
                kernel_size=conv1d_kernel_size,
                padding=padding
            )
        else:
            self.input_embedding = nn.Linear(input_dim_seq, embed_size)

        # Positional Encoding
        self.position_encoder = PositionalEncoding(d_model=embed_size, dropout=dropout, max_len=FEATURE_LAG)

        # Transformer Encoder Layers
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regression Head
        input_for_regression = FEATURE_LAG * embed_size + input_dim_static
        hidden_sizes = [input_for_regression, dim_feedforward, dim_feedforward // 2, output_dim]
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

        # Embedding
        if self.conv1d_emb:
            x = self.input_embedding(x_seq.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.input_embedding(x_seq)

        # Add positional encoding
        x = self.position_encoder(x.transpose(0, 1)).transpose(0, 1)

        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Flatten and concatenate static features
        x = x.reshape(batch_size, -1)
        x = torch.cat((x, x_static), dim=1)

        # Regression head
        for layer in self.regression_layers[:-1]:
            x = self.dropout(self.relu(layer(x)))
        out = self.regression_layers[-1](x)

        return out
