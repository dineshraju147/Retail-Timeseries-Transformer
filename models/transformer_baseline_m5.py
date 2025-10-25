# ---------------------------------------------------------
# transformer_baseline_m5_fixed.py
# Updated Transformer model for M5 Forecasting
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


class ForecastingModel(nn.Module):
    """
    Transformer model for M5 forecasting with:
      - mean pooling over time (no flattening)
      - store/item categorical embeddings
      - optional static continuous input
      - multi-step forecasting (output_dim = FORECAST_STEPS)
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

        # Project sequence features to embedding
        self.input_projection = nn.Linear(input_dim_seq, embed_size)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embed_size, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Embeddings for static categorical features
        self.store_emb = nn.Embedding(num_stores, store_emb_dim)
        self.item_emb = nn.Embedding(num_items, item_emb_dim)

        # Regression head with mean pooling
        mlp_input = embed_size + store_emb_dim + item_emb_dim
        self.regressor = nn.Sequential(
            nn.Linear(mlp_input, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x_seq, store_idx, item_idx):
        # Sequence encoding
        x = self.input_projection(x_seq)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        # Mean pooling
        x_pooled = x.mean(dim=1)

        # Static embeddings
        s = self.store_emb(store_idx)
        i = self.item_emb(item_idx)

        x_cat = torch.cat((x_pooled, s, i), dim=1)
        out = self.regressor(x_cat)
        return out
