# ---------------------------------------------------------
# train_m5_v3.py
# Adds preprocessing for raw M5 dataset, richer features,
# longer context window, direct log forecasting, and smaller transformer
# ---------------------------------------------------------

import os
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from transformer_baseline_m5_fixed import ForecastingModel


# ---------------------------------------------------------
# 1. Preprocessing Function
# ---------------------------------------------------------
def preprocess_m5_data(config_dataset):
    """
    Cleans and prepares raw M5-style data for training.
    Handles parsing, encoding, and missing value cleanup.
    """

    df = pd.read_csv(config_dataset["dataset"]["raw_data_path"], parse_dates=["date"])
    print(f"✅ Loaded raw data: {df.shape}")

    # Ensure consistent column naming
    required_cols = [
        "store_id_encoded",
        "item_id_encoded",
        "sales",
        "sell_price",
        "date",
        "week_of_year",
        "month",
        "snap_CA_encoded",
        "event_name_1_encoded",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Sort and drop duplicates if any
    df = df.sort_values(["store_id_encoded", "item_id_encoded", "date"]).drop_duplicates()

    # Fill NaN values in categorical/event columns
    for c in [
        "sell_price",
        "event_name_1_encoded",
        "snap_CA_encoded",
        "week_of_year",
        "month",
    ]:
        df[c] = df[c].fillna(0)

    # Convert encoded IDs to int
    df["store_id_encoded"] = df["store_id_encoded"].astype(int)
    df["item_id_encoded"] = df["item_id_encoded"].astype(int)

    # Filter date range (optional)
    if "dates" in config_dataset and config_dataset["dates"].get("start_date"):
        start = pd.to_datetime(config_dataset["dates"]["start_date"])
        end = pd.to_datetime(config_dataset["dates"]["end_date"])
        df = df[(df["date"] >= start) & (df["date"] <= end)]

    print(f"✅ Processed data: {df.shape}")
    return df


# ---------------------------------------------------------
# 2. Dataset Class
# ---------------------------------------------------------
class M5Dataset(Dataset):
    def __init__(self, df, seq_len, horizon, cont_features):
        self.seq_len = seq_len
        self.horizon = horizon
        self.cont_features = cont_features
        self.samples = []

        grouped = df.groupby(["store_id_encoded", "item_id_encoded"])
        for (store, item), g in grouped:
            g = g.sort_values("date")
            sales_log = np.log1p(g["sales"].values)
            cont_data = g[self.cont_features].fillna(0).values

            if len(g) <= seq_len + horizon:
                continue

            for i in range(len(g) - seq_len - horizon):
                seq = cont_data[i : i + seq_len]
                y = sales_log[i + seq_len : i + seq_len + horizon]
                self.samples.append(
                    {
                        "x_seq": torch.tensor(seq, dtype=torch.float32),
                        "store": torch.tensor(int(store), dtype=torch.long),
                        "item": torch.tensor(int(item), dtype=torch.long),
                        "y": torch.tensor(y, dtype=torch.float32),
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["x_seq"], s["store"], s["item"], s["y"]


# ---------------------------------------------------------
# 3. Training Function
# ---------------------------------------------------------
def train_m5(config_dataset, config_model):
    torch.manual_seed(config_model["SEED"])
    np.random.seed(config_model["SEED"])

    # Load and preprocess data
    df = preprocess_m5_data(config_dataset)

    seq_len = config_model["FEATURE_LAG"]
    horizon = config_model["FORECAST_STEPS"]

    cont_features = [
        "sales",
        "sell_price",
        "week_of_year",
        "month",
        "snap_CA_encoded",
        "event_name_1_encoded",
    ]

    dataset = M5Dataset(df, seq_len, horizon, cont_features)
    loader = DataLoader(dataset, batch_size=config_model["BATCH_SIZE"], shuffle=True)
    print(f"✅ Training samples: {len(dataset)}")

    num_stores = df["store_id_encoded"].nunique()
    num_items = df["item_id_encoded"].nunique()

    model = ForecastingModel(
        input_dim_seq=len(cont_features),
        input_dim_static=2,
        embed_size=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        output_dim=horizon,
        num_stores=num_stores,
        num_items=num_items,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config_model["LEARNING_RATE"])
    criterion = nn.MSELoss()

    print("🚀 Starting training...")
    for epoch in range(config_model["EPOCHS"]):
        model.train()
        total_loss = 0
        for x_seq, store, item, y in loader:
            x_seq, store, item, y = x_seq.to(device), store.to(device), item.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x_seq, store, item)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{config_model['EPOCHS']}], Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(config_model["CHECKPOINT_PATH"]), exist_ok=True)
    torch.save(model.state_dict(), config_model["CHECKPOINT_PATH"])
    print(f"✅ Model saved to {config_model['CHECKPOINT_PATH']}")


if __name__ == "__main__":
    with open("config/config_m5.yaml", "r") as f1, open("config/model_config_m5.yaml", "r") as f2:
        config_dataset = yaml.safe_load(f1)
        config_model = yaml.safe_load(f2)

    config_model["FEATURE_LAG"] = 16
    config_model["FORECAST_STEPS"] = 4
    config_model["MODEL_PARAMS"]["embed_size"] = 32
    config_model["MODEL_PARAMS"]["nhead"] = 4
    config_model["MODEL_PARAMS"]["num_layers"] = 2
    config_model["MODEL_PARAMS"]["dim_feedforward"] = 128

    train_m5(config_dataset, config_model)
