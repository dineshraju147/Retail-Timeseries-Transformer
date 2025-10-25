# ---------------------------------------------------------
# train_m5_fixed.py (Updated with per-series normalization save)
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
# 1. Dataset Loader
# ---------------------------------------------------------
class M5Dataset(Dataset):
    def __init__(self, df, seq_len, horizon, cont_features):
        self.seq_len = seq_len
        self.horizon = horizon
        self.cont_features = cont_features
        self.samples = []
        self.series_stats = []

        grouped = df.groupby(["store_id_encoded", "item_id_encoded"])
        for (store, item), g in grouped:
            g = g.sort_values("week_num_global")
            sales = np.log1p(g["sales"].values)
            mean, std = sales.mean(), sales.std()
            std = 1.0 if std == 0 else std
            self.series_stats.append({"store_id": store, "item_id": item, "mean": mean, "std": std})

            cont_data = g[cont_features].values
            for i in range(len(g) - seq_len - horizon):
                seq = cont_data[i : i + seq_len]
                y = (sales[i + seq_len : i + seq_len + horizon] - mean) / std
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
# 2. Training Loop
# ---------------------------------------------------------
def train_m5(config_dataset, config_model):
    torch.manual_seed(config_model["SEED"])
    np.random.seed(config_model["SEED"])

    # Load dataset
    df = pd.read_csv(config_dataset["dataset"]["raw_data_path"], parse_dates=["week_num_global"])
    seq_len = config_model["FEATURE_LAG"]
    horizon = config_model["FORECAST_STEPS"]
    cont_features = ["sales", "sell_price"]

    dataset = M5Dataset(df, seq_len, horizon, cont_features)
    loader = DataLoader(dataset, batch_size=config_model["BATCH_SIZE"], shuffle=True)

    # Save per-series stats
    stats_path = os.path.join(config_model["CHECKPOINT_PATH"].rsplit("/", 1)[0], "m5_series_stats.csv")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    pd.DataFrame(dataset.series_stats).to_csv(stats_path, index=False)
    print(f"Saved per-series mean/std to {stats_path}")

    # Instantiate model
    num_stores = df["store_id_encoded"].nunique()
    num_items = df["item_id_encoded"].nunique()
    model = ForecastingModel(
        input_dim_seq=len(cont_features),
        input_dim_static=2,
        embed_size=config_model["MODEL_PARAMS"]["embed_size"],
        nhead=config_model["MODEL_PARAMS"]["nhead"],
        num_layers=2,
        dim_feedforward=256,
        output_dim=horizon,
        num_stores=num_stores,
        num_items=num_items,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config_model["LEARNING_RATE"])
    criterion = nn.MSELoss()

    # Training loop
    print("Starting training...")
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

    # Save model weights
    os.makedirs(os.path.dirname(config_model["CHECKPOINT_PATH"]), exist_ok=True)
    torch.save(model.state_dict(), config_model["CHECKPOINT_PATH"])
    print(f"Model saved to {config_model['CHECKPOINT_PATH']}")


# ---------------------------------------------------------
# 3. Main Entry
# ---------------------------------------------------------
if __name__ == "__main__":
    with open("config/config_m5.yaml", "r") as f1, open("config/model_config_m5.yaml", "r") as f2:
        config_dataset = yaml.safe_load(f1)
        config_model = yaml.safe_load(f2)
    train_m5(config_dataset, config_model)
