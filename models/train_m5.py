# train_m5.py
import os
import pickle
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .transformer_baseline_m5 import ForecastingModel

def normalize_data(data, name="feature"):
    mean, std = float(np.mean(data)), float(np.std(data))
    std = 1.0 if std == 0 else std
    print(f"[DEBUG] Normalized {name}: mean={mean:.4f}, std={std:.4f}")
    return (data - mean) / std, mean, std

def prepare_data(sales, price, store, item, FEATURE_LAG, forecast_steps):
    sales, price, store, item = map(np.asarray, (sales, price, store, item))
    num_samples = len(sales) - FEATURE_LAG - forecast_steps + 1
    X_seq, X_static, Y = [], [], []
    for i in range(num_samples):
        seq = np.stack([sales[i:i+FEATURE_LAG], price[i:i+FEATURE_LAG]], axis=1)
        X_seq.append(seq)
        X_static.append([store[i], item[i]])
        Y.append(sales[i+FEATURE_LAG:i+FEATURE_LAG+forecast_steps])
    return np.array(X_seq), np.array(X_static), np.array(Y)

def train_loop(model, dataloader, criterion, optimizer, scheduler, epochs, device):
    model.train()
    for epoch in range(epochs):
        losses = []
        for x_seq, x_static, y in dataloader:
            x_seq, x_static, y = x_seq.to(device), x_static.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x_seq, x_static)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        scheduler.step(avg_loss)
        print(f"[INFO] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

def main(config_path, csv_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(csv_path)
    print("[INFO] Loaded M5 dataset:", df.shape)

    sales = df["sales"].values
    price = df["sell_price"].values
    store = df["store_id_encoded"].values
    item = df["item_id_encoded"].values

    sales_norm, sales_mean, sales_std = normalize_data(sales, "sales")
    price_norm, price_mean, price_std = normalize_data(price, "sell_price")
    store_norm, store_mean, store_std = normalize_data(store, "store_id_encoded")
    item_norm, item_mean, item_std = normalize_data(item, "item_id_encoded")

    X_seq, X_static, Y = prepare_data(sales_norm, price_norm, store_norm, item_norm,
                                      config["FEATURE_LAG"], config["FORECAST_STEPS"])
    print(f"[INFO] Training samples: {X_seq.shape[0]} | Sequence shape: {X_seq.shape[1:]}")

    dataset = TensorDataset(torch.tensor(X_seq, dtype=torch.float32),
                            torch.tensor(X_static, dtype=torch.float32),
                            torch.tensor(Y, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=config["BATCH_SIZE"], shuffle=True)

    model = ForecastingModel(**config["MODEL_PARAMS"]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_loop(model, dataloader, criterion, optimizer, scheduler, config["EPOCHS"], device)

    checkpoint_path = config.get("CHECKPOINT_PATH", "m5_forecast_model.pth")
    metadata_path = config.get("METADATA_PATH", "m5_model_metadata.pkl")
    torch.save(model.state_dict(), checkpoint_path)

    metadata = {
        "model_params": config["MODEL_PARAMS"],
        "normalization": {
            "sales_mean": sales_mean, "sales_std": sales_std,
            "price_mean": price_mean, "price_std": price_std,
            "store_mean": store_mean, "store_std": store_std,
            "item_mean": item_mean, "item_std": item_std
        },
        "checkpoint_path": checkpoint_path,
    }
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"[INFO] Saved model and metadata to {checkpoint_path}, {metadata_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    main(args.config, args.csv)
