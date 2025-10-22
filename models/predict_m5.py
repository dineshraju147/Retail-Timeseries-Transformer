# predict_m5.py
import os, yaml, pickle, argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from models.transformer_baseline_m5 import ForecastingModel

def denorm(x, mean, std):
    return x * std + mean

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

def main(config_path, csv_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(config["METADATA_PATH"], "rb") as f:
        meta = pickle.load(f)

    # --- Safe device loading ---
    device = torch.device("cpu")
    model = ForecastingModel(**meta["model_params"])
    model.load_state_dict(torch.load(meta["checkpoint_path"], map_location=device))
    model.to(device)
    model.eval()

    # --- Load and normalize data ---
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df):,} rows")

    sales, price, store, item = df["sales"], df["sell_price"], df["store_id_encoded"], df["item_id_encoded"]

    sales_norm = (sales - meta["normalization"]["sales_mean"]) / meta["normalization"]["sales_std"]
    price_norm = (price - meta["normalization"]["price_mean"]) / meta["normalization"]["price_std"]
    store_norm = (store - meta["normalization"]["store_mean"]) / meta["normalization"]["store_std"]
    item_norm = (item - meta["normalization"]["item_mean"]) / meta["normalization"]["item_std"]

    X_seq, X_static, Y = prepare_data(
        sales_norm, price_norm, store_norm, item_norm,
        config["FEATURE_LAG"], config["FORECAST_STEPS"]
    )

    print(f"[INFO] Prepared {len(X_seq):,} sequences for prediction")

    # --- Memory-safe batched inference ---
    preds = []
    batch_size = 128  # adjust based on your system
    with torch.no_grad():
        for i in range(0, len(X_seq), batch_size):
            X_seq_batch = torch.tensor(X_seq[i:i+batch_size], dtype=torch.float32).to(device)
            X_static_batch = torch.tensor(X_static[i:i+batch_size], dtype=torch.float32).to(device)
            batch_preds = model(X_seq_batch, X_static_batch).detach().cpu().numpy()
            preds.append(batch_preds)
            if i % (batch_size * 100) == 0:
                print(f"[INFO] Processed {i:,}/{len(X_seq):,} samples")

    preds = np.concatenate(preds, axis=0)

    # --- Denormalize and evaluate ---
    preds = denorm(preds, meta["normalization"]["sales_mean"], meta["normalization"]["sales_std"])
    Y = denorm(Y, meta["normalization"]["sales_mean"], meta["normalization"]["sales_std"])

    print("\n[INFO] Evaluation metrics:")
    print("MAE:", mean_absolute_error(Y.flatten(), preds.flatten()))
    print("RMSE:", np.sqrt(mean_squared_error(Y.flatten(), preds.flatten())))
    print("R²:", r2_score(Y.flatten(), preds.flatten()))

    # --- Plot a small sample ---
    plt.figure(figsize=(10, 4))
    plt.plot(Y.flatten()[:200], label="Actual")
    plt.plot(preds.flatten()[:200], label="Predicted")
    plt.legend()
    plt.title("M5 Sales Forecasting (Transformer)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    main(args.config, args.csv)
