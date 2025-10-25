# ---------------------------------------------------------
# predict_m5_v4.py
# Predicts for 3 random (store,item) pairs and visualizes
# past vs forecast vs actual in line plots.
# ---------------------------------------------------------

import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformer_baseline_m5_fixed import ForecastingModel
from train_m5_v3 import preprocess_m5_data


@torch.no_grad()
def predict_and_visualize_random(config_dataset, config_model, forecast_steps=4, num_samples=3):
    # Step 1: Load & preprocess dataset
    df = preprocess_m5_data(config_dataset)
    # Normalize column names: lowercase and underscores only
    df.columns = [c.strip().lower() for c in df.columns]

    cont_features = [
        "sales",
        "sell_price",
        "week_of_year",
        "month",
        "snap_ca_encoded",
        "event_name_1_encoded",
    ]

    seq_len = config_model["FEATURE_LAG"]

    # Step 2: Load trained model
    num_stores = df["store_id_encoded"].nunique()
    num_items = df["item_id_encoded"].nunique()

    model = ForecastingModel(
        input_dim_seq=len(cont_features),
        input_dim_static=2,
        embed_size=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        output_dim=forecast_steps,
        num_stores=num_stores,
        num_items=num_items,
    )

    model.load_state_dict(torch.load(config_model["CHECKPOINT_PATH"], map_location="cpu"))
    model.eval()

    # Step 3: Choose random store-item pairs
    unique_pairs = df[["store_id_encoded", "item_id_encoded"]].drop_duplicates().sample(num_samples, random_state=42)
    print(f"🎯 Selected {num_samples} random (store, item) pairs for visualization:")
    print(unique_pairs)

    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    # Step 4: Predict for each series
    for idx, row in enumerate(unique_pairs.itertuples(index=False)):
        store_id, item_id = int(row.store_id_encoded), int(row.item_id_encoded)
        g = df[(df["store_id_encoded"] == store_id) & (df["item_id_encoded"] == item_id)].sort_values("date")

        if len(g) < seq_len + forecast_steps:
            print(f"⚠️ Skipping (store={store_id}, item={item_id}) — not enough history.")
            continue

        # Prepare input
        seq = g[cont_features].values[-(seq_len + forecast_steps) : -forecast_steps]
        seq[:, 0] = np.log1p(seq[:, 0])  # log-transform sales feature
        x_seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        store_idx = torch.tensor([store_id], dtype=torch.long)
        item_idx = torch.tensor([item_id], dtype=torch.long)

        # Predict
        pred_log = model(x_seq, store_idx, item_idx).squeeze(0).numpy()
        pred_sales = np.expm1(pred_log)

        # Actual values
        actual_sales = g["sales"].values[-forecast_steps:]
        actual_log = np.log1p(actual_sales)

        # Metrics
        rmse_log = mean_squared_error(actual_log, pred_log)
        mae_log = mean_absolute_error(actual_log, pred_log)
        print(f"✅ (store={store_id}, item={item_id}) → Log-space RMSE={rmse_log:.4f}, MAE={mae_log:.4f}")

        # Plot
        ax = axes[idx]
        past_dates = g["date"].values[-(seq_len + forecast_steps) : -forecast_steps]
        future_dates = g["date"].values[-forecast_steps:]

        ax.plot(past_dates, np.expm1(seq[:, 0]), label="Past Sales", color="blue")
        ax.plot(future_dates, pred_sales, label="Forecast (Pred)", color="orange", marker="o")
        ax.plot(future_dates, actual_sales, label="Actual", color="green", linestyle="--", marker="x")

        ax.set_title(f"Store {store_id}, Item {item_id} — {forecast_steps}-week Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    with open("config/config_m5.yaml", "r") as f1, open("config/model_config_m5.yaml", "r") as f2:
        config_dataset = yaml.safe_load(f1)
        config_model = yaml.safe_load(f2)

    forecast_steps = int(input(f"Enter forecast length (default={config_model['FORECAST_STEPS']}): ") or config_model["FORECAST_STEPS"])
    predict_and_visualize_random(config_dataset, config_model, forecast_steps=forecast_steps, num_samples=3)
