# ---------------------------------------------------------
# predict_m5_fixed.py (Updated with normalization inverse + evaluation + user input)
# ---------------------------------------------------------

import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformer_baseline_m5_fixed import ForecastingModel


@torch.no_grad()
def predict_and_evaluate(config_dataset, config_model, store_id, item_id, forecast_steps):
    # Load dataset and normalization stats
    df = pd.read_csv(config_dataset["dataset"]["raw_data_path"], parse_dates=["week_num_global"])
    stats_path = os.path.join(config_model["CHECKPOINT_PATH"].rsplit("/", 1)[0], "m5_series_stats.csv")
    stats_df = pd.read_csv(stats_path)

    seq_len = config_model["FEATURE_LAG"]
    cont_features = ["sales", "sell_price"]

    # Model setup
    num_stores = df["store_id_encoded"].nunique()
    num_items = df["item_id_encoded"].nunique()
    model = ForecastingModel(
        input_dim_seq=len(cont_features),
        input_dim_static=2,
        embed_size=config_model["MODEL_PARAMS"]["embed_size"],
        nhead=config_model["MODEL_PARAMS"]["nhead"],
        num_layers=2,
        dim_feedforward=256,
        output_dim=forecast_steps,
        num_stores=num_stores,
        num_items=num_items,
    )

    model.load_state_dict(torch.load(config_model["CHECKPOINT_PATH"], map_location="cpu"))
    model.eval()

    # Select series
    g = df[(df["store_id_encoded"] == store_id) & (df["item_id_encoded"] == item_id)].sort_values("week_num_global")
    if len(g) < seq_len + forecast_steps:
        print("Not enough data for this series.")
        return

    seq = g[cont_features].values[-(seq_len + forecast_steps) : -forecast_steps]
    seq[:, 0] = np.log1p(seq[:, 0])
    x_seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    store_idx = torch.tensor([store_id], dtype=torch.long)
    item_idx = torch.tensor([item_id], dtype=torch.long)

    pred_norm = model(x_seq, store_idx, item_idx).squeeze(0).numpy()

    # Inverse normalization
    row = stats_df[(stats_df["store_id"] == store_id) & (stats_df["item_id"] == item_id)].iloc[0]
    mean, std = row["mean"], row["std"]
    pred_sales = np.expm1(pred_norm * std + mean)
    

    # Actual future values
    actual_future = g["sales"].values[-forecast_steps:]

    sales_log = np.log1p(actual_future)
    actual_norm = (sales_log - mean) / std

    rmse_norm = mean_squared_error(actual_norm, pred_norm)
    mae_norm = mean_absolute_error(actual_norm, pred_norm)
    rmse_log = mean_squared_error(sales_log, np.log1p(pred_sales))
    mae_log = mean_absolute_error(sales_log, np.log1p(pred_sales))

    print(f"Normalized-space RMSE: {rmse_norm:.4f}, MAE: {mae_norm:.4f}")
    print(f"Log-space RMSE: {rmse_log:.4f}, MAE: {mae_log:.4f}")

    print(f'actual_future_value: {actual_future}, pred_sales: {pred_sales}')
    print(f'actual_norm_value: {actual_norm}, pred_norm: {pred_norm}')

    # # Evaluate metrics
    # rmse = mean_squared_error(actual_future, pred_sales)
    # mae = mean_absolute_error(actual_future, pred_sales)
    # print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(g["week_num_global"].values[-(seq_len + forecast_steps) : -forecast_steps], np.expm1(seq[:, 0]), label="Past Sales", color="blue")
    plt.plot(g["week_num_global"].values[-forecast_steps:], pred_sales, label="Forecast", color="orange", marker="o")
    plt.plot(g["week_num_global"].values[-forecast_steps:], actual_future, label="Actual", color="green", linestyle="--", marker="x")
    plt.title(f"Store {store_id}, Item {item_id} — Forecast vs Actual")
    plt.xlabel("Week Num")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return pred_sales, actual_future, rmse, mae


if __name__ == "__main__":
    with open("config/config_m5.yaml", "r") as f1, open("config/model_config_m5.yaml", "r") as f2:
        config_dataset = yaml.safe_load(f1)
        config_model = yaml.safe_load(f2)

    # Interactive console inputs
    store_id = int(input("Enter store_id: "))
    item_id = int(input("Enter item_id: "))
    forecast_steps = int(input(f"Enter forecast length (default={config_model['FORECAST_STEPS']}): ") or config_model["FORECAST_STEPS"])

    predict_and_evaluate(config_dataset, config_model, store_id, item_id, forecast_steps)
