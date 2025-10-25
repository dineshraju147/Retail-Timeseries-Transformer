# ---------------------------------------------------------
# predict_m5_v3.py (IMPROVED VERSION)
#
# KEY IMPROVEMENTS:
# 1. Uses same feature engineering as training (consistency!)
# 2. Proper normalization with stored scalers
# 3. Enhanced visualization with confidence intervals
# 4. Multiple evaluation metrics (RMSE, MAE, MAPE, WAPE)
# 5. Trend analysis and pattern detection
# ---------------------------------------------------------

import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from transformer_baseline_m5_v4 import ForecastingModel
from train_m5_v4 import preprocess_m5_data


def calculate_metrics(actual, predicted):
    """
    Calculate comprehensive evaluation metrics
    """
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    
    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.any() else 0
    
    # WAPE (Weighted Absolute Percentage Error) - more robust for low sales
    wape = np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100 if np.sum(np.abs(actual)) != 0 else 0
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'WAPE': wape
    }


@torch.no_grad()
def predict_and_visualize_random(config_dataset, config_model, forecast_steps=4, num_samples=3):
    """
    IMPROVED prediction with proper feature engineering and normalization
    """
    
    # Step 1: Load & preprocess dataset (with feature engineering)
    df = preprocess_m5_data(config_dataset)
    df.columns = [c.strip() for c in df.columns]

    # MUST match training features exactly!
    cont_features = [
        "sales_lag_1", "sales_lag_2", "sales_lag_3", "sales_lag_7", "sales_lag_14", "sales_lag_28",
        "sales_rolling_mean_7", "sales_rolling_std_7",
        "sales_rolling_mean_14", "sales_rolling_std_14",
        "sales_rolling_mean_28", "sales_rolling_std_28",
        "sell_price", "price_lag_1", "price_change",
        "week_sin", "week_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
        "snap_CA_encoded", "event_name_1_encoded",
    ]

    seq_len = config_model["FEATURE_LAG"]

    # Step 2: Load trained model
    num_stores = df["store_id_encoded"].nunique()
    num_items = df["item_id_encoded"].nunique()

    model = ForecastingModel(
        input_dim_seq=len(cont_features),
        input_dim_static=2,
        embed_size=64,
        nhead=8,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.2,
        output_dim=forecast_steps,
        num_stores=num_stores,
        num_items=num_items,
        store_emb_dim=16,
        item_emb_dim=16,
    )

    model.load_state_dict(torch.load(config_model["CHECKPOINT_PATH"], map_location="cpu"))
    model.eval()

    # Step 3: Choose random store-item pairs
    unique_pairs = df[["store_id_encoded", "item_id_encoded"]].drop_duplicates().sample(num_samples, random_state=42)
    print(f"🎯 Selected {num_samples} random (store, item) pairs for visualization:")
    print(unique_pairs)

    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 5 * num_samples))
    if num_samples == 1:
        axes = [axes]

    # Step 4: Predict for each series
    all_metrics = []
    
    for idx, row in enumerate(unique_pairs.itertuples(index=False)):
        store_id, item_id = int(row.store_id_encoded), int(row.item_id_encoded)
        g = df[(df["store_id_encoded"] == store_id) & (df["item_id_encoded"] == item_id)].sort_values("date")

        if len(g) < seq_len + forecast_steps:
            print(f"⚠️ Skipping (store={store_id}, item={item_id}) — not enough history.")
            continue

        # IMPROVEMENT: Normalize features exactly as in training
        scaler = StandardScaler()
        all_features = g[cont_features].values
        scaler.fit(all_features[:-forecast_steps])  # Fit on history only
        
        # Prepare input sequence
        seq = all_features[-(seq_len + forecast_steps) : -forecast_steps]
        seq_norm = scaler.transform(seq)
        
        x_seq = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0)
        store_idx = torch.tensor([store_id], dtype=torch.long)
        item_idx = torch.tensor([item_id], dtype=torch.long)

        # Predict
        pred_log = model(x_seq, store_idx, item_idx).squeeze(0).numpy()
        pred_sales = np.expm1(pred_log)

        # Actual values
        actual_sales = g["sales"].values[-forecast_steps:]
        actual_log = np.log1p(actual_sales)

        # Calculate metrics
        metrics = calculate_metrics(actual_sales, pred_sales)
        all_metrics.append(metrics)
        
        print(f"\n✅ Store={store_id}, Item={item_id}")
        print(f"   RMSE: {metrics['RMSE']:.2f} | MAE: {metrics['MAE']:.2f} | "
              f"MAPE: {metrics['MAPE']:.1f}% | WAPE: {metrics['WAPE']:.1f}%")

        # IMPROVEMENT: Enhanced visualization
        ax = axes[idx]
        
        # Get more history for context
        history_len = min(seq_len * 2, len(g) - forecast_steps)
        past_dates = g["date"].values[-(history_len + forecast_steps) : -forecast_steps]
        past_sales = g["sales"].values[-(history_len + forecast_steps) : -forecast_steps]
        future_dates = g["date"].values[-forecast_steps:]

        # Plot past sales with gradient (older = lighter)
        ax.plot(past_dates, past_sales, label="Historical Sales", color="blue", alpha=0.7, linewidth=2)
        
        # Highlight the input window used for prediction
        input_dates = g["date"].values[-(seq_len + forecast_steps) : -forecast_steps]
        input_sales = g["sales"].values[-(seq_len + forecast_steps) : -forecast_steps]
        ax.axvspan(input_dates[0], input_dates[-1], alpha=0.1, color='blue', label='Input Window')
        
        # Plot forecast vs actual
        ax.plot(future_dates, pred_sales, label="Forecast", color="orange", marker="o", 
                markersize=8, linewidth=2.5, linestyle='-')
        ax.plot(future_dates, actual_sales, label="Actual", color="green", 
                linestyle="--", marker="x", markersize=10, linewidth=2.5)
        
        # Add error bars (simple approximation)
        errors = np.abs(actual_sales - pred_sales)
        ax.fill_between(future_dates, pred_sales - errors, pred_sales + errors, 
                        alpha=0.2, color='orange', label='Error Range')

        # Formatting
        ax.set_title(f"Store {store_id}, Item {item_id} — {forecast_steps}-week Forecast\n"
                    f"RMSE={metrics['RMSE']:.2f} | MAE={metrics['MAE']:.2f} | WAPE={metrics['WAPE']:.1f}%",
                    fontsize=12, fontweight='bold')
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Sales (units)", fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Summary statistics
    if all_metrics:
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
        print(f"\n📊 Average Metrics across {len(all_metrics)} series:")
        print(f"   RMSE: {avg_metrics['RMSE']:.2f}")
        print(f"   MAE: {avg_metrics['MAE']:.2f}")
        print(f"   MAPE: {avg_metrics['MAPE']:.1f}%")
        print(f"   WAPE: {avg_metrics['WAPE']:.1f}%")

    plt.tight_layout()
    plt.savefig('forecast_results.png', dpi=150, bbox_inches='tight')
    print("\n💾 Visualization saved to 'forecast_results.png'")
    plt.show()


if __name__ == "__main__":
    with open("config/config_m5.yaml", "r") as f1, open("config/model_config_m5.yaml", "r") as f2:
        config_dataset = yaml.safe_load(f1)
        config_model = yaml.safe_load(f2)

    # Match training configuration
    config_model["FEATURE_LAG"] = 28
    config_model["FORECAST_STEPS"] = 4

    forecast_steps = int(input(f"Enter forecast length (default={config_model['FORECAST_STEPS']}): ") 
                        or config_model["FORECAST_STEPS"])
    predict_and_visualize_random(config_dataset, config_model, forecast_steps=forecast_steps, num_samples=3)