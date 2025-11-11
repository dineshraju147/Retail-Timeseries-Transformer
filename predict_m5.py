# predict_m5.py
"""
Original prediction script, upgraded to support:
- Multi-month autoregressive forecasting
- Combined plotting of past and future predictions
- Historical backtest metrics (MAE, RMSE, MAPE, BIAS)
- Correct handling of future covariates and event encodings
"""
import os
import yaml
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import timedelta
from typing import List, Tuple, Dict, Any

# --- Import from your project files ---
from models.transformer_baseline_m5 import ForecastingModel
from pre_processing.preprocessor import (
    create_sequences_daily,
    normalize_data
)

# --- MODIFICATION: Import metrics function from the enhanced model file ---
from models.transformer_enhanced_m5 import compute_metrics


def denorm(x, mean, std):
    """Denormalizes data."""
    return x * (std + 1e-8) + mean


# --- (autoregressive_forecast function is unchanged from last time) ---
def autoregressive_forecast(
        model,
        initial_seq: torch.Tensor,
        initial_static: torch.Tensor,
        future_template_tensor: torch.Tensor,
        seq_cols: List[str],
        norm_meta: Dict[str, Any],
        forecast_months: int,
        forecast_steps: int,
        feature_lag: int,
        device
) -> List[np.ndarray]:
    """
    Generates multi-month forecast using autoregressive iteration, correctly
    handling known future covariates and re-calculating derived lag features.
    """
    model.eval()
    all_forecasts = []

    # --- 1. Initialization ---
    sales_idx = seq_cols.index('sales')
    derived_indices = {
        col: seq_cols.index(col)
        for col in seq_cols
        if 'lag_' in col or 'rolling_' in col
    }
    sales_mean = norm_meta['sales']['mean']
    sales_std = norm_meta['sales']['std']

    initial_sales_norm = initial_seq[0, :, sales_idx].cpu().numpy()
    denorm_sales_history = denorm(initial_sales_norm, sales_mean, sales_std).tolist()

    current_seq = initial_seq.clone()
    total_iterations = (forecast_months * 30) // forecast_steps

    with torch.no_grad():
        for i in range(total_iterations):

            # --- 2. Predict next window ---
            output = model(current_seq, initial_static)
            pred_norm = output  # Shape [1, forecast_steps]
            all_forecasts.append(pred_norm.cpu().numpy())

            pred_denorm_flat = denorm(
                pred_norm.cpu().numpy().flatten(),
                sales_mean,
                sales_std
            )

            new_seq_segment = torch.zeros(
                1, forecast_steps, current_seq.size(2)
            ).to(device)

            # --- 3. Build the new sequence day-by-day ---
            for j in range(forecast_steps):
                day_to_fill_idx = i * forecast_steps + j

                new_sale_denorm = pred_denorm_flat[j]
                denorm_sales_history.append(new_sale_denorm)
                hist_len = len(denorm_sales_history)

                new_day_slice = future_template_tensor[0, day_to_fill_idx, :].clone()
                new_day_slice[sales_idx] = pred_norm[0, j]

                for col_name, col_idx in derived_indices.items():
                    col_meta = norm_meta[col_name]
                    value_denorm = 0.0

                    if 'lag_' in col_name:
                        lag_num = int(col_name.split('_')[-1])
                        if hist_len >= lag_num:
                            value_denorm = denorm_sales_history[hist_len - lag_num]

                    elif 'rolling_' in col_name:
                        window = int(col_name.split('_')[-1])
                        if hist_len >= window:
                            value_denorm = np.mean(
                                denorm_sales_history[hist_len - window: hist_len]
                            )
                        else:
                            value_denorm = np.mean(denorm_sales_history[0: hist_len])

                    value_norm = (value_denorm - col_meta['mean']) / (col_meta['std'] + 1e-8)
                    new_day_slice[col_idx] = value_norm

                new_seq_segment[0, j, :] = new_day_slice

            # --- 4. Update sequence for next iteration ---
            full_seq = torch.cat([current_seq, new_seq_segment], dim=1)
            current_seq = full_seq[:, -feature_lag:, :]

    return all_forecasts


# --- (plot_forecast function is unchanged) ---
def plot_forecast(
        historical_sales,
        historical_dates,
        historical_preds,
        future_forecast,
        future_dates,
        actual_future=None,
        actual_future_dates=None,
        store_id=None,
        item_id=None,
        save_path=None
):
    """
    Creates enhanced visualization (like the enhanced script).
    """
    plt.figure(figsize=(24, 10))

    # 1. Plot historical actuals
    plt.plot(
        historical_dates,
        historical_sales,
        label='Historical Actual Sales',
        color='blue',
        linewidth=2,
        alpha=0.8
    )

    # 2. Plot historical predictions (sample every N for clarity)
    feature_lag = 28  # Assuming 28 from config
    sample_rate = max(1, len(historical_preds) // 50)
    for i in range(0, len(historical_preds), sample_rate):
        start_idx = i + feature_lag
        end_idx = start_idx + len(historical_preds[i])

        if end_idx <= len(historical_dates):
            plt.plot(
                historical_dates[start_idx:end_idx],
                historical_preds[i],
                color='red',
                linewidth=2,
                alpha=0.7
            )

    # Add legend entry for historical preds
    plt.plot([], [], color='red', linestyle='--', label='Historical Predictions', linewidth=2)

    # 3. Plot future forecast
    plt.plot(
        future_dates,
        future_forecast,
        label=f'{len(future_dates)}-Day Forecast',
        color='green',
        linestyle='--',
        linewidth=2.5,
        marker='o',
        markersize=3
    )

    # 4. Plot actual future data if available
    if actual_future is not None and actual_future_dates is not None:
        plt.plot(
            actual_future_dates,
            actual_future,
            label='Actual Future Sales',
            color='purple',
            linewidth=2,
            alpha=0.8
        )

    # Formatting
    last_history_date = historical_dates[-1]
    plt.axvline(
        last_history_date,
        color='gray',
        linestyle=':',
        linewidth=2,
        label='Forecast Start'
    )

    title = f"M5 Multi-Month Forecast (Original Model)"
    if store_id is not None and item_id is not None:
        title += f" (Store: {store_id}, Item: {item_id})"

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Sales', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\n✓ Plot saved to {save_path}")
    else:
        plt.show()


# --- (main function has critical fixes) ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='config/model_config_m5.yaml', help='Path to model config file')
    parser.add_argument('--store_id_encoded', type=int, default=0, help='Encoded store ID to predict (e.g., 0-9)')
    parser.add_argument('--item_id_encoded', type=int, default=10, help='Encoded item ID to predict (e.g., 0-3048)')
    parser.add_argument('--forecast_months', type=int, default=12,
                        help='Number of months to forecast into future')
    args = parser.parse_args()

    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)

    metadata_path = model_config["METADATA_PATH"]
    checkpoint_path = model_config["CHECKPOINT_PATH"]

    print(f"Loading metadata from {metadata_path}...")
    try:
        with open(metadata_path, "rb") as f:
            meta = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    norm_meta = meta['normalization']
    train_config = meta['data_config']

    # --- Load Model ---
    device = torch.device("cpu")
    model_params = meta["model_params"]
    model_params["device"] = "cpu"
    model = ForecastingModel(**model_params).to(device)

    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        return
    model.eval()
    print("[INFO] Model loaded successfully.")

    # --- Load ONE series from processed data ---
    processed_path = train_config['output']['processed_data_path']
    if not os.path.exists(processed_path):
        print(f"Error: Processed data file not found at {processed_path}")
        return

    print(f"Loading full processed data from {processed_path} to find series...")
    df = pd.read_pickle(processed_path)

    print(f"Filtering for store_id_encoded={args.store_id_encoded}, item_id_encoded={args.item_id_encoded}...")
    one_series_df = df[
        (df['store_id_encoded'] == args.store_id_encoded) &
        (df['item_id_encoded'] == args.item_id_encoded)
        ].copy()

    if one_series_df.empty:
        print(f"Error: No data found for this combination.")
        return

    # --- MODIFIED: Prepare data for forecasting ---

    # Get config values
    seq_cols = model_config['SEQUENTIAL_FEATURES']
    static_cols = model_config['STATIC_FEATURES']
    target_col = model_config['TARGET_FEATURE_COLUMN']
    feature_lag = model_config['FEATURE_LAG']
    forecast_steps = model_config['FORECAST_STEPS']
    norm_cols = model_config['SEQUENTIAL_FEATURES']

    derived_indices = {
        col: i
        for i, col in enumerate(seq_cols)
        if 'lag_' in col or 'rolling_' in col
    }

    # --- MODIFIED: Create a 'future' dataframe ---
    total_forecast_days = (args.forecast_months * 30)
    last_history_date = one_series_df['date'].max()
    last_known_price = one_series_df.iloc[-1]['sell_price']

    # --- START: FIX for Event Encoding ---
    # Load raw calendar
    try:
        calendar_path = meta['data_config']['dataset']['calendar_path']
        calendar_df = pd.read_csv(calendar_path)
        calendar_df['date'] = pd.to_datetime(calendar_df['date'])
    except FileNotFoundError:
        print(f"Error: Could not find calendar.csv at {calendar_path}")
        return

    # Get the event->encoded mapping from the *processed* df
    event_cols_encoded = [
        'event_name_1_encoded', 'event_type_1_encoded',
        'event_name_2_encoded', 'event_type_2_encoded'
    ]

    # Check if encoded columns exist in the processed file
    if all(col in df.columns for col in event_cols_encoded):
        # Grab the date-to-encoded mapping from the processed file
        processed_events_df = df[
            ['date'] + event_cols_encoded
            ].drop_duplicates()

        # Merge the *correct* encoded values into the raw calendar_df
        calendar_df = pd.merge(
            calendar_df,
            processed_events_df,
            on='date',
            how='left'
        )
    else:
        print("Error: Processed data file is missing required event_..._encoded columns.")
        # Fallback: create zero columns
        for col in event_cols_encoded:
            if col not in calendar_df.columns:
                calendar_df[col] = 0
    # --- END: FIX for Event Encoding ---

    # Create future dates
    future_dates_full_range = pd.date_range(
        start=last_history_date + pd.Timedelta(days=1),
        periods=total_forecast_days
    )

    df_future_template = pd.DataFrame({'date': future_dates_full_range})
    df_future_template['date'] = pd.to_datetime(df_future_template['date'])

    # Merge calendar features. This will now bring 'event_name_1_encoded' etc.
    df_future_template = pd.merge(
        df_future_template,
        calendar_df,
        on='date',
        how='left'
    )

    # --- Fill in missing/assumed future features ---

    # 1. Assume price stays constant
    df_future_template['sell_price'] = last_known_price

    # 2. Fill in static features
    for col in static_cols:
        df_future_template[col] = one_series_df[col].iloc[0]

    # 3. Fill 'snap'
    df_future_template['snap'] = 0  # Simplification

    # 4. Fill date features
    df_future_template['day_of_week'] = df_future_template['date'].dt.dayofweek
    df_future_template['month'] = df_future_template['date'].dt.month
    df_future_template['year'] = df_future_template['date'].dt.year

    # 5. Fill event features (handle NaNs for future dates with no events)
    for col in event_cols_encoded:
        if col in df_future_template.columns:
            df_future_template[col] = df_future_template[col].fillna(0)
        else:
            print(f"Warning: Column '{col}' missing after merge. Filling with 0.")
            df_future_template[col] = 0

    # 6. Fill unknown/derived features with 0
    df_future_template['sales'] = 0.0
    for col in derived_indices.keys():
        df_future_template[col] = 0.0

    # Ensure column order matches
    all_req_cols = seq_cols + static_cols
    df_future_template = df_future_template[
        [col for col in all_req_cols if col in df_future_template.columns]
    ]

    # --- Normalize this series (History) ---
    one_series_df_norm, _ = normalize_data(one_series_df, norm_cols, norm_meta)

    # --- Normalize the future template ---
    # --- FIX for TypeError ---
    # Removed the 'fit=False' argument
    df_future_template_norm, _ = normalize_data(
        df_future_template,
        norm_cols,
        norm_meta
    )

    # --- Create historical sequences ---
    X_seq, X_static, Y_norm = create_sequences_daily(
        one_series_df_norm,
        seq_cols=seq_cols,
        static_cols=static_cols,
        target_col=target_col,
        feature_lag=feature_lag,
        forecast_steps=forecast_steps
    )

    if X_seq.shape[0] == 0:
        print("Error: This series is too short to create any samples.")
        return

    # --- A. Run Inference on HISTORICAL data ---
    X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    X_static_tensor = torch.tensor(X_static, dtype=torch.float32).to(device)

    with torch.no_grad():
        hist_preds_norm = model(X_seq_tensor, X_static_tensor).cpu().numpy()

    # --- Metrics calculation block ---
    print("\n[INFO] Calculating metrics for historical predictions (backtest)...")
    hist_preds_tensor = torch.tensor(hist_preds_norm)
    y_actual_tensor = torch.tensor(Y_norm)

    hist_metrics = compute_metrics(
        hist_preds_tensor,
        y_actual_tensor,
        quantile_mode=False
    )

    print(f"  Historical MAE:   {hist_metrics['mae']:.4f}")
    print(f"  Historical RMSE:  {hist_metrics['rmse']:.4f}")
    print(f"  Historical MAPE:  {hist_metrics['mape']:.2f}%")
    print(f"  Historical BIAS:  {hist_metrics['bias']:.4f}")

    # --- B. Denormalize historical predictions ---
    sales_mean = norm_meta['sales']['mean']
    sales_std = norm_meta['sales']['std']
    hist_preds = denorm(hist_preds_norm, sales_mean, sales_std)
    original_sales = (one_series_df['sales'] * sales_std) + sales_mean
    all_dates = one_series_df['date'].values

    # --- C. Run AUTOREGRESSIVE forecast for FUTURE (MODIFIED) ---
    print(f"\n[INFO] Generating {args.forecast_months}-month forecast...")

    initial_seq_tensor = torch.tensor(
        X_seq[-1:, :, :], dtype=torch.float32
    ).to(device)
    initial_static_tensor = torch.tensor(
        X_static[-1:, :], dtype=torch.float32
    ).to(device)

    future_template_tensor = torch.tensor(
        df_future_template_norm[seq_cols].values, dtype=torch.float32
    ).unsqueeze(0).to(device)

    all_future_forecasts = autoregressive_forecast(
        model,
        initial_seq_tensor,
        initial_static_tensor,
        future_template_tensor,
        seq_cols,
        norm_meta,
        args.forecast_months,
        forecast_steps,
        feature_lag,
        device
    )

    future_forecast_flat_norm = np.concatenate([f.flatten() for f in all_future_forecasts])
    future_forecast_flat_norm = future_forecast_flat_norm[:total_forecast_days]
    future_forecast_denorm = denorm(future_forecast_flat_norm, sales_mean, sales_std)

    future_dates = pd.date_range(
        start=last_history_date + pd.Timedelta(days=1),
        periods=len(future_forecast_denorm)
    )

    # --- Plot ---
    print("[INFO] Generating combined historical and future plot...")

    plot_dir = model_config.get('PREDICT_OUTPUT_DIR', 'predictions_m5')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir,
                             f"plot_full_forecast_store{args.store_id_encoded}_item{args.item_id_encoded}.png")

    plot_forecast(
        historical_sales=original_sales.values,
        historical_dates=all_dates,
        historical_preds=hist_preds,
        future_forecast=future_forecast_denorm,
        future_dates=future_dates,
        actual_future=None,
        actual_future_dates=None,
        store_id=args.store_id_encoded,
        item_id=args.item_id_encoded,
        save_path=plot_path
    )

    print(f"\n[INFO] Forecast complete!")
    print(f"  Forecasted {len(future_forecast_denorm)} days ({len(future_forecast_denorm) / 30:.1f} months)")


if __name__ == "__main__":
    main()