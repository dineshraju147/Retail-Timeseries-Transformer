# predict_m5_enhanced.py
"""
Enhanced prediction script with:
- Correct multi-month forecasting via autoregressive iteration
- Actual future data comparison
- Confidence intervals from quantile predictions
- Plotting Q90 (90th percentile) as the main forecast
- **NEW: Plotting historical Q10-Q90 confidence bands**
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

from models.transformer_enhanced_m5 import EnhancedForecastingModel, compute_metrics
from pre_processing.preprocessor import create_sequences_daily, normalize_data


def denorm(x, mean, std):
    """Denormalizes data."""
    return x * (std + 1e-8) + mean


# --- (autoregressive_forecast function is unchanged) ---
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
        device,
        quantile_mode=False
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

            if quantile_mode:
                pred_norm_for_lags = output[:, :, 1]
                all_forecasts.append(output.cpu().numpy())
            else:
                pred_norm_for_lags = output
                all_forecasts.append(output.unsqueeze(-1).cpu().numpy())

            pred_denorm_flat = denorm(
                pred_norm_for_lags.cpu().numpy().flatten(),
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
                new_day_slice[sales_idx] = pred_norm_for_lags[0, j]

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
                            value_denorm = np.mean(denorm_sales_history[hist_len - window: hist_len])
                        else:
                            value_denorm = np.mean(denorm_sales_history[0: hist_len])
                    value_norm = (value_denorm - col_meta['mean']) / (col_meta['std'] + 1e-8)
                    new_day_slice[col_idx] = value_norm
                new_seq_segment[0, j, :] = new_day_slice

            # --- 4. Update sequence for next iteration ---
            full_seq = torch.cat([current_seq, new_seq_segment], dim=1)
            current_seq = full_seq[:, -feature_lag:, :]

    return all_forecasts


# --- (plot_enhanced_forecast function is MODIFIED) ---
def plot_enhanced_forecast(
        historical_sales,
        historical_dates,
        historical_preds_all_q,  # <-- CHANGED: Now expects all quantiles
        future_forecast,
        future_dates,
        actual_future=None,
        actual_future_dates=None,
        quantile_bounds=None,
        store_id=None,
        item_id=None,
        save_path=None
):
    """
    Creates enhanced visualization with historical confidence bands.
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

    # 2. Plot historical predictions (Q50) and confidence band (Q10-Q90)
    #    This now expects historical_preds_all_q [N, steps, 3]
    sample_rate = max(1, len(historical_preds_all_q) // 50)

    # Check if we have quantile data for history
    has_hist_quantiles = historical_preds_all_q.ndim == 3 and historical_preds_all_q.shape[2] == 3

    for i in range(0, len(historical_preds_all_q), sample_rate):
        start_idx = i + 28  # FEATURE_LAG

        if has_hist_quantiles:
            # Get all quantiles for this slice
            hist_q10 = historical_preds_all_q[i, :, 0]
            hist_q50 = historical_preds_all_q[i, :, 1]
            hist_q90 = historical_preds_all_q[i, :, 2]
            end_idx = start_idx + len(hist_q50)
        else:
            # Fallback for non-quantile model
            hist_q50 = historical_preds_all_q[i]
            end_idx = start_idx + len(hist_q50)

        if end_idx <= len(historical_dates):
            dates_to_plot = historical_dates[start_idx:end_idx]

            # Plot the median (Q50) red line
            plt.plot(
                dates_to_plot,
                hist_q50,
                color='red',
                linestyle='--',
                linewidth=2,
                alpha=0.7
            )

            # ** NEW: Plot the historical confidence band **
            if has_hist_quantiles:
                plt.fill_between(
                    dates_to_plot,
                    hist_q10,
                    hist_q90,
                    color='red',
                    alpha=0.3 # Very light alpha
                )

    # Add legend entries
    plt.plot([], [], color='red', linestyle='--', label='Historical Predictions (Q50)', linewidth=2)
    if has_hist_quantiles:
        plt.fill_between([], [], [], color='red', alpha=0.1, label='Historical 90% C.I.')

    # 3. Plot future forecast (Q90)
    plt.plot(
        future_dates,
        future_forecast,
        label=f'{len(future_dates)}-Day Forecast (Q90)',
        color='green',
        linestyle='--',
        linewidth=2.5,
        marker='o',
        markersize=3
    )

    # 4. Plot future confidence intervals (Q10-Q90)
    if quantile_bounds is not None:
        lower, upper = quantile_bounds
        plt.fill_between(
            future_dates,
            lower,
            upper,
            color='green',
            alpha=0.2,
            label='90% Confidence Interval (Q10-Q90)'
        )

    # 5. Plot actual future data
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

    title = f"M5 Multi-Month Forecast (Q90)"
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


# --- (main function is MODIFIED) ---
def main():
    # ... (arg parsing, config loading, model loading... unchanged) ...
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='config/model_config_enhanced_m5.yaml')
    parser.add_argument('--store_id_encoded', type=int, default=None)
    parser.add_argument('--item_id_encoded', type=int, default=None)
    parser.add_argument('--forecast_months', type=int, default=12)
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
        print(f"Error: Metadata not found at {metadata_path}")
        return

    norm_meta = meta['normalization']
    train_config = meta['data_config']
    use_quantile = meta.get('use_quantile', False)

    device = torch.device("cpu")
    model_params = meta["model_params"]
    model_params["device"] = "cpu"
    model = EnhancedForecastingModel(**model_params).to(device)

    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        return
    model.eval()
    print("[INFO] Model loaded successfully.")

    processed_path = train_config['output']['processed_data_path']
    if not os.path.exists(processed_path):
        print(f"Error: Processed data not found at {processed_path}")
        return
    print(f"Loading data from {processed_path}...")
    df = pd.read_pickle(processed_path)

    store_id = args.store_id_encoded
    item_id = args.item_id_encoded

    if store_id is None or item_id is None:
        auto_select = True
    else:
        one_series_df = df[
            (df['store_id_encoded'] == store_id) &
            (df['item_id_encoded'] == item_id)
            ].copy()
        if one_series_df.empty:
            auto_select = True
        else:
            auto_select = False

    if auto_select:
        print("Auto-selecting series...")
        series_stats = df.groupby(['store_id_encoded', 'item_id_encoded']).agg({
            'sales': ['sum']
        }).reset_index()
        series_stats.columns = ['store', 'item', 'total_sales']
        series_stats = series_stats.sort_values('total_sales', ascending=False)
        best = series_stats.iloc[0]
        store_id = int(best['store'])
        item_id = int(best['item'])
        print(f"Auto-selected: Store ID: {store_id}, Item ID: {item_id}")
        one_series_df = df[
            (df['store_id_encoded'] == store_id) &
            (df['item_id_encoded'] == item_id)
            ].copy()

    print(f"\n✓ Found {len(one_series_df)} days of data for this series.")

    # ... (Prepare future data template... unchanged) ...
    seq_cols = model_config['SEQUENTIAL_FEATURES']
    static_cols = model_config['STATIC_FEATURES']
    target_col = model_config['TARGET_FEATURE_COLUMN']
    feature_lag = model_config['FEATURE_LAG']
    forecast_steps = model_config['FORECAST_STEPS']
    norm_cols = model_config['SEQUENTIAL_FEATURES']

    derived_indices = {
        col: i for i, col in enumerate(seq_cols) if 'lag_' in col or 'rolling_' in col
    }

    test_months = 3
    test_days = test_months * 30

    if len(one_series_df) > test_days:
        history_df = one_series_df.iloc[:-test_days].copy()
        actual_future_df = one_series_df.iloc[-test_days:].copy()
    else:
        history_df = one_series_df.copy()
        actual_future_df = None

    total_forecast_days = (args.forecast_months * 30)
    last_history_date = history_df['date'].max()
    last_known_price = history_df.iloc[-1]['sell_price']

    try:
        calendar_path = meta['data_config']['dataset']['calendar_path']
        calendar_df = pd.read_csv(calendar_path)
        calendar_df['date'] = pd.to_datetime(calendar_df['date'])
    except FileNotFoundError:
        print(f"Error: Could not find calendar.csv at {calendar_path}")
        return

    event_cols_encoded = [
        'event_name_1_encoded', 'event_type_1_encoded',
        'event_name_2_encoded', 'event_type_2_encoded'
    ]

    if all(col in df.columns for col in event_cols_encoded):
        processed_events_df = df[['date'] + event_cols_encoded].drop_duplicates()
        calendar_df = pd.merge(calendar_df, processed_events_df, on='date', how='left')
    else:
        for col in event_cols_encoded: calendar_df[col] = 0

    future_dates_full_range = pd.date_range(
        start=last_history_date + pd.Timedelta(days=1),
        periods=total_forecast_days
    )

    df_future_template = pd.DataFrame({'date': future_dates_full_range})
    df_future_template['date'] = pd.to_datetime(df_future_template['date'])
    df_future_template = pd.merge(df_future_template, calendar_df, on='date', how='left')

    df_future_template['sell_price'] = last_known_price
    for col in static_cols: df_future_template[col] = history_df[col].iloc[0]
    df_future_template['snap'] = 0
    df_future_template['day_of_week'] = df_future_template['date'].dt.dayofweek
    df_future_template['month'] = df_future_template['date'].dt.month
    df_future_template['year'] = df_future_template['date'].dt.year
    for col in event_cols_encoded: df_future_template[col] = df_future_template[col].fillna(0)
    df_future_template['sales'] = 0.0
    for col in derived_indices.keys(): df_future_template[col] = 0.0
    df_future_template = df_future_template[seq_cols + static_cols]

    history_df_norm, _ = normalize_data(history_df, norm_cols, norm_meta)
    df_future_template_norm, _ = normalize_data(df_future_template, norm_cols, norm_meta)

    X_seq, X_static, Y_norm = create_sequences_daily(
        history_df_norm,
        seq_cols=seq_cols, static_cols=static_cols, target_col=target_col,
        feature_lag=feature_lag, forecast_steps=forecast_steps
    )

    if X_seq.shape[0] == 0:
        print("Error: Series too short to create sequences.")
        return

    X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    X_static_tensor = torch.tensor(X_static, dtype=torch.float32).to(device)

    with torch.no_grad():
        hist_preds_norm = model(X_seq_tensor, X_static_tensor).cpu().numpy()

    print("\n[INFO] Calculating metrics for historical predictions (backtest)...")
    hist_preds_tensor = torch.tensor(hist_preds_norm)
    y_actual_tensor = torch.tensor(Y_norm)
    hist_metrics = compute_metrics(hist_preds_tensor, y_actual_tensor, use_quantile)
    print(f"  Historical MAE (Q50): {hist_metrics['mae']:.4f}")
    print(f"  Historical RMSE (Q50): {hist_metrics['rmse']:.4f}")
    print(f"  Historical MAPE (Q50): {hist_metrics['mape']:.2f}%")
    print(f"  Historical BIAS (Q50): {hist_metrics['bias']:.4f}")

    # Denormalize
    sales_mean = norm_meta['sales']['mean']
    sales_std = norm_meta['sales']['std']

    # --- MODIFIED: Keep all historical quantiles ---
    if use_quantile:
        hist_preds_all_q_denorm = denorm(hist_preds_norm, sales_mean, sales_std)
        # We pass this full [N, steps, 3] array to the plot function
    else:
        # Fallback: make it 3D with one "quantile"
        hist_preds_all_q_denorm = denorm(hist_preds_norm.squeeze(-1), sales_mean, sales_std)

    original_sales = (history_df['sales'] * sales_std) + sales_mean
    historical_dates = history_df['date'].values

    # ... (Multi-month forecast... unchanged) ...
    print(f"\n[INFO] Generating {args.forecast_months}-month forecast...")
    initial_seq_tensor = torch.tensor(X_seq[-1:, :, :], dtype=torch.float32).to(device)
    initial_static_tensor = torch.tensor(X_static[-1:, :], dtype=torch.float32).to(device)
    future_template_tensor = torch.tensor(
        df_future_template_norm[seq_cols].values, dtype=torch.float32
    ).unsqueeze(0).to(device)

    all_future_forecasts = autoregressive_forecast(
        model, initial_seq_tensor, initial_static_tensor, future_template_tensor,
        seq_cols, norm_meta, args.forecast_months, forecast_steps, feature_lag,
        device, quantile_mode=use_quantile
    )

    future_forecast_norm = np.concatenate(all_future_forecasts, axis=1)
    future_forecast_all_q_denorm = denorm(future_forecast_norm, sales_mean, sales_std)
    future_forecast_all_q_denorm = future_forecast_all_q_denorm.squeeze(0)

    if use_quantile:
        future_forecast_denorm = future_forecast_all_q_denorm[:, 2]  # Q90
        q10_denorm = future_forecast_all_q_denorm[:, 0]
        q90_denorm = future_forecast_all_q_denorm[:, 2]
        quantile_bounds = (q10_denorm, q90_denorm)
    else:
        future_forecast_denorm = future_forecast_all_q_denorm.flatten()
        quantile_bounds = None

    future_forecast_denorm = future_forecast_denorm[:total_forecast_days]
    if quantile_bounds:
        quantile_bounds = (
            quantile_bounds[0][:total_forecast_days],
            quantile_bounds[1][:total_forecast_days]
        )

    future_dates = pd.date_range(
        start=last_history_date + pd.Timedelta(days=1),
        periods=len(future_forecast_denorm)
    )

    actual_future_sales = None
    actual_future_dates = None
    if actual_future_df is not None:
        actual_future_sales = (actual_future_df['sales'] * sales_std) + sales_mean
        actual_future_dates = actual_future_df['date'].values

    # --- PLOT (MODIFIED) ---
    print("[INFO] Generating visualization...")
    plot_dir = model_config.get('PREDICT_OUTPUT_DIR', 'predictions_m5_enhanced')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(
        plot_dir,
        f"enhanced_forecast_store{store_id}_item{item_id}_Q90_hist.png"
    )

    plot_enhanced_forecast(
        historical_sales=original_sales.values,
        historical_dates=historical_dates,
        historical_preds_all_q=hist_preds_all_q_denorm,  # <-- PASSING ALL QUANTILES
        future_forecast=future_forecast_denorm,
        future_dates=future_dates,
        actual_future=actual_future_sales.values if actual_future_sales is not None else None,
        actual_future_dates=actual_future_dates,
        quantile_bounds=quantile_bounds,
        store_id=store_id,
        item_id=item_id,
        save_path=plot_path
    )

    # ... (final print statements... unchanged) ...
    print(f"\n[INFO] Forecast complete!")
    print(f"  Series: Store={store_id}, Item={item_id}")
    print(f"  Forecasted {len(future_forecast_denorm)} days ({len(future_forecast_denorm) / 30:.1f} months)")

    if actual_future_sales is not None:
        num_forecast_days = len(future_forecast_denorm)
        if len(actual_future_sales) >= num_forecast_days:
            comparison_actuals = actual_future_sales.values[:num_forecast_days]
            mae_q90 = np.abs(future_forecast_denorm - comparison_actuals).mean()
            print(f"  MAE (Q90) vs actual future ({num_forecast_days} days): {mae_q90:.2f}")
            if use_quantile:
                median_forecast = future_forecast_all_q_denorm[:, 1][:total_forecast_days]
                mae_q50 = np.abs(median_forecast - comparison_actuals).mean()
                print(f"  MAE (Q50) vs actual future ({num_forecast_days} days): {mae_q50:.2f}")
        else:
            print(f"  Not enough actual future data to compute MAE.")


if __name__ == "__main__":
    main()