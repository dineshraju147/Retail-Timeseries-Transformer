import streamlit as st
import pandas as pd
import torch
import pickle
import yaml
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import timedelta
from models.transformer_baseline_m5 import ForecastingModel
from pre_processing.preprocessor import create_sequences_daily, normalize_data
from models.transformer_enhanced_m5 import compute_metrics

# -----------------------------------------------------------------
# Caching Functions (for performance)
# -----------------------------------------------------------------

@st.cache_resource
def load_model_and_metadata(model_config_path):
    """Loads the model and associated metadata once."""
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    
    metadata_path = model_config["METADATA_PATH"]
    checkpoint_path = model_config["CHECKPOINT_PATH"]
    
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    device = torch.device("cpu")
    model_params = metadata["model_params"]
    model_params["device"] = "cpu"
    
    model = ForecastingModel(**model_params).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    return model, metadata, model_config

@st.cache_data
def load_processed_data(processed_data_path):
    """
    Loads the processed pickle data and creates lists of available IDs.
    """
    df = pd.read_pickle(processed_data_path)
    
    # Get unique combinations of store_id_encoded and item_id_encoded
    # Create readable labels for the UI
    unique_combos = df[['store_id_encoded', 'item_id_encoded']].drop_duplicates()
    
    # Create display labels (you can enhance this to show actual IDs if available)
    store_ids_encoded = sorted(unique_combos['store_id_encoded'].unique())
    item_ids_encoded = sorted(unique_combos['item_id_encoded'].unique())
    
    return df, store_ids_encoded, item_ids_encoded

# -----------------------------------------------------------------
# Prediction & Data Prep Functions
# -----------------------------------------------------------------

def denorm(x, mean, std):
    """Denormalizes data."""
    return x * (std + 1e-8) + mean

def autoregressive_forecast_days(
    model,
    initial_seq,
    initial_static,
    forecast_days: int,
    forecast_steps: int,
    feature_lag: int,
    device
):
    """
    Generates multi-day forecast using autoregressive iteration.
    Similar to predict_m5.py but works with days instead of months.
    """
    model.eval()
    all_forecasts = []
    
    current_seq = initial_seq.clone()
    
    # Calculate total iterations needed
    total_iterations = (forecast_days + forecast_steps - 1) // forecast_steps
    
    with torch.no_grad():
        for i in range(total_iterations):
            # Predict next window
            output = model(current_seq, initial_static)
            pred = output  # [1, forecast_steps]
            
            all_forecasts.append(pred.cpu().numpy())
            
            # Update sequence for next iteration
            # Create new sequence segment - initialize with zeros
            new_seq_segment = torch.zeros(
                1, forecast_steps, current_seq.size(2)
            ).to(device)
            
            # Update sales feature (first column) with predictions
            new_seq_segment[:, :, 0] = pred
            
            # For other features, copy from the last row of current sequence
            # This is a simplification - in production you'd update calendar features, etc.
            if current_seq.size(2) > 1:
                last_row_other_features = current_seq[:, -1:, 1:].expand(1, forecast_steps, -1)
                new_seq_segment[:, :, 1:] = last_row_other_features
            
            # Append the new segment to the old sequence
            full_seq = torch.cat([current_seq, new_seq_segment], dim=1)
            
            # The new input sequence is the last `feature_lag` steps
            current_seq = full_seq[:, -feature_lag:, :]
    
    # Flatten all forecasts
    forecast_flat = np.concatenate([f.flatten() for f in all_forecasts])
    
    # Trim to exact forecast_days
    forecast_flat = forecast_flat[:forecast_days]
    
    return forecast_flat

def plot_forecast_streamlit(
    historical_sales,
    historical_dates,
    historical_preds,
    future_forecast,
    future_dates,
    store_id=None,
    item_id=None,
    feature_lag=28
):
    """
    Creates enhanced visualization matching predict_m5.py style.
    """
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # 1. Plot historical actuals
    ax.plot(
        historical_dates, 
        historical_sales, 
        label='Historical Actual Sales', 
        color='blue', 
        linewidth=2,
        alpha=0.8
    )
    
    # 2. Plot historical predictions (sample every N for clarity)
    if len(historical_preds) > 0:
        sample_rate = max(1, len(historical_preds) // 50)
        for i in range(0, len(historical_preds), sample_rate):
            start_idx = i + feature_lag 
            end_idx = start_idx + len(historical_preds[i])
            
            if end_idx <= len(historical_dates):
                ax.plot(
                    historical_dates[start_idx:end_idx],
                    historical_preds[i],
                    color='red',
                    linewidth=2,
                    alpha=0.7,
                    linestyle='--'
                )
        
        # Add legend entry for historical preds
        ax.plot([], [], color='red', linestyle='--', label='Historical Predictions', linewidth=2)
    
    # 3. Plot future forecast
    ax.plot(
        future_dates,
        future_forecast,
        label=f'{len(future_dates)}-Day Forecast',
        color='green',
        linestyle='--',
        linewidth=2.5,
        marker='o',
        markersize=3
    )
    
    # Formatting
    last_history_date = historical_dates[-1]
    ax.axvline(
        last_history_date, 
        color='gray', 
        linestyle=':', 
        linewidth=2,
        label='Forecast Start'
    )
    
    title = f"M5 Multi-Month Forecast (Original Model)"
    if store_id is not None and item_id is not None:
        title += f" (Store: {store_id}, Item: {item_id})"
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Sales', fontsize=14)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    return fig

# -----------------------------------------------------------------
# Main Streamlit App
# -----------------------------------------------------------------

st.set_page_config(page_title="M5 Sales Forecaster", layout="wide")
st.title("🛍️ M5 Daily Sales Forecaster")

# --- Initialize Session State ---
if 'prediction_df' not in st.session_state:
    st.session_state.prediction_df = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'historical_preds' not in st.session_state:
    st.session_state.historical_preds = None
if 'historical_dates' not in st.session_state:
    st.session_state.historical_dates = None
if 'historical_sales' not in st.session_state:
    st.session_state.historical_sales = None
if 'future_forecast' not in st.session_state:
    st.session_state.future_forecast = None
if 'future_dates' not in st.session_state:
    st.session_state.future_dates = None

# --- File Paths & Loading ---
MODEL_CONFIG_FILE = "config/model_config_m5.yaml"
DATA_CONFIG_FILE = "config/config_m5.yaml"

try:
    model, metadata, model_config = load_model_and_metadata(MODEL_CONFIG_FILE)
    
    # Load data config to get processed data path
    with open(DATA_CONFIG_FILE, "r") as f:
        data_config = yaml.safe_load(f)
    
    processed_data_path = data_config['output']['processed_data_path']
    
    if not os.path.exists(processed_data_path):
        st.error(f"Processed data file not found at {processed_data_path}")
        st.info("Please run train_m5.py first to create the processed data file.")
        st.stop()
    
    df, store_ids_encoded, item_ids_encoded = load_processed_data(processed_data_path)
    
    st.sidebar.success(f"✅ Loaded {len(df)} rows of processed data")
    
except FileNotFoundError as e:
    st.error(f"FATAL: A required file was not found. Please check paths. Error: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during loading: {e}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

# --- Sidebar for User Inputs ---
st.sidebar.header("Forecast Parameters")

selected_store_id_encoded = st.sidebar.selectbox(
    "Select Store ID (Encoded):", 
    options=store_ids_encoded,
    format_func=lambda x: f"Store {x}"
)

selected_item_id_encoded = st.sidebar.selectbox(
    "Select Item ID (Encoded):", 
    options=item_ids_encoded,
    format_func=lambda x: f"Item {x}"
)

# Get model config values
feature_lag = model_config['FEATURE_LAG']
forecast_steps = model_config['FORECAST_STEPS']

st.sidebar.info(f"**Model Config:**\n- Feature Lag: {feature_lag} days\n- Forecast Steps: {forecast_steps} days per window")

selected_forecast_days = st.sidebar.number_input(
    "Days to Forecast Ahead:",
    min_value=forecast_steps,
    max_value=365,
    value=90,
    step=forecast_steps,
    help=f"Must be a multiple of {forecast_steps} for best results"
)

show_metrics = st.sidebar.checkbox("Show Historical Metrics", value=True)

# --- Forecast Button ---
if st.sidebar.button("🚀 Generate Forecast", type="primary"):
    with st.spinner("Filtering data and generating forecast..."):
        # Filter data for selected store and item
        one_series_df = df[
            (df['store_id_encoded'] == selected_store_id_encoded) &
            (df['item_id_encoded'] == selected_item_id_encoded)
        ].copy()

        if one_series_df.empty:
            st.error(f"No data found for Store {selected_store_id_encoded}, Item {selected_item_id_encoded}")
            st.session_state.prediction_df = None
            st.session_state.metrics = None
            st.session_state.historical_preds = None
            st.session_state.historical_dates = None
            st.session_state.historical_sales = None
            st.session_state.future_forecast = None
            st.session_state.future_dates = None
        else:
            # Sort by date
            if 'date' in one_series_df.columns:
                one_series_df = one_series_df.sort_values('date').reset_index(drop=True)
            
            if len(one_series_df) < feature_lag + forecast_steps:
                st.error(f"Not enough data. Need at least {feature_lag + forecast_steps} days, but found {len(one_series_df)}.")
                st.session_state.prediction_df = None
                st.session_state.metrics = None
                st.session_state.historical_preds = None
                st.session_state.historical_dates = None
                st.session_state.historical_sales = None
                st.session_state.future_forecast = None
                st.session_state.future_dates = None
            else:
                # Normalize the series
                train_model_config = metadata['model_config']
                norm_cols = train_model_config['SEQUENTIAL_FEATURES']
                one_series_df_norm, _ = normalize_data(
                    one_series_df.copy(), 
                    norm_cols, 
                    metadata['normalization']
                )
                
                # Create sequences for historical prediction
                X_seq, X_static, Y_norm = create_sequences_daily(
                    one_series_df_norm,
                    seq_cols=train_model_config['SEQUENTIAL_FEATURES'],
                    static_cols=train_model_config['STATIC_FEATURES'],
                    target_col=train_model_config['TARGET_FEATURE_COLUMN'],
                    feature_lag=feature_lag,
                    forecast_steps=forecast_steps
                )
                
                if X_seq.shape[0] == 0:
                    st.error("Could not create sequences from this series.")
                    st.session_state.prediction_df = None
                    st.session_state.metrics = None
                    st.session_state.historical_preds = None
                    st.session_state.historical_dates = None
                    st.session_state.historical_sales = None
                    st.session_state.future_forecast = None
                    st.session_state.future_dates = None
                else:
                    device = torch.device("cpu")
                    
                    # A. Run inference on historical data for metrics
                    X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
                    X_static_tensor = torch.tensor(X_static, dtype=torch.float32).to(device)
                    
                    with torch.no_grad():
                        hist_preds_norm = model(X_seq_tensor, X_static_tensor).cpu().numpy()
                    
                    # Calculate metrics if requested
                    if show_metrics:
                        hist_preds_tensor = torch.tensor(hist_preds_norm)
                        y_actual_tensor = torch.tensor(Y_norm)
                        metrics = compute_metrics(
                            hist_preds_tensor,
                            y_actual_tensor,
                            quantile_mode=False
                        )
                        st.session_state.metrics = metrics
                    
                    # B. Denormalize historical predictions
                    sales_mean = metadata['normalization']['sales']['mean']
                    sales_std = metadata['normalization']['sales']['std']
                    
                    # Denormalize historical predictions for plotting
                    hist_preds_denorm = []
                    for pred_norm in hist_preds_norm:
                        pred_denorm = denorm(pred_norm, sales_mean, sales_std)
                        hist_preds_denorm.append(pred_denorm)
                    
                    # Get the last sequence for autoregressive forecast
                    initial_seq = X_seq_tensor[-1:, :, :]  # [1, seq_len, features]
                    initial_static = X_static_tensor[-1:, :]  # [1, static_features]
                    
                    # C. Generate future forecast
                    future_forecast_norm = autoregressive_forecast_days(
                        model,
                        initial_seq,
                        initial_static,
                        selected_forecast_days,
                        forecast_steps,
                        feature_lag,
                        device
                    )
                    
                    # Denormalize future forecast
                    future_forecast = denorm(future_forecast_norm, sales_mean, sales_std)
                    future_forecast = np.maximum(future_forecast, 0)  # Ensure non-negative
                    
                    # Get historical sales (denormalized)
                    original_sales = (one_series_df['sales'] * sales_std) + sales_mean
                    
                    # Generate dates
                    if 'date' in one_series_df.columns:
                        all_dates = pd.to_datetime(one_series_df['date'].values)
                        last_history_date = all_dates[-1]
                        future_dates = pd.date_range(
                            start=last_history_date + pd.Timedelta(days=1),
                            periods=len(future_forecast)
                        )
                    else:
                        # Fallback: use index as days
                        all_dates = pd.date_range(
                            start='2011-01-29',
                            periods=len(original_sales),
                            freq='D'
                        )
                        last_history_date = all_dates[-1]
                        future_dates = pd.date_range(
                            start=last_history_date + pd.Timedelta(days=1),
                            periods=len(future_forecast)
                        )
                    
                    # Store in session state for plotting
                    st.session_state.historical_sales = original_sales.values
                    st.session_state.historical_dates = all_dates
                    st.session_state.historical_preds = hist_preds_denorm
                    st.session_state.future_forecast = future_forecast
                    st.session_state.future_dates = future_dates
                    
                    # Create DataFrame for data table
                    history_df = pd.DataFrame({
                        'date': all_dates,
                        'sales': original_sales.values,
                        'Type': 'Historical'
                    })
                    
                    forecast_df = pd.DataFrame({
                        'date': future_dates,
                        'sales': future_forecast,
                        'Type': 'Forecast'
                    })
                    
                    st.session_state.prediction_df = pd.concat([history_df, forecast_df], ignore_index=True)

# --- Main Panel for Displaying Results ---
if st.session_state.prediction_df is not None:
    st.subheader(f"Sales Forecast for Store {selected_store_id_encoded}, Item {selected_item_id_encoded}")
    
    # Display metrics if available
    if st.session_state.metrics is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Historical MAE", f"{st.session_state.metrics['mae']:.4f}")
        with col2:
            st.metric("Historical RMSE", f"{st.session_state.metrics['rmse']:.4f}")
        with col3:
            st.metric("Historical MAPE", f"{st.session_state.metrics['mape']:.2f}%")
        with col4:
            st.metric("Historical BIAS", f"{st.session_state.metrics['bias']:.4f}")
    
    # Create matplotlib plot matching predict_m5.py style
    if (st.session_state.historical_sales is not None and 
        st.session_state.historical_dates is not None and
        st.session_state.historical_preds is not None and
        st.session_state.future_forecast is not None and
        st.session_state.future_dates is not None):
        
        fig = plot_forecast_streamlit(
            historical_sales=st.session_state.historical_sales,
            historical_dates=st.session_state.historical_dates,
            historical_preds=st.session_state.historical_preds,
            future_forecast=st.session_state.future_forecast,
            future_dates=st.session_state.future_dates,
            store_id=selected_store_id_encoded,
            item_id=selected_item_id_encoded,
            feature_lag=feature_lag
        )
        st.pyplot(fig)
        plt.close(fig)  # Close the figure to free memory
    
    # Show forecast period data table
    plot_df = st.session_state.prediction_df.copy()
    forecast_df_only = plot_df[plot_df['Type'] == 'Forecast']
    if len(forecast_df_only) > 0:
        st.subheader("Forecast Details")
        st.dataframe(
            forecast_df_only[['date', 'sales']].style.format({
                'sales': '{:.2f}'
            }),
            use_container_width=True
        )
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Forecast Period", f"{len(forecast_df_only)} days")
        with col2:
            st.metric("Avg Daily Forecast", f"{forecast_df_only['sales'].mean():.2f}")
        with col3:
            st.metric("Total Forecast", f"{forecast_df_only['sales'].sum():.2f}")
else:
    st.info("👈 Select parameters on the left and click 'Generate Forecast' to see the results.")
    
    # Show some info about the data
    with st.expander("📊 Data Information"):
        st.write(f"**Total rows in dataset:** {len(df):,}")
        st.write(f"**Unique stores:** {len(store_ids_encoded)}")
        st.write(f"**Unique items:** {len(item_ids_encoded)}")
        st.write(f"**Date range:** {df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "Date column not available")
