import streamlit as st
import pandas as pd
import torch
import pickle
import yaml
import numpy as np
import os
from models.transformer_baseline_m5 import ForecastingModel

# -----------------------------------------------------------------
# Caching Functions (for performance)
# -----------------------------------------------------------------

@st.cache_resource
def load_model_and_metadata(config_path):
    """Loads the model and associated metadata once."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(config["METADATA_PATH"], "rb") as f:
        metadata = pickle.load(f)
    
    model = ForecastingModel(**metadata["model_params"])
    model.load_state_dict(torch.load(metadata["checkpoint_path"], map_location="cpu"))
    model.eval()
    return model, metadata, config

@st.cache_data
def load_raw_data(csv_path):
    """
    Loads the RAW data CSV once and creates lists of unique IDs for the UI.
    """
    df = pd.read_csv(csv_path)
    store_ids = sorted(df["store_id"].unique())
    item_ids = sorted(df["item_id"].unique())
    return df, store_ids, item_ids

# -----------------------------------------------------------------
# Prediction & Data Prep Functions
# -----------------------------------------------------------------

def denorm(x, mean, std):
    """Denormalizes a value."""
    return x * std + mean

def prepare_prediction_input(current_sequence_df, metadata):
    """Prepares a sequence of data into tensors for the model."""
    norm_stats = metadata["normalization"]
    
    # The model still needs the encoded columns for prediction
    sales_norm = (current_sequence_df["sales"].values - norm_stats["sales_mean"]) / norm_stats["sales_std"]
    price_norm = (current_sequence_df["sell_price"].values - norm_stats["price_mean"]) / norm_stats["price_std"]
    
    store_norm = (current_sequence_df["store_id_encoded"].iloc[-1] - norm_stats["store_mean"]) / norm_stats["store_std"]
    item_norm = (current_sequence_df["item_id_encoded"].iloc[-1] - norm_stats["item_mean"]) / norm_stats["item_std"]

    x_seq = np.stack([sales_norm, price_norm], axis=1)
    x_seq_tensor = torch.tensor([x_seq], dtype=torch.float32)
    x_static_tensor = torch.tensor([[store_norm, item_norm]], dtype=torch.float32)
    
    return x_seq_tensor, x_static_tensor

def autoregressive_forecast(df_history, model, metadata, feature_lag, forecast_steps):
    """
    Generates a multi-step forecast using an autoregressive loop.
    """
    predictions = []
    # Start with the most recent historical data
    current_sequence_df = df_history.copy().tail(feature_lag)
    
    for _ in range(forecast_steps):
        # Prepare input for the model
        x_seq, x_static = prepare_prediction_input(current_sequence_df, metadata)
        
        # ** INFERENCE STEP **
        with torch.no_grad():
            pred_norm = model(x_seq, x_static)

        # Denormalize the single prediction
        norm_stats = metadata["normalization"]
        pred_denorm = denorm(
            pred_norm.numpy().flatten()[0],
            norm_stats["sales_mean"],
            norm_stats["sales_std"]
        )
        prediction = max(0, pred_denorm)
        predictions.append(prediction)
        
        # --- Update the sequence for the next prediction ---
        last_row = current_sequence_df.iloc[-1]
        next_week_num = last_row["week_num_global"] + 1
        
        new_row = pd.DataFrame({
            "sales": [prediction],
            "sell_price": [last_row["sell_price"]], # Assume sell price stays the same
            "store_id_encoded": [last_row["store_id_encoded"]],
            "item_id_encoded": [last_row["item_id_encoded"]],
            "week_num_global": [next_week_num],
        })
        
        current_sequence_df = pd.concat([current_sequence_df, new_row]).iloc[1:]

    return predictions

# -----------------------------------------------------------------
# Main Streamlit App
# -----------------------------------------------------------------

st.set_page_config(page_title="M5 Sales Forecaster", layout="wide")
st.title("🛍️ On-Demand M5 Sales Forecaster")

# --- Initialize Session State ---
if 'prediction_df' not in st.session_state:
    st.session_state.prediction_df = None

# --- File Paths & Loading ---
CONFIG_FILE = "config/model_config_m5.yaml"
# ++ UPDATED: Now points to the raw, human-readable CSV file ++
DATA_CSV_FILE = "data/m5_weekly.csv" 

try:
    model, metadata, config = load_model_and_metadata(CONFIG_FILE)
    # ++ UPDATED: Calls the new data loading function ++
    df, store_ids, item_ids = load_raw_data(DATA_CSV_FILE)
except FileNotFoundError as e:
    st.error(f"FATAL: A required file was not found. Please check paths. Error: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during loading: {e}")
    st.stop()

# --- Sidebar for User Inputs ---
st.sidebar.header("Forecast Parameters")

selected_store_id = st.sidebar.selectbox("Select Store ID:", options=store_ids)
selected_item_id = st.sidebar.selectbox("Select Item ID:", options=item_ids)

feature_lag_options = [4, 8, 12, 16]
selected_feature_lag = st.sidebar.select_slider(
    "Select Weeks of History to Use (Feature Lag):",
    options=feature_lag_options,
    value=config["MODEL_PARAMS"]["FEATURE_LAG"]
)

selected_forecast_steps = st.sidebar.number_input(
    "Select Weeks to Forecast Ahead:",
    min_value=1,
    max_value=52,
    value=4
)

# --- Forecast Button ---
if st.sidebar.button("🚀 Generate Forecast", type="primary"):
    with st.spinner("Filtering data and forecasting..."):
        # ++ UPDATED: Filters the raw dataframe on the human-readable IDs ++
        df_filtered = df[
            (df["store_id"] == selected_store_id) &
            (df["item_id"] == selected_item_id)
        ].copy()

        if "week_num_global" in df_filtered.columns:
            df_filtered.sort_values("week_num_global", inplace=True)

        if len(df_filtered) < selected_feature_lag:
            st.error(f"Not enough data. Need {selected_feature_lag} weeks, but found {len(df_filtered)}.")
            st.session_state.prediction_df = None
        else:
            forecasted_values = autoregressive_forecast(
                df_filtered, model, metadata, selected_feature_lag, selected_forecast_steps
            )
            
            history_df = df_filtered.tail(52)
            history_df = history_df[["week_num_global", "sales"]].copy()
            history_df["Type"] = "Historical"

            last_week_num = history_df["week_num_global"].iloc[-1]
            forecast_df = pd.DataFrame({
                "week_num_global": [last_week_num + i for i in range(1, selected_forecast_steps + 1)],
                "sales": forecasted_values,
                "Type": ["Forecast"] * selected_forecast_steps
            })

            st.session_state.prediction_df = pd.concat([history_df, forecast_df])

# --- Main Panel for Displaying Results ---
if st.session_state.prediction_df is not None:
    st.subheader(f"Sales Forecast for Item `{selected_item_id}` at Store `{selected_store_id}`")
    plot_df = st.session_state.prediction_df
    
    st.line_chart(
        plot_df, x="week_num_global", y="sales", color="Type", height=500
    )

    st.subheader("Raw Prediction Data")
    st.dataframe(plot_df[plot_df["Type"] == "Forecast"])
else:
    st.info("Select parameters on the left and click 'Generate Forecast' to see the results.")