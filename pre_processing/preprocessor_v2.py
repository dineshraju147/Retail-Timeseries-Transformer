# preprocessor.py
"""
M5 Dataset Preprocessor (Daily)

Loads raw M5 data (sales, calendar, prices) and performs 
feature engineering to create a daily-level DataFrame.
"""
from typing import List, Tuple, Dict, Any # Make sure this is at the top of the file
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset
from tqdm import tqdm



# pre_processing/preprocessor.py

# ... (all the existing code, including load_and_preprocess_daily, stays here) ...

# -------------------------------------------------------------------
# This function should be at the top of your file
# -------------------------------------------------------------------

def add_lags_rollings(g: pd.DataFrame, lags: List[int], windows: List[int]) -> pd.DataFrame:
    """
    Applies lags and rollings from notebook.
    Note: shift(1) is used for rollings to prevent leakage.
    """
    g = g.copy()
    for lag in lags:
        g[f'lag_{lag}'] = g['sales'].shift(lag)
    
    for window in windows:
        # Use shift(1) to ensure we are not using the current day's sales
        g[f'rolling_{window}'] = g['sales'].shift(1).rolling(window).mean()
        
    return g

# -------------------------------------------------------------------
# This is the main function you requested
# -------------------------------------------------------------------

def load_and_preprocess_daily(config_dataset: Dict) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Loads raw M5 data and processes it into a daily feature DataFrame.
    """
    
    print("="*70)
    print("🚀 M5 PREPROCESSOR (Daily Mode)")
    print("="*70)

    # === 1. Load Base Data ===
    print("Loading base data...")
    DATA_DIR = config_dataset['dataset']['sales_path'].replace('sales_train_validation.csv', '')
    sample_items = config_dataset['dataset'].get('sample_items')
    
    calendar = pd.read_csv(f"{DATA_DIR}/calendar.csv")
    prices = pd.read_csv(f"{DATA_DIR}/sell_prices.csv")
    sales = pd.read_csv(f"{DATA_DIR}/sales_train_validation.csv")
    
    # === 2. Melt Sales Data (Wide to Long) ===
    print("Melting sales data to long format...")
    
    # Handle sampling
    if sample_items:
        print(f"Sampling {sample_items} items...")
        # Get a list of unique item IDs
        all_ids = sales['id'].unique()
        # Sample the IDs
        sampled_ids = np.random.choice(all_ids, sample_items, replace=False)
        # Filter the sales DataFrame
        sales = sales[sales['id'].isin(sampled_ids)].reset_index(drop=True)

    # Melt the DataFrame
    id_cols = [col for col in sales.columns if col.startswith('id') or col in ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']]
    d_cols = [col for col in sales.columns if col.startswith('d_')]
    
    df = sales.melt(
        id_vars=id_cols,
        value_vars=d_cols,
        var_name='d',
        value_name='sales'
    )
    # --- NEW: APPLY LOG-TRANSFORM ---
    # We use log1p which is log(x + 1) to handle sales of 0.
    df['sales'] = np.log1p(df['sales'])
    print("Applied np.log1p (log(x+1)) to 'sales' target.")

    # === 3. Label Encode All Categorical Features ===
    print("Label encoding all categorical features...")
    
    # We will store the encoders to save them later
    encoders = {}
    norm_meta = {} # Initialize normalization metadata dict
    
    # Static features
    static_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    for col in tqdm(static_cols, desc="Encoding static features"):
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        encoders[col] = le
        
    # Dynamic (event) features
    event_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    # Fill NaNs first so 'None' becomes a valid category
    calendar[event_cols] = calendar[event_cols].fillna('None') 
    
    for col in tqdm(event_cols, desc="Encoding event features"):
        le = LabelEncoder()
        # Fit on the calendar column
        calendar[f'{col}_encoded'] = le.fit_transform(calendar[col])
        encoders[col] = le

    # === 4. Print Cardinalities (NEW) ===
    print("\n" + "="*30)
    print("Static Feature Cardinalities:")
    print("="*30)
    print("These are the counts for your model_config_m5.yaml")
    
    for col in static_cols:
        col_name = f'{col}_encoded'
        count = len(encoders[col].classes_)
        print(f"  {col_name}: {count}")
        
    print("="*30 + "\n")

    # === 5. Merge Price & Calendar Features ===
    print("Merging calendar and price data...")
    # Merge calendar
    df = pd.merge(df, calendar, on='d', how='left')
    # Merge prices
    df = pd.merge(df, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

    # === 6. Engineer Calendar Features ===
    print("Engineeering date and SNAP features...")
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    # Drop original d, wm_yr_wk, and un-encoded event columns
    encoded_event_cols = [f'{col}_encoded' for col in event_cols]
    df = df.drop(columns=['d', 'wm_yr_wk'] + event_cols)
    
    # --- START OF FIX ---
    # SNAP features
    df['snap'] = 0
    
    # Get the list of states the encoder *actually* learned from the sample
    seen_states = encoders['state_id'].classes_
    
    print(f"Applying SNAP logic for seen states: {list(seen_states)}")
    
    # Only apply SNAP logic for states present in the sample
    if 'CA' in seen_states:
        ca_id = encoders['state_id'].transform(['CA'])[0]
        df.loc[df['state_id_encoded'] == ca_id, 'snap'] = df['snap_CA']
        
    if 'TX' in seen_states:
        tx_id = encoders['state_id'].transform(['TX'])[0]
        df.loc[df['state_id_encoded'] == tx_id, 'snap'] = df['snap_TX']
        
    if 'WI' in seen_states:
        wi_id = encoders['state_id'].transform(['WI'])[0]
        df.loc[df['state_id_encoded'] == wi_id, 'snap'] = df['snap_WI']
        
    df = df.drop(['snap_CA', 'snap_TX', 'snap_WI'], axis=1)
    # --- END OF FIX ---

    # === 7. Create Lags & Rollings ===
    # This is VERY slow. For a real project, use polars or numba.
    lags = [1, 2, 3, 7, 28] 
    windows = [7, 28]
    print(f"Creating lags {lags} and rollings {windows}. This will take a while...")
    
    # Sort by id and date to ensure correct lag/rolling calculation
    df = df.sort_values(['id', 'date']).reset_index(drop=True)
    
    # Group by the unique series ID
    df = df.groupby('id', group_keys=False).apply(
        lambda g: add_lags_rollings(g, lags, windows)
    )

    # === 8. Final Cleanup ===
    print("Final cleaning and NaN handling...")
    
    # Fill sell_price NaNs
    # Use ffill (forward fill) first within a group to fill missing prices
    # Then use global mean for items that *never* had a price (at the start)
    df['sell_price'] = df.groupby('id')['sell_price'].ffill()
    df['sell_price'] = df['sell_price'].fillna(df['sell_price'].mean())
    
    # Fill lag/rolling NaNs (with 0, as they represent no past data)
    lag_cols = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
    print(f"Filling NaNs in {len(lag_cols)} lagged columns with 0...")
    df[lag_cols] = df[lag_cols].fillna(0)
    
    # Drop any remaining NaNs (e.g., from the very start of time series)
    df = df.dropna().reset_index(drop=True)
    
    print(f"Preprocessing complete. Final DataFrame shape: {df.shape}")
    
    return df, encoders, norm_meta

# --- Other functions (like normalize_data, create_sequences_daily) go here ---


# --- ADD THE FUNCTIONS BELOW TO THE END OF THE FILE ---



def normalize_data(df: pd.DataFrame, columns: List[str], norm_meta: Dict = None) -> Tuple[pd.DataFrame, Dict]:
    """Normalizes specified columns in the dataframe."""
    if norm_meta is None:
        norm_meta = {}
    
    for col in columns:
        if col not in norm_meta:
            mean, std = df[col].mean(), df[col].std()
            std = 1.0 if std == 0 else std
            norm_meta[col] = {'mean': mean, 'std': std}
        
        mean = norm_meta[col]['mean']
        std = norm_meta[col]['std']
        df[col] = (df[col] - mean) / (std + 1e-8) # Added epsilon for safety
        
    return df, norm_meta


def create_sequences_daily(
    df_series: pd.DataFrame,
    seq_cols: List[str],
    static_cols: List[str],
    target_col: str,
    feature_lag: int,
    forecast_steps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates sequences (X_seq, X_static, Y) for a SINGLE time series DataFrame.
    This is applied to each group (e.g., one item-store combination).
    """
    # Get the raw values
    data_seq = df_series[seq_cols].values
    data_static = df_series[static_cols].iloc[0].values # Static features are constant
    data_target = df_series[target_col].values
    
    num_samples = len(data_target) - feature_lag - forecast_steps + 1
    
    X_seq_list, X_static_list, Y_list = [], [], []

    for i in range(num_samples):
        X_seq_list.append(data_seq[i : i + feature_lag])
        X_static_list.append(data_static)
        Y_list.append(data_target[i + feature_lag : i + feature_lag + forecast_steps])

    if not X_seq_list:
        empty_seq = np.empty((0, feature_lag, len(seq_cols)), dtype=np.float32)
        empty_static = np.empty((0, len(static_cols)), dtype=np.float32)
        empty_y = np.empty((0, forecast_steps), dtype=np.float32)
        return empty_seq, empty_static, empty_y

    return np.array(X_seq_list, dtype=np.float32), \
           np.array(X_static_list, dtype=np.float32), \
           np.array(Y_list, dtype=np.float32)




# --- ADD THIS FUNCTION TO THE END OF preprocessor.py ---

import torch # Make sure torch is imported at the top

def create_tft_sequences(
    df_series: pd.DataFrame,
    enc_cont_cols: List[str],
    enc_cat_cols: List[str],
    dec_cat_cols: List[str],
    target_col: str,
    feature_lag: int,
    forecast_steps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates sequences (enc_cont, enc_cat, dec_cat, Y) for a SINGLE 
    time series DataFrame, formatted for the TFTLite model.
    
    Args:
        df_series: DataFrame for a single item-store.
        enc_cont_cols: List of continuous features for the encoder (past).
        enc_cat_cols: List of categorical features for the encoder (past).
        dec_cat_cols: List of categorical features for the decoder (future).
        target_col: The target variable (e.g., 'sales').
        feature_lag: How many past steps to use (T).
        forecast_steps: How many future steps to predict (H).
    
    Returns:
        Tuple of (all_enc_cont, all_enc_cat, all_dec_cat, all_Y)
    """
    
    total_len = len(df_series)
    
    # We need at least T (feature_lag) + H (forecast_steps) data points
    if total_len < feature_lag + forecast_steps:
        return (
            np.empty((0, feature_lag, len(enc_cont_cols))),
            np.empty((0, feature_lag, len(enc_cat_cols))),
            np.empty((0, forecast_steps, len(dec_cat_cols))),
            np.empty((0, forecast_steps))
        )

    all_enc_cont, all_enc_cat, all_dec_cat, all_Y = [], [], [], []

    # Iterate to create overlapping windows
    for i in range(total_len - feature_lag - forecast_steps + 1):
        
        # Encoder (Past): t = i to i + feature_lag - 1
        encoder_start = i
        encoder_end = i + feature_lag
        
        # Decoder (Future): t = i + feature_lag to i + feature_lag + forecast_steps - 1
        decoder_start = encoder_end
        decoder_end = encoder_end + forecast_steps

        # --- Encoder Inputs (Past) ---
        encoder_slice = df_series.iloc[encoder_start:encoder_end]
        
        enc_cont = encoder_slice[enc_cont_cols].values
        enc_cat = encoder_slice[enc_cat_cols].values
        
        # --- Decoder Inputs (Future) ---
        # Note: For TFT, the decoder inputs are *known* future features
        decoder_slice = df_series.iloc[decoder_start:decoder_end]
        
        dec_cat = decoder_slice[dec_cat_cols].values
        
        # --- Target (Future) ---
        target_slice = df_series.iloc[decoder_start:decoder_end]
        y = target_slice[target_col].values

        all_enc_cont.append(enc_cont)
        all_enc_cat.append(enc_cat)
        all_dec_cat.append(dec_cat)
        all_Y.append(y)

    return (
        np.array(all_enc_cont, dtype=np.float32),
        np.array(all_enc_cat, dtype=np.int64),  # Categoricals are indices
        np.array(all_dec_cat, dtype=np.int64),  # Categoricals are indices
        np.array(all_Y, dtype=np.float32)
    )

def build_tft_dataset(
    df: pd.DataFrame,
    model_config: Dict,
    norm_meta: Dict,
    is_train: bool = True
) -> Tuple[TensorDataset, Dict]:
    """Builds a TensorDataset for the TFT model."""
    
    print("Building TFT dataset...")
    
    # Normalize continuous features
    # Note: We only normalize *encoder* continuous features
    # The target will be normalized if it's in this list, which is correct.
    norm_cols = model_config['TFT_ENC_CONT_COLS']
    df, norm_meta = normalize_data(df, norm_cols, norm_meta)

    # Create sequences
    all_enc_cont, all_enc_cat, all_dec_cat, all_Y = [], [], [], []
    
    grouped = df.groupby('id')
    
    # Get all column lists from config
    enc_cont_cols = model_config['TFT_ENC_CONT_COLS']
    enc_cat_cols = model_config['TFT_ENC_CAT_COLS']
    dec_cat_cols = model_config['TFT_DEC_CAT_COLS']
    target_col = model_config['TARGET_FEATURE_COLUMN']
    feature_lag = model_config['FEATURE_LAG']
    forecast_steps = model_config['FORECAST_STEPS']

    # Adjust data split
    split_date = pd.to_datetime(model_config['SPLIT_DATE'])
    
    for _, group_df in tqdm(grouped, desc="Creating TFT sequences"):
        if is_train:
            series_df = group_df[group_df['date'] < split_date]
        else:
            # For validation/test, we need the *full* history to create
            # the first sequence, but we only *keep* sequences
            # where the *target* (Y) is in the validation period.
            
            # Find the start index for validation targets
            val_start_idx = group_df.index.get_loc(
                group_df[group_df['date'] >= split_date].index[0]
            )
            
            # The first possible *encoder* start index is
            # `val_start_idx - feature_lag - forecast_steps + 1`
            # But to be safe and simple, we'll just use the full history
            # and let the sequence creator handle the start.
            series_df = group_df

        # Create sequences for this one series
        s_enc_cont, s_enc_cat, s_dec_cat, s_Y = create_tft_sequences(
            series_df,
            enc_cont_cols=enc_cont_cols,
            enc_cat_cols=enc_cat_cols,
            dec_cat_cols=dec_cat_cols,
            target_col=target_col,
            feature_lag=feature_lag,
            forecast_steps=forecast_steps
        )

        if not is_train:
            # Filter sequences to *only* those whose
            # *first target* is on or after the split date
            
            # Get the date for the *first target* of each sequence
            # The index of the first target is `i + feature_lag`
            seq_start_indices = np.arange(len(s_Y)) + feature_lag
            
            # Check if these indices are within the bounds of the series
            if len(seq_start_indices) > 0 and seq_start_indices[-1] < len(series_df):
                target_start_dates = series_df.iloc[seq_start_indices]['date'].values
                
                val_mask = target_start_dates >= split_date
                
                s_enc_cont = s_enc_cont[val_mask]
                s_enc_cat = s_enc_cat[val_mask]
                s_dec_cat = s_dec_cat[val_mask]
                s_Y = s_Y[val_mask]
            else:
                # No valid sequences
                continue
        
        if s_Y.shape[0] > 0:
            all_enc_cont.append(s_enc_cont)
            all_enc_cat.append(s_enc_cat)
            all_dec_cat.append(s_dec_cat)
            all_Y.append(s_Y)

    # Combine all series
    X_enc_cont = torch.tensor(np.concatenate(all_enc_cont), dtype=torch.float32)
    X_enc_cat = torch.tensor(np.concatenate(all_enc_cat), dtype=torch.long)
    X_dec_cat = torch.tensor(np.concatenate(all_dec_cat), dtype=torch.long)
    Y = torch.tensor(np.concatenate(all_Y), dtype=torch.float32)
    
    print(f"Dataset shapes:")
    print(f"  Encoder Cont: {X_enc_cont.shape}")
    print(f"  Encoder Cat:  {X_enc_cat.shape}")
    print(f"  Decoder Cat:  {X_dec_cat.shape}")
    print(f"  Target (Y):   {Y.shape}")
    
    return TensorDataset(X_enc_cont, X_enc_cat, X_dec_cat, Y), norm_meta