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
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
from sklearn.metrics import mean_absolute_error # <-- ADDED THIS IMPORT

# --- Import from your project files ---
from models.transformer_baseline_m5 import ForecastingModel 
from pre_processing.preprocessor import (
    load_and_preprocess_daily, 
    create_sequences_daily, 
    normalize_data
)


def build_dataset(
    df: pd.DataFrame,
    model_config: Dict,
    norm_meta: Dict
) -> Tuple[TensorDataset, Dict]:
    """Builds a TensorDataset from the preprocessed DataFrame."""
    
    print("Building dataset...")
    
    # --- 1. Normalize Data ---
    norm_cols = model_config['SEQUENTIAL_FEATURES'] + model_config['STATIC_FEATURES']
    df, norm_meta = normalize_data(df, norm_cols, norm_meta)

    # --- 2. Create Sequences ---
    all_X_seq, all_X_static, all_Y = [], [], []
    
    grouped = df.groupby('id')
    
    for _, group_df in tqdm(grouped, desc="Creating sequences"):
        X_seq, X_static, Y = create_sequences_daily(
            group_df,
            seq_cols=model_config['SEQUENTIAL_FEATURES'],
            static_cols=model_config['STATIC_FEATURES'],
            target_col=model_config['TARGET_FEATURE_COLUMN'],
            feature_lag=model_config['FEATURE_LAG'],
            forecast_steps=model_config['FORECAST_STEPS']
        )
        
        if Y.shape[0] > 0:
            all_X_seq.append(X_seq)
            all_X_static.append(X_static)
            all_Y.append(Y)

    if not all_X_seq:
        raise ValueError("No sequences were created. Check data and config.")
        
    X_seq_all = np.concatenate(all_X_seq, axis=0)
    X_static_all = np.concatenate(all_X_static, axis=0)
    Y_all = np.concatenate(all_Y, axis=0)

    print(f"Total samples created: {X_seq_all.shape[0]}")

    # --- 3. Create Tensors ---
    dataset = TensorDataset(
        torch.tensor(X_seq_all, dtype=torch.float32),
        torch.tensor(X_static_all, dtype=torch.float32),
        torch.tensor(Y_all, dtype=torch.float32)
    )
    
    return dataset, norm_meta

# --- NEW: Evaluation Function ---
def evaluate(model, dataloader, criterion, device):
    """Calculates validation loss and MAE."""
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad(): # No gradients needed
        for X_seq_batch, X_static_batch, Y_batch in dataloader:
            X_seq_batch = X_seq_batch.to(device)
            X_static_batch = X_static_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            output = model(X_seq_batch, X_static_batch)
            loss = criterion(output, Y_batch)
            total_loss += loss.item()
            
            # Store preds and targets for MAE
            all_preds.append(output.cpu().numpy())
            all_targets.append(Y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate MAE on all validation data
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    # We use flatten() because MAE doesn't care about the 7-day structure
    mae = mean_absolute_error(all_targets.flatten(), all_preds.flatten())
    
    return avg_loss, mae

# --- UPDATED: Training Loop ---
def train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
    """
    Updated training loop that reports validation metrics.
    """
    for epoch in range(1, epochs + 1):
        model.train() # Set to train mode
        epoch_loss = 0.0
        
        # Training Phase
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{epochs}] Training")
        for X_seq_batch, X_static_batch, Y_batch in pbar:
            X_seq_batch, X_static_batch, Y_batch = (
                X_seq_batch.to(device), X_static_batch.to(device), Y_batch.to(device)
            )
            optimizer.zero_grad()
            output = model(X_seq_batch, X_static_batch)
            loss = criterion(output, Y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation Phase
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        
        # Step the scheduler on the validation loss
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"[Epoch {epoch}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss (MSE): {val_loss:.4f}, "
              f"Val MAE: {val_mae:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config_m5.yaml', help='Path to data config file')
    parser.add_argument('--model_config', default='config/model_config_m5.yaml', help='Path to model config file')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)

    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])

    # --- 1. Load and Preprocess Data ---
    processed_path = config['output']['processed_data_path']
    if os.path.exists(processed_path):
        print(f"Loading preprocessed data from {processed_path}...")
        df = pd.read_pickle(processed_path)
    else:
        print("No preprocessed file found. Running preprocessor...")
        df, _ = load_and_preprocess_daily(config)
    
    # --- 2. Split Data (Train/Val) ---
    # val_days = model_config['FORECAST_STEPS']
    val_days = 180. ### -----check it 
    df['day_num'] = df.groupby('id')['sales'].cumcount()
    max_days = df.groupby('id')['day_num'].max()
    
    val_indices = df.apply(lambda row: row['day_num'] > max_days[row['id']] - val_days, axis=1)
    train_indices = df.apply(
        lambda row: row['day_num'] <= max_days[row['id']] - val_days and 
                    row['day_num'] >= model_config['FEATURE_LAG'], 
        axis=1
    )
    
    train_df = df[train_indices].copy()
    # We need to get the history for the validation set to create sequences
    val_history_df = df[df['day_num'] > max_days.min() - val_days - model_config['FEATURE_LAG']].copy()

    print(f"Train data shape: {train_df.shape}")
    print(f"Validation data shape: {val_history_df.shape} (includes history for seq creation)")

    # --- 3. Build Datasets ---
    train_dataset, norm_meta = build_dataset(train_df, model_config, norm_meta=None)
    val_dataset, _ = build_dataset(val_history_df, model_config, norm_meta=norm_meta)

    train_loader = DataLoader(train_dataset, batch_size=model_config['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=model_config['BATCH_SIZE'], shuffle=False)

    # --- 4. Initialize Model and Training ---
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    params = dict(model_config["MODEL_PARAMS"])
    params["device"] = str(device) 
    model = ForecastingModel(**params).to(device)

    criterion = nn.MSELoss()
    # criterion = nn.L1Loss() # <-- NEW: MAE is less sensitive to spikes
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config["LEARNING_RATE"])
    # Tell scheduler to monitor the 'min' value of the validation loss
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    print("[INFO] Starting model training...")
    # --- UPDATED: Pass val_loader to the training loop ---
    train_loop(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        model_config["EPOCHS"], 
        device
    )

    # --- 5. Save Model and Metadata ---
    os.makedirs(os.path.dirname(model_config['CHECKPOINT_PATH']), exist_ok=True)
    
    torch.save(model.state_dict(), model_config['CHECKPOINT_PATH'])

    metadata = {
        "model_params": params,
        "normalization": norm_meta,
        "model_config": model_config,
        "data_config": config
    }
    with open(model_config['METADATA_PATH'], "wb") as f:
        pickle.dump(metadata, f)
        
    print(f"\n[INFO] Saved model and metadata to {model_config['CHECKPOINT_PATH']}, {model_config['METADATA_PATH']}")

if __name__ == "__main__":
    main()