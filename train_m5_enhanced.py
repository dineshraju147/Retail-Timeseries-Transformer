# train_m5_enhanced.py
"""
Enhanced training script with:
- Quantile loss
- Gradient clipping
- Early stopping
- Better validation metrics
- Learning rate warmup
"""

import os
import pickle
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from typing import Dict, Tuple
from sklearn.metrics import mean_absolute_error

# Import enhanced model
from models.transformer_enhanced_m5 import (
    EnhancedForecastingModel, 
    QuantileLoss, 
    compute_metrics
)
from pre_processing.preprocessor import (
    load_and_preprocess_daily, 
    create_sequences_daily, 
    normalize_data
)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=300, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def build_dataset(
    df: pd.DataFrame,
    model_config: Dict,
    norm_meta: Dict
) -> Tuple[TensorDataset, Dict]:
    """Builds a TensorDataset from the preprocessed DataFrame."""
    
    print("Building dataset...")
    
    # Normalize
    # norm_cols = model_config['SEQUENTIAL_FEATURES'] + model_config['STATIC_FEATURES']
    norm_cols = model_config['SEQUENTIAL_FEATURES'] 
    df, norm_meta = normalize_data(df, norm_cols, norm_meta)

    # Create sequences
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
        raise ValueError("No sequences were created.")
        
    X_seq_all = np.concatenate(all_X_seq, axis=0)
    X_static_all = np.concatenate(all_X_static, axis=0)
    Y_all = np.concatenate(all_Y, axis=0)

    print(f"Total samples: {X_seq_all.shape[0]}")

    dataset = TensorDataset(
        torch.tensor(X_seq_all, dtype=torch.float32),
        torch.tensor(X_static_all, dtype=torch.float32),
        torch.tensor(Y_all, dtype=torch.float32)
    )
    
    return dataset, norm_meta


def evaluate(model, dataloader, criterion, device, quantile_mode=False):
    """Enhanced evaluation with multiple metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_seq_batch, X_static_batch, Y_batch in dataloader:
            X_seq_batch = X_seq_batch.to(device)
            X_static_batch = X_static_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            output = model(X_seq_batch, X_static_batch)
            loss = criterion(output, Y_batch)
            total_loss += loss.item()
            
            all_preds.append(output.cpu())
            all_targets.append(Y_batch.cpu())
    
    avg_loss = total_loss / len(dataloader)
    
    # Compute metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_preds, all_targets, quantile_mode)
    
    return avg_loss, metrics


def train_loop(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    scheduler, 
    epochs, 
    device,
    checkpoint_path,
    quantile_mode=False,
    early_stopping=None
):
    """Enhanced training loop with gradient clipping and early stopping."""
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{epochs}] Training")
        for X_seq_batch, X_static_batch, Y_batch in pbar:
            X_seq_batch = X_seq_batch.to(device)
            X_static_batch = X_static_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_seq_batch, X_static_batch)
            loss = criterion(output, Y_batch)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device, quantile_mode
        )
        
        # Step scheduler (OneCycleLR steps per batch, not per epoch)
        if isinstance(scheduler, OneCycleLR):
            pass  # OneCycleLR is stepped in the training loop
        else:
            scheduler.step(val_loss)
        
        print(f"[Epoch {epoch}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val MAE: {val_metrics['mae']:.4f}, "
              f"Val RMSE: {val_metrics['rmse']:.4f}, "
              f"Val MAPE: {val_metrics['mape']:.2f}%, "
              f"Bias: {val_metrics['bias']:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Saved best model (Val Loss: {val_loss:.4f})")
        
        # Early stopping
        if early_stopping:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"\n⚠ Early stopping triggered at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"\n✓ Loaded best model from {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config_m5.yaml')
    parser.add_argument('--model_config', default='config/model_config_enhanced_m5.yaml')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)

    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])

    # --- 1. Load Data ---
    processed_path = config['output']['processed_data_path']
    if os.path.exists(processed_path):
        print(f"Loading preprocessed data from {processed_path}...")
        df = pd.read_pickle(processed_path)
    else:
        print("Running preprocessor...")
        df, _ = load_and_preprocess_daily(config)
    
    # --- 2. Train/Val Split ---
    val_days = model_config.get('VAL_DAYS', 180)
    df['day_num'] = df.groupby('id')['sales'].cumcount()
    max_days = df.groupby('id')['day_num'].max()
    
    val_indices = df.apply(
        lambda row: row['day_num'] > max_days[row['id']] - val_days, 
        axis=1
    )
    train_indices = df.apply(
        lambda row: row['day_num'] <= max_days[row['id']] - val_days and 
                    row['day_num'] >= model_config['FEATURE_LAG'], 
        axis=1
    )
    
    train_df = df[train_indices].copy()
    val_history_df = df[df['day_num'] > max_days.min() - val_days - model_config['FEATURE_LAG']].copy()

    print(f"Train samples: {train_df.shape[0]}")
    print(f"Val samples: {val_history_df.shape[0]}")

    # --- 3. Build Datasets ---
    train_dataset, norm_meta = build_dataset(train_df, model_config, norm_meta=None)
    val_dataset, _ = build_dataset(val_history_df, model_config, norm_meta=norm_meta)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=model_config['BATCH_SIZE'], 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=model_config['BATCH_SIZE'], 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # --- 4. Initialize Model ---
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    print(f"[INFO] Using device: {device}")

    params = dict(model_config["MODEL_PARAMS"])
    params["device"] = str(device)
    
    # Add cardinality info for embeddings
    params["num_items"] = df['item_id_encoded'].nunique()
    params["num_stores"] = df['store_id_encoded'].nunique()
    params["num_depts"] = df['dept_id_encoded'].nunique()
    params["num_cats"] = df['cat_id_encoded'].nunique()
    
    use_quantile = model_config.get('USE_QUANTILE_LOSS', True)
    params["use_quantile"] = use_quantile
    
    model = EnhancedForecastingModel(**params).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- 5. Loss and Optimizer ---
    if use_quantile:
        criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    else:
        criterion = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=model_config["LEARNING_RATE"],
        weight_decay=1e-4  # L2 regularization
    )
    
    # OneCycleLR for better convergence
    total_steps = len(train_loader) * model_config["EPOCHS"]
    scheduler = OneCycleLR(
        optimizer,
        max_lr=model_config["LEARNING_RATE"],
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=model_config.get('EARLY_STOPPING_PATIENCE', 300),
        min_delta=0.001
    )

    # --- 6. Train ---
    print("\n[INFO] Starting training...")
    train_loop(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        model_config["EPOCHS"], 
        device,
        model_config['CHECKPOINT_PATH'],
        quantile_mode=use_quantile,
        early_stopping=early_stopping
    )

    # --- 7. Save Metadata ---
    os.makedirs(os.path.dirname(model_config['CHECKPOINT_PATH']), exist_ok=True)
    
    metadata = {
        "model_params": params,
        "normalization": norm_meta,
        "model_config": model_config,
        "data_config": config,
        "use_quantile": use_quantile
    }
    with open(model_config['METADATA_PATH'], "wb") as f:
        pickle.dump(metadata, f)
        
    print(f"\n✓ Training complete! Saved to {model_config['CHECKPOINT_PATH']}")


if __name__ == "__main__":
    main()