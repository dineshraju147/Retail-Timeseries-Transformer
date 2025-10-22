# preprocess_m5_for_transformer.py
import pandas as pd

def preprocess_m5(csv_path, output_csv_path="data/m5_weekly_processed.csv"):
    print("[INFO] Loading dataset...")
    df = pd.read_csv(csv_path)
    print("[INFO] Initial shape:", df.shape)

    df = df.sort_values(["store_id_encoded", "item_id_encoded", "week_num_global"]).reset_index(drop=True)

    # Select columns needed for transformer
    df = df[[
        "sales", "sell_price", "store_id_encoded", "item_id_encoded", "week_num_global"
    ]].dropna().reset_index(drop=True)

    print("[INFO] Prepared M5 dataset with relevant columns.")
    print(df.head())
    df.to_csv(output_csv_path, index=False)
    print(f"[INFO] Saved processed dataset to {output_csv_path}")

if __name__ == "__main__":
    preprocess_m5("data/m5_weekly.csv")  # change to your file path
