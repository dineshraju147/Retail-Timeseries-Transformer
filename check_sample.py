import pickle
import pandas as pd

print("Loading data/m5_daily_processed.pkl to find valid combinations...")

try:
    with open('data/m5_daily_processed.pkl', 'rb') as f:
        data_cache = pickle.load(f)
    
    df = data_cache['df']
    encoders = data_cache['encoders']
    
    store_encoder = encoders['store_id']
    item_encoder = encoders['item_id']

    # Get the first 10 unique combinations of *encoded* IDs
    valid_combos_encoded = df[['store_id_encoded', 'item_id_encoded']].drop_duplicates().head(10)
    
    print("\n--- Here are some valid combinations from your sample file: ---")
    
    # Convert them back to strings for the user
    for _, row in valid_combos_encoded.iterrows():
        store_str = store_encoder.inverse_transform([int(row['store_id_encoded'])])[0]
        item_str = item_encoder.inverse_transform([int(row['item_id_encoded'])])[0]
        print(f'  python predict_m5.py --store_id "{store_str}" --item_id "{item_str}" --forecast_months 3')
        
    print("\nUse one of the full commands above.")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Make sure 'data/m5_daily_processed.pkl' exists and is correct.")
    