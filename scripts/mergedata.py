import pandas as pd
import numpy as np
import os

# ================= CONFIGURATION =================
# Define your input filenames
SRC_DIR = "data"  # Change this if your files are in a different directory
FILE_WEATHER = os.path.join(SRC_DIR, "1km_dynamic_all_imputed.csv")
FILE_GRID = os.path.join(SRC_DIR, "1km_grid.csv")
FILE_EGG = os.path.join(SRC_DIR, "bucket_1km_egg_counts_filtered_reg.csv")
FILE_LAND = os.path.join(SRC_DIR, "grid_land_use.csv")

# Output filename
OUTPUT_FILE = os.path.join(SRC_DIR, "merged_dataset.csv")

# Time filtering (Optional: Set to None if you want all data)
TIME_LOWER_BOUND = 202301 
# =================================================

def main():
    print("Loading datasets...")
    
    # 1. Load Weather (Dynamic Data)
    # Expected cols: grid_id, week, [Weather Features...]
    df_weather = pd.read_csv(FILE_WEATHER)
    if 'week' in df_weather.columns:
        df_weather.rename(columns={'week': 'time'}, inplace=True)
    
    # 2. Load Grid Info (Coordinates)
    # Expected cols: grid_id, x_center, y_center, ...
    df_grid = pd.read_csv(FILE_GRID)
    
    # 3. Load Land Use (Static Features)
    # Expected cols: grid_id, Residential, Water, ...
    df_land = pd.read_csv(FILE_LAND)
    
    # 4. Load Egg Data (Target Label)
    # Expected cols: grid_id, time, egg_num
    df_egg = pd.read_csv(FILE_EGG)

    print("Merging data...")

    # --- Step A: Merge Static Info onto Weather ---
    # We maintain the "Time" dimension from weather
    
    # Select only necessary grid cols to avoid duplicates (e.g., if x_min exists in both)
    # We definitely need coordinates for the GNN
    grid_cols = ['grid_id', 'x_center', 'y_center'] 
    
    df_merged = df_weather.merge(df_grid[grid_cols], on='grid_id', how='left')
    
    # Merge Land Use
    df_merged = df_merged.merge(df_land, on='grid_id', how='left')

    # --- Step B: Merge Egg Labels ---
    # Merge on both ID and Time
    # Using 'left' join ensures we keep all weather rows even if there are no eggs recorded
    df_merged = df_merged.merge(df_egg[['grid_id', 'time', 'egg_num']], on=['grid_id', 'time'], how='left')

    # --- Step C: Post-Processing ---
    
    # 1. Fill missing egg counts with 0 
    # (Assumption: No record in the egg file means 0 eggs or no trap, treated as 0 for training)
    # df_merged['egg_num'] = df_merged['egg_num'].fillna(-1)

    # 2. Rename 'grid_id' to 'id' for model compatibility (Optional, but recommended for your GNN)
    # df_merged.rename(columns={'grid_id': 'id'}, inplace=True)

    # 3. Time Filtering (Remove old data if needed)
    if TIME_LOWER_BOUND:
        print(f"Filtering data since {TIME_LOWER_BOUND}...")
        df_merged = df_merged[df_merged['time'] >= TIME_LOWER_BOUND]

    # 4. Sort data (Crucial for LSTM/Time-series models)
    df_merged = df_merged.sort_values(by=['grid_id', 'time']).reset_index(drop=True)

    # --- Save ---
    print(f"Saving merged data to {OUTPUT_FILE}...")
    print(f"Final Shape: {df_merged.shape}")
    df_merged.to_csv(OUTPUT_FILE, index=False)
    print("Done!")

if __name__ == "__main__":
    main()