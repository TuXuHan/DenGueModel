import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ================= CONFIGURATION =================
# Define your input paths here
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Input Data Files
INPUT_DIR = ROOT / "data"
INPUT_WEATHER = INPUT_DIR / "1km_dynamic_all_imputed.csv"
INPUT_GRID = INPUT_DIR / "1km_grid.csv"
INPUT_EGG = INPUT_DIR / "bucket_1km_egg_counts_filtered_reg.csv"
INPUT_LAND = INPUT_DIR / "grid_land_use.csv"
# Output Directory
OUTPUT_DIR = ROOT / "dataset"

# Parameters
TARGET_RATIO = 0.5
K_NEIGHBOR = 10
TIME_LOWER_BOUND = 202301
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

# Columns to Exclude from Static Features (IDs, Coordinates, etc.)
EXCLUDE_COLS = {
    "grid_id", "id", "meo_id", "time", "egg_num", 
    "x_min", "x_max", "y_min", "y_max", "x_center", "y_center",
    "lat", "lng", "meo_lat", "meo_lng", "trap_num", "trap_id"
}
# =================================================

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_neighbor", type=int, default=K_NEIGHBOR, help="Number of neighbors for KNN")
    parser.add_argument("--time_lower_bound", type=int, default=TIME_LOWER_BOUND, help="Lower bound for time filtering")
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO, help="Ratio for validation set")
    parser.add_argument("--test_ratio", type=float, default=TEST_RATIO, help="Ratio for test set")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    return parser

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Load and Merge Data ---
    print("Loading data files...")
    df_weather = pd.read_csv(INPUT_WEATHER)
    df_grid = pd.read_csv(INPUT_GRID)
    df_egg = pd.read_csv(INPUT_EGG)
    df_land = pd.read_csv(INPUT_LAND)

    print("Merging data...")
    if 'week' in df_weather.columns:
        df_weather = df_weather.rename(columns={'week': 'time'})
    
    df_merged = df_weather.merge(df_grid[['grid_id', 'x_center', 'y_center']], on='grid_id', how='left')
    
    df_merged = df_merged.merge(df_land, on='grid_id', how='left')
    
    df_merged = df_merged.merge(df_egg[['grid_id', 'time', 'egg_num']], on=['grid_id', 'time'], how='left')
    
    # Rename grid_id to 'id' for compatibility with model code
    df_merged = df_merged.rename(columns={'grid_id': 'id'})
    
    # Filter time
    df_merged = df_merged[df_merged['time'] >= TIME_LOWER_BOUND].copy()
    
    # Sort
    df_merged = df_merged.sort_values(['id', 'time']).reset_index(drop=True)
    
    # Save the Master CSV
    master_csv_path = OUTPUT_DIR / "all_processed_data_9box_nexty.csv"
    print(f"Saving master CSV to {master_csv_path}...")
    df_merged.to_csv(master_csv_path, index=False)

    # Generate Label IDs (Anchors)
    print("Generating label_id.txt...")
    label_ids = df_egg['grid_id'].unique()
    label_ids = np.sort(label_ids)
    
    with open(OUTPUT_DIR / "label_id.txt", "w") as f:
        f.write("\n".join(map(str, label_ids)))

    # Automatic Splitting (The Optimization)
    print(f"Generating Stratified Splits (Target Ratio: {TARGET_RATIO})...")

    TRAIN_RATIO = 1 - VAL_RATIO - TEST_RATIO

    grid_activity = df_merged.groupby('id')['egg_num'].sum().reset_index()
    grid_activity.columns = ['id', 'total_eggs']

    all_ids = grid_activity['id'].values.copy()
    np.random.seed(SEED)
    np.random.shuffle(all_ids)

    n_total = len(all_ids)
    n_targets = int(n_total * TARGET_RATIO)
    
    target_ids = all_ids[:n_targets]
    anchor_ids = all_ids[n_targets:]

    target_activity = grid_activity[grid_activity['id'].isin(target_ids)].copy()

    # Create Bins for Stratification
    bins = [-1, 0, 50, 200, 1000, 999999]
    labels = [0, 1, 2, 3, 4]
    
    target_activity['activity_bin'] = pd.cut(
        target_activity['total_eggs'], 
        bins=bins, 
        labels=labels
    )


    try:
        X_train, X_temp, _, y_temp = train_test_split(
            target_activity['id'].values, 
            target_activity['activity_bin'].values,
            train_size=TRAIN_RATIO,
            random_state=SEED,
            stratify=target_activity['activity_bin'].values 
        )
    except ValueError as e:
        # Fallback if even the 80/20 split fails (e.g. only 1 extreme case total)
        print(f"Warning: Strict stratification failed ({e}). Falling back to random split for Train.")
        X_train, X_temp, _, y_temp = train_test_split(
            target_activity['id'].values, 
            target_activity['activity_bin'].values,
            train_size=TRAIN_RATIO,
            random_state=SEED,
            stratify=None
        )

    X_val, X_test, _, _ = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5, 
        random_state=SEED,
        stratify=None
    )
    
    train_ids = X_train
    val_ids = X_val
    test_ids = X_test

    print(f"Total Grids: {n_total}")
    print(f"  Targets (Predicted): {len(target_ids)}")
    print(f"    - Train: {len(train_ids)}")
    print(f"    - Val:   {len(val_ids)}")
    print(f"    - Test:  {len(test_ids)}")
    print(f"  Anchors (Context):   {len(anchor_ids)}")

    # Save Splits
    splits = [train_ids] * 8 + [val_ids] + [test_ids]
    for i, ids in enumerate(splits):
        with open(OUTPUT_DIR / f"unlabeled_split_{i}.txt", "w") as f:
            f.write("\n".join(map(str, ids)))

    # Generate Neighbor Graph (KNN)
    print("Generating neighbor graph...")
    # Unique coordinates for each ID
    coords = df_merged[['id', 'x_center', 'y_center']].drop_duplicates().sort_values('id')
    
    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBOR + 1, metric="euclidean").fit(coords[['x_center', 'y_center']])
    dists, idxs = nbrs.kneighbors(coords[['x_center', 'y_center']])
    
    # Remove self (first column)
    dists = dists[:, 1:]
    idxs = idxs[:, 1:]
    
    # Map indices back to Grid IDs
    id_array = coords['id'].to_numpy()
    nearest_ids = id_array[idxs]
    
    knn_df = pd.DataFrame({'grid_id': id_array})
    for j in range(K_NEIGHBOR):
        knn_df[f'nearest_{j+1}'] = nearest_ids[:, j]
        knn_df[f'nearest_dist_{j+1}'] = dists[:, j]
        
    knn_df.to_csv(OUTPUT_DIR / "grid_100neighbor_dist.csv", index=False)

    # Generate GNN Graphs
    print("Generating GNN graph snapshots...")
    graph_dir = OUTPUT_DIR / "graph_data"
    if graph_dir.exists(): shutil.rmtree(graph_dir)
    
    for sub in ["adj_spatial_dist", "adj_spatial_cluster", "adj_temporal", "feat"]:
        (graph_dir / sub).mkdir(parents=True, exist_ok=True)
        
    # Save Node IDs order
    node_order = coords['id'].to_numpy()
    np.save(graph_dir / "all_node_id.npy", node_order)
    
    # Identify Feature Columns
    all_cols = set(df_merged.columns)
    feature_cols = list(all_cols - EXCLUDE_COLS)
    
    # Separate into Static (Land) and Dynamic (Weather) roughly
    st_cols = [c for c in df_land.columns if c != 'grid_id']
    
    print(f"Static Features (Cluster): {len(st_cols)}")
    print(f"Total Features (Input): {len(feature_cols)}")

    # Pre-calculate Spatial Adjacency (Static)
    # Matrix size: N x N
    N = len(node_order)
    adj_spatial = np.zeros((N, N), dtype=np.float32)
    
    # Map grid_id to matrix index 0..N-1
    id_to_idx = {grid_id: i for i, grid_id in enumerate(node_order)}
    
    for _, row in knn_df.iterrows():
        src_idx = id_to_idx[row['grid_id']]
        for k in range(1, K_NEIGHBOR + 1):
            neighbor_id = row[f'nearest_{k}']
            if neighbor_id in id_to_idx:
                dst_idx = id_to_idx[neighbor_id]
                adj_spatial[src_idx, dst_idx] = 1.0
                adj_spatial[dst_idx, src_idx] = 1.0

    times = np.sort(df_merged['time'].unique())
    
    for t in tqdm(times, desc="Building Graphs"):
        # Extract data for this time step
        df_t = df_merged[df_merged['time'] == t].set_index('id').reindex(node_order)
        
        # Feature Matrix
        feat_data = df_t[feature_cols].fillna(0).to_numpy().astype(np.float32)
        
        # Cluster Adjacency
        st_data = df_t[st_cols].fillna(0).to_numpy().astype(np.float32)
        norm = np.linalg.norm(st_data, axis=1, keepdims=True) + 1e-6
        st_norm = st_data / norm
        adj_cluster = np.matmul(st_norm, st_norm.T)
        
        # Temporal Adjacency
        adj_temporal = adj_spatial.copy()
        
        # Save
        np.save(graph_dir / "adj_spatial_dist" / f"{t}.npy", adj_spatial)
        np.save(graph_dir / "adj_spatial_cluster" / f"{t}.npy", adj_cluster)
        np.save(graph_dir / "adj_temporal" / f"{t}.npy", adj_temporal)
        np.save(graph_dir / "feat" / f"{t}.npy", feat_data)

    print("Done! All data processed.")

if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    K_NEIGHBOR = args.k_neighbor
    TIME_LOWER_BOUND = args.time_lower_bound
    VAL_RATIO = args.val_ratio
    TEST_RATIO = args.test_ratio
    SEED = args.seed
    main()