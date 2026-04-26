import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from tqdm import tqdm

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    from joblib import Parallel, delayed
    HAS_FASTDTW = True
except ImportError:
    HAS_FASTDTW = False


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
INPUT_LAND = ROOT / "1km_static_postprocessed_new.csv"
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
    "grid_id", "id", "meo_id", "time", "egg_num", "previous_egg", "next_egg",
    "x_min", "x_max", "y_min", "y_max", "x_center", "y_center",
    "lat", "lng", "meo_lat", "meo_lng", "trap_num", "trap_id", "meo_grid"
}
# =================================================

def calculate_idw_features(df, k=10):
    """
    計算基於 previous_egg 的優化版 IDW 特徵
    """
    print("Calculating IDW features using previous_egg...")
    df = df.copy()
    df = df.sort_values(['id', 'time']).reset_index(drop=True)
    if 'previous_egg' not in df.columns:
        df['previous_egg'] = df.groupby('id')['egg_num'].shift(1).fillna(0)
    
    for col in ['idw_p1', 'idw_p2', 'idw_p05']:
        df[col] = 0.0
        
    times = df['time'].unique()
    global_mean = np.log1p(df['previous_egg']).mean()
    
    for t in tqdm(times, desc="IDW Features"):
        mask_t = df['time'] == t
        df_t = df[mask_t]
        
        if len(df_t) < 2:
            val = np.log1p(df_t['previous_egg']).mean() if not df_t.empty else global_mean
            df.loc[mask_t, ['idw_p1', 'idw_p2', 'idw_p05']] = val
            continue
            
        coords = df_t[['x_center', 'y_center']].values
        vals = np.log1p(df_t['previous_egg'].values)
        
        knn = NearestNeighbors(n_neighbors=min(len(df_t), k+1), metric='euclidean')
        knn.fit(coords)
        dists, idxs = knn.kneighbors(coords)
        
        for i in range(len(df_t)):
            d = dists[i, 1:] + 1e-6
            v = vals[idxs[i, 1:]]
            if len(d) == 0:
                iv = vals[i]
                df.loc[df_t.index[i], ['idw_p1', 'idw_p2', 'idw_p05']] = iv
            else:
                w1 = 1.0/d; w2 = 1.0/(d**2); w05 = 1.0/np.sqrt(d)
                df.at[df_t.index[i], 'idw_p1'] = np.sum(w1*v)/np.sum(w1)
                df.at[df_t.index[i], 'idw_p2'] = np.sum(w2*v)/np.sum(w2)
                df.at[df_t.index[i], 'idw_p05'] = np.sum(w05*v)/np.sum(w05)
                
    df[['idw_p1', 'idw_p2', 'idw_p05']] = df[['idw_p1', 'idw_p2', 'idw_p05']].fillna(global_mean)
    return df

def generate_lgbm_oof_predictions(df):
    """
    使用 5-Fold Cross Validation 生成 Out-Of-Fold (OOF) LGBM 預測值
    """
    print("Generating Out-of-Fold LGBM Predictions...")
    df['log_target'] = np.log1p(df['egg_num'].fillna(0))
    df['lgbm_pred'] = 0.0
    
    exclude_cols = ['id', 'grid_id', 'trap_id', 'time', 'egg_num', 'next_egg', 'log_target']
    features = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors='ignore')
    features = features.select_dtypes(include=[np.number]).fillna(0)
    
    if 'time' in df.columns:
        m = (df['time'] % 100)
        features['m_sin'] = np.sin(2*np.pi*m/12)
        features['m_cos'] = np.cos(2*np.pi*m/12)
        
    X = features.values
    y = df['log_target'].values
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        
        m_lgb = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, random_state=42, verbosity=-1)
        m_lgb.fit(X_tr, y_tr)
        
        p_va = np.expm1(np.clip(m_lgb.predict(X_va), 0, 10))
        df.loc[val_idx, 'lgbm_pred'] = p_va
        
        fold_mae = mean_absolute_error(np.expm1(y_va), p_va)
        maes.append(fold_mae)
        print(f"  Fold {fold+1} MAE: {fold_mae:.2f}")
        
    print(f"Overall LGBM OOF MAE: {np.mean(maes):.2f}")
    df = df.drop(columns=['log_target'])
    return df

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
    
    # --- LGBM + IDW Feature Augmentation ---
    df_merged = calculate_idw_features(df_merged, k=10)
    df_merged = generate_lgbm_oof_predictions(df_merged)
    # ---------------------------------------
    
    # Save the Master CSV
    master_csv_path = OUTPUT_DIR / "all_processed_data.csv"
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
    feature_cols = sorted(list(all_cols - EXCLUDE_COLS))
    
    # Separate into Static (Land) and Dynamic (Weather) roughly
    st_cols = sorted([c for c in df_land.columns if c != 'grid_id'])
    
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
    
    # -----------------------------------------------------
    # Calculate Temporal Adjacency (FastDTW)
    # -----------------------------------------------------
    adj_temporal_global = np.zeros((N, N), dtype=np.float32)
    
    if HAS_FASTDTW:
        print("Pre-calculating Temporal Adjacency (FastDTW)...")
        dyn_cols = sorted(list(set(feature_cols) - set(st_cols)))
        T_len = len(times)
        node_ts = np.zeros((N, T_len, len(dyn_cols)), dtype=np.float32)
        
        # Populate history
        for t_idx, t in enumerate(times):
            df_t = df_merged[df_merged['time'] == t].set_index('id').reindex(node_order)
            node_ts[:, t_idx, :] = df_t[dyn_cols].fillna(0).to_numpy().astype(np.float32)
        
        # Scale to avoid huge DTW values
        for d in range(len(dyn_cols)):
            mean_v = np.mean(node_ts[:, :, d])
            std_v = np.std(node_ts[:, :, d]) + 1e-8
            node_ts[:, :, d] = (node_ts[:, :, d] - mean_v) / std_v

        def compute_dtw(i):
            dists = np.zeros(N)
            for j in range(i+1, N):
                dist, _ = fastdtw(node_ts[i], node_ts[j], dist=euclidean)
                dists[j] = dist
            return i, dists
            
        print(f"Running DTW for {N} nodes (This may take a while)...")
        results = Parallel(n_jobs=-1)(delayed(compute_dtw)(i) for i in tqdm(range(N), desc="DTW Progress"))
        for i, dists in results:
            for j in range(i+1, N):
                adj_temporal_global[i, j] = dists[j]
                adj_temporal_global[j, i] = dists[j]
                
        # Convert distance to similarity (Gaussian kernel)
        sigma = np.std(adj_temporal_global[adj_temporal_global > 0])
        adj_temporal_global = np.exp(-(adj_temporal_global ** 2) / (sigma ** 2 + 1e-6))
        np.fill_diagonal(adj_temporal_global, 1.0)
    else:
        print("FastDTW or Joblib not found. Falling back to Spatial Distance for Temporal Adjacency...")
        print("Install with: pip install fastdtw joblib scipy")
        adj_temporal_global = adj_spatial.copy()
    # -----------------------------------------------------

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
        adj_temporal = adj_temporal_global.copy()

        
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