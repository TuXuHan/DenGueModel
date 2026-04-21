from pathlib import Path
from typing import List, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config as cfg


def load_split_ids(cv_k: int, data_root: Path) -> Tuple[List[int], List[int], List[int]]:
    """Load train/valid/test ids according to cv_k split definition."""
    split_id_list: List[List[int]] = []
    # Load all 10 split files
    for i in range(10):
        split_path = data_root / f"unlabeled_split_{i}.txt"
        if split_path.exists():
            lines = split_path.read_text(encoding="utf-8").strip().splitlines()
            gid = [int(x) for x in lines if x]
        else:
            gid = []
        split_id_list.append(gid)

    k_val = (cv_k + 8) % 10
    k_test = (cv_k + 9) % 10
    
    if cv_k >= 2:
        tmp = split_id_list[cv_k:] + split_id_list[: (cv_k - 2) % 10]
    else:
        tmp = split_id_list[cv_k : cv_k + 8]

    train_id = list(np.concatenate(tmp).flat) if tmp else []
    valid_id = split_id_list[k_val]
    test_id = split_id_list[k_test]
    return train_id, valid_id, test_id


def load_infer_ids(data_root: Path) -> List[int]:
    """Load all unlabeled ids for inference."""
    infer_path = data_root / "unlabeled_infer_id.txt"
    if infer_path.exists():
        lines = infer_path.read_text(encoding="utf-8").strip().splitlines()
        return [int(x) for x in lines if x]
    return []


class StationDataset(Dataset):
    """
    Optimized Dataset wrapper. 
    Removes Pandas from __getitem__ loop by pre-converting to NumPy and Dictionaries.
    """

    def __init__(
        self,
        mode: str = "train",
        cv_k: int = 0,
        data_root: Path = cfg.DATA_DIR,
        k_neighbor: int = cfg.K_NEIGHBOR,
        prev_slot: int = cfg.HISTORICAL_T,
        pred_slot: int = cfg.MODEL_OUTPUT_SIZE,
        drop_no_target: bool = True,
        split_by_time: bool = cfg.SPLIT_BY_TIME,
        train_ratio: float = cfg.TRAIN_RATIO,
        val_ratio: float = cfg.VAL_RATIO,
        test_ratio: float = cfg.TEST_RATIO,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.k_neighbor = k_neighbor
        self.prev_slot = prev_slot
        self.pred_slot = pred_slot
        self.split_by_time = split_by_time
        
        # --- 1. Load Split IDs ---
        # Determine which IDs belong to this dataset mode
        if mode == "infer":
            self.target_ids = set(load_infer_ids(self.data_root))
        else:
            train_id, valid_id, test_id = load_split_ids(cv_k, self.data_root)
            if mode == "train":
                self.target_ids = set(train_id)
            elif mode == "valid":
                self.target_ids = set(valid_id)
            else:
                self.target_ids = set(test_id)

        # --- 2. Load Master Data ---
        print(f"Loading Master Data for {mode}...")
        df_all = pd.read_csv(self.data_root / "all_processed_data_9box_nexty.csv")
        
        # Sort is CRITICAL for history slicing [idx - prev : idx]
        # We assume rows are contiguous in time for each station
        df_all = df_all.sort_values(["id", "time"]).reset_index(drop=True)

        # --- 3. Pre-process to NumPy (The Speed Fix) ---
        print("Converting to NumPy & Building Index Maps...")
        
        # A. Global Arrays (Holds data for ALL stations, neighbors included)
        self.all_ids = df_all["id"].values.astype(int)
        self.all_times = df_all["time"].values.astype(int)
        self.all_meo = df_all[cfg.MEO_COL].values.astype(np.float32)
        self.all_mask = (~np.isnan(df_all["egg_num"].values)).astype(np.float32)
        self.all_ovi = df_all["egg_num"].fillna(0).values.astype(np.float32)
        
        # B. Index Map: (grid_id, time) -> Row Index in Global Arrays
        # This turns O(N) filtering into O(1) lookup
        self.index_map = dict(zip(zip(self.all_ids, self.all_times), range(len(self.all_ids))))

        # C. Static Features Dict: grid_id -> numpy array
        # Drop duplicates to get unique static features per ID
        static_df = df_all[["id"] + cfg.ST_COL_ALL].drop_duplicates(subset=["id"])
        self.static_dict = {}
        for row in static_df.itertuples(index=False):
            # row[0] is id, row[1:] are features
            self.static_dict[row[0]] = np.array(row[1:], dtype=np.float32)

        # --- 4. Load Neighbors & Distances ---
        print("Loading Neighbor Maps...")
        grid_neighbor = pd.read_csv(self.data_root / "grid_100neighbor_dist.csv")
        
        nearest_cols = cfg.nearest_columns(k_neighbor)
        dist_cols = cfg.near_dist_columns(k_neighbor)
        
        self.neighbor_map = {} # id -> [neighbor_ids]
        self.dist_map = {}     # id -> (dists, inv_dists)
        
        for row in grid_neighbor.itertuples(index=False):
            # Identify columns by index or name. Using name logic via getattr is safest.
            # We assume grid_neighbor columns are: grid_id, nearest_1...10, nearest_dist_1...10
            # Let's map column names to values
            row_dict = row._asdict()
            gid = row_dict["grid_id"]
            
            # Neighbors
            nbrs = [row_dict[c] for c in nearest_cols]
            self.neighbor_map[gid] = np.array(nbrs, dtype=int)
            
            # Distances
            dists = np.array([row_dict[c] for c in dist_cols], dtype=np.float32)
            inv_dists = np.divide(1.0, dists, out=np.zeros_like(dists), where=dists!=0)
            self.dist_map[gid] = (dists, inv_dists)

        # --- 5. Define Samples for __getitem__ ---
        # We only iterate over rows that belong to the current mode (target_ids)
        # We store the *indices* of these rows in the Global Arrays
        id_mask = np.isin(self.all_ids, list(self.target_ids))

        split_mask = None
        if split_by_time and mode != "infer":
            unique_times = np.sort(np.unique(self.all_times))
            n_times = len(unique_times)
            n_train = int(n_times * train_ratio)
            n_val = int(n_times * val_ratio)
            if n_train <= 0 or n_val <= 0 or n_train + n_val >= n_times:
                raise ValueError("Invalid time split ratios; adjust TRAIN_RATIO/VAL_RATIO/TEST_RATIO.")
            train_end = unique_times[n_train - 1]
            val_end = unique_times[n_train + n_val - 1]
            if mode == "train":
                split_mask = self.all_times <= train_end
            elif mode == "valid":
                split_mask = (self.all_times > train_end) & (self.all_times <= val_end)
            else:
                split_mask = self.all_times > val_end
        else:
            split_mask = np.ones_like(self.all_times, dtype=bool)

        mask = id_mask & split_mask
        self.sample_indices = np.where(mask)[0]

        if mode == "infer":
            drop_no_target = False

        if drop_no_target and mode != "infer":
            valid_target = np.zeros(len(self.all_ids), dtype=bool)
            for step in range(1, self.pred_slot + 1):
                same_id = self.all_ids[:-step] == self.all_ids[step:]
                has_label = self.all_mask[step:] > 0
                in_split = split_mask[step:]
                valid_target[:-step] |= same_id & has_label & in_split

            before = len(self.sample_indices)
            self.sample_indices = self.sample_indices[valid_target[self.sample_indices]]
            dropped = before - len(self.sample_indices)
            if dropped > 0:
                print(f"Dataset {mode}: dropped {dropped} samples with no valid future target.")

        print(f"Dataset {mode} initialized: {len(self.sample_indices)} samples.")

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, index: int):
        # 1. Get Global Index for this sample
        global_idx = self.sample_indices[index]
        
        unlabel_id = self.all_ids[global_idx]
        timestamp = self.all_times[global_idx]

        # 2. Build Target Sequence (Future OVI)
        # Check next `pred_slot` rows in the global array.
        # Ensure they belong to the same ID.
        ovi_list = []
        mask_list = []
        for step in range(1, self.pred_slot + 1):
            next_idx = global_idx + step
            # Boundary check: must be within array and same ID
            if next_idx < len(self.all_ids) and self.all_ids[next_idx] == unlabel_id:
                raw_val = np.log1p(self.all_ovi[next_idx])
                val = max(0.0, raw_val)
                ovi_list.append(val)
                
                mask_val = self.all_mask[next_idx] 
                mask_list.append(mask_val)
            else:
                ovi_list.append(0.0)
                mask_list.append(0.0)
        ovi_target = torch.tensor(ovi_list).float()
        target_mask = torch.tensor(mask_list).float()
        # 3. Get History (MEO)
        # Slice [idx - prev : idx]
        start = global_idx - self.prev_slot
        end = global_idx
        
        # Handle edges (start < 0 or ID mismatch at start)
        # Fast path: if ID at start matches ID at end, we are safe (contiguous)
        if start >= 0 and self.all_ids[start] == unlabel_id:
            meo_slice = self.all_meo[start:end]
        else:
            # Slow path: padding required
            # Find how many valid steps we have backwards
            valid_len = 0
            for k in range(1, self.prev_slot + 1):
                curr = global_idx - k
                if curr >= 0 and self.all_ids[curr] == unlabel_id:
                    valid_len += 1
                else:
                    break
            
            # Slice valid part
            valid_slice = self.all_meo[global_idx - valid_len : global_idx]
            # Pad
            pad_len = self.prev_slot - valid_len
            pad = np.zeros((pad_len, valid_slice.shape[1]), dtype=np.float32)
            meo_slice = np.vstack([pad, valid_slice])

        meo_unlabel = torch.from_numpy(meo_slice.copy()).float()

        # 4. Unlabel Static Features
        if unlabel_id in self.static_dict:
            feat_unlabel = self.static_dict[unlabel_id]
        else:
            feat_unlabel = np.zeros(len(cfg.ST_COL_ALL), dtype=np.float32)
        feature_unlabel = torch.from_numpy(feat_unlabel.copy()).float()

        # 5. Get Neighbors Data
        # Retrieve neighbor IDs from pre-loaded map
        if unlabel_id in self.neighbor_map:
            label_id_list = self.neighbor_map[unlabel_id]
        else:
            label_id_list = [] # Should not happen if data is clean

        ovi_label, meo_label, feature_label = self.get_feats_label(label_id_list, timestamp)
        
        # 6. Get Distances
        if unlabel_id in self.dist_map:
            dists, inv_dists = self.dist_map[unlabel_id]
            inv_dis_label = torch.from_numpy(inv_dists.copy()).float()
        else:
            inv_dis_label = torch.zeros(self.k_neighbor).float()

        feature_label_out = torch.cat((feature_label, inv_dis_label.unsqueeze(1)), 1)

        # Note: label_id_list is numpy array, convert to tensor
        return (
            ovi_target,
            target_mask,
            meo_unlabel,
            feature_unlabel,
            ovi_label,
            meo_label,
            feature_label_out,
            inv_dis_label,
            torch.from_numpy(np.array(label_id_list)).float(),
            timestamp,
            unlabel_id
        )

    def get_feats_label(self, label_id_list: Sequence[int], timestamp: int):
        # Initialize Output Arrays
        # shape: (K, prev_slot, n_meo), (K, prev_slot), (K, n_static)
        meo_out = np.zeros((self.k_neighbor, self.prev_slot, self.all_meo.shape[1]), dtype=np.float32)
        ovi_out = np.zeros((self.k_neighbor, self.prev_slot), dtype=np.float32)
        feat_out = np.zeros((self.k_neighbor, len(cfg.ST_COL_ALL)), dtype=np.float32)

        for i, gid in enumerate(label_id_list):
            if i >= self.k_neighbor: break

            # A. Static Features (Dict Lookup)
            if gid in self.static_dict:
                feat_out[i] = self.static_dict[gid]

            # B. Dynamic History (Index Map Lookup)
            idx = self.index_map.get((gid, timestamp))
            
            if idx is not None:
                # Same logic as unlabel history: slice safely
                start = idx - self.prev_slot
                
                # Fast Check
                if start >= 0 and self.all_ids[start] == gid:
                    meo_out[i] = self.all_meo[start:idx]
                    ovi_out[i] = self.all_ovi[start:idx]
                else:
                    # Padding Logic
                    valid_len = 0
                    for k in range(1, self.prev_slot + 1):
                        curr = idx - k
                        if curr >= 0 and self.all_ids[curr] == gid:
                            valid_len += 1
                        else:
                            break
                    
                    if valid_len > 0:
                        # Fill the end of the buffer
                        meo_out[i, -valid_len:, :] = self.all_meo[idx - valid_len : idx]
                        ovi_out[i, -valid_len:] = self.all_ovi[idx - valid_len : idx]

        return (
            torch.from_numpy(ovi_out).float(),
            torch.from_numpy(meo_out).float(),
            torch.from_numpy(feat_out).float()
        )
