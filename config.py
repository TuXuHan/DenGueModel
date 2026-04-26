from pathlib import Path

import pandas as pd
import torch

# Data paths
DATA_DIR = Path("dataset")
GRAPH_PATH = DATA_DIR / "graph_data"

# Graph settings
VIEW_NUM = 4
FUSE_ADJ_METHOD = "add"  # options: add/cat

# Model + data settings
K_NEIGHBOR = 10  # for IDW attention: number of nearest labeled nodes to attend to, the number of K should less than 10
HISTORICAL_T = 4
MODEL_OUTPUT_SIZE = 1
ADD_LABELED_EMBED = False
ALPHA_MULTIVIEW_FUSION = 0.3

# Training hyperparameters
LR = 0.0005
BATCH_SIZE = 12
MAX_EPOCH = 20
PATIENCE = 5
RANDOM_SEED = 99
SHUFFLE = True
WEIGHT_DECAY = 0.01

# Split control.
# When True, ignore ID split files and split chronologically by time. 
SPLIT_BY_TIME = False
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Force CPU to avoid CUDA capability mismatch on local GPU
# Force CPU to avoid CUDA capability mismatch on local GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_DIR = Path("log")
MODEL_DIR = Path("checkpoints")
FIG_DIR = Path("fig")

# New dataset dynamic columns (1km 全台南)
MEO_COL = [
    "StationPressureMean",
    "AirTemperatureMean",
    "RelativeHumidityMean",
    "WindSpeedMean",
    "PrecipitationAccumulation",
    "StationPressureMaximum",
    "AirTemperatureMaximum",
    "RelativeHumidityMaximum",
    "PeakGustMaximum",
    "StationPressureMinimum",
    "AirTemperatureMinimum",
    "RelativeHumidityMinimum",
    "idw_p1",
    "idw_p2",
    "idw_p05",
    "lgbm_pred"
]

# If processed data exists, infer static columns from file; otherwise fall back to legacy definition.
ST_COL = [
    "watersupply_hole",
    "well",
    "sewage_hole",
    "underwater_con",
    "pumping",
    "watersupply_others",
    "watersupply_value",
    "food_poi",
    "rainwater_hole",
    "river",
    "drainname",
    "sewage_well",
    "gaugingstation",
    "underpass",
    "watersupply_firehydrant",
]
DIRS = [] # Cancelled 9-box setting
def build_static_columns_from_dirs():
    return ST_COL.copy()


def infer_static_columns_from_data():
    sample_file = DATA_DIR / "all_processed_data.csv"
    if not sample_file.exists():
        return None
    df = pd.read_csv(sample_file, nrows=1)
    exclude = {
        "id",
        "grid_id",
        "trap_id",
        "time",
        "egg_num",
        "previous_egg",
        "next_egg",
        "meo_id",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "x_center",
        "y_center",
        "meo_grid",
        "meo_lng",
        "meo_lat",
    }
    exclude.update(MEO_COL)
    static_cols = [c for c in df.columns if c not in exclude]
    return static_cols


ST_COL_ALL = infer_static_columns_from_data() or build_static_columns_from_dirs()
NODE_FEAT_DIM = len(ST_COL_ALL) + len(MEO_COL)


def nearest_columns(k: int):
    return [f"nearest_{i+1}" for i in range(k)]


def near_dist_columns(k: int):
    return [f"nearest_dist_{i+1}" for i in range(k)]
