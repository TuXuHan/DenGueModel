"""
Quick sanity checks for processed data.

Usage:
  python -m scripts.check_data_stats
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

import config as cfg


def main():
    data_path = Path(cfg.DATA_DIR) / "all_processed_data.csv"
    print("DATA_DIR:", cfg.DATA_DIR)
    print("Loading:", data_path)
    df = pd.read_csv(data_path)
    print("Rows:", len(df))

    egg = df["egg_num"].fillna(0)
    print("\n[egg_num]")
    print("min:", egg.min(), "max:", egg.max(), "mean:", egg.mean(), "std:", egg.std())
    nz = egg[egg != 0]
    print("non-zero count:", len(nz), " / total:", len(egg))
    if len(nz) > 0:
        print("non-zero mean:", nz.mean(), "std:", nz.std())
        print("quantiles:", nz.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict())

    print("\n[coverage]")
    print("unique ids:", df["id"].nunique())
    print("unique time:", df["time"].nunique())
    dup = df.groupby(["id", "time"]).size()
    print("max dup per id-time:", dup.max())


if __name__ == "__main__":
    main()
