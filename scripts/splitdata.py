import os
import argparse
import random
import numpy as np
import pandas as pd


def split_ids_randomly(unique_ids, mask_ratio=0.5, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    將全部 id 拆成兩群：
    - mask_ids: 之後 egg_num 會被 mask 掉的那 50% id
    - unmask_ids: 剩下 50% id（永遠保持原本的 egg_num）
    並且再把 mask_ids 拆成 train/val/test 三份（照 id 切）
    """
    rng = np.random.RandomState(seed)

    # 打亂全部 id
    shuffled_ids = rng.permutation(unique_ids)
    n_total = len(shuffled_ids)

    # 取前 50% 當要 mask 的那群
    n_mask = int(n_total * mask_ratio)
    mask_ids = shuffled_ids[:n_mask]
    unmask_ids = shuffled_ids[n_mask:]

    # 再把 mask_ids 拆成 train/val/test
    n_train = int(n_mask * train_ratio)
    n_val = int(n_mask * val_ratio)

    train_ids = mask_ids[:n_train]
    val_ids = mask_ids[n_train:n_train + n_val]
    test_ids = mask_ids[n_train + n_val:]

    return mask_ids, unmask_ids, train_ids, val_ids, test_ids


def main(args):
    # 讀取 Excel
    print(f"Reading Excel file: {args.input}")
    df = pd.read_csv(args.input)

    if args.id_col not in df.columns:
        raise ValueError(f"id_col '{args.id_col}' not found in columns: {df.columns.tolist()}")
    if args.target_col not in df.columns:
        raise ValueError(f"target_col '{args.target_col}' not found in columns: {df.columns.tolist()}")

    # 取得全部 id
    unique_ids = df[args.id_col].unique()
    print(f"Total unique ids: {len(unique_ids)}")

    # 隨機切 指定百分比% id 要 mask，並再切 train/val/test
    mask_ids, unmask_ids, train_ids, val_ids, test_ids = split_ids_randomly(
        unique_ids,
        mask_ratio=args.ratio,
        train_ratio=args.ratio_train,
        val_ratio=args.ratio_valid,
        seed=args.seed
    )

    print(f"IDs to mask: {len(mask_ids)}")
    print(f"  Train IDs: {len(train_ids)}, Val IDs: {len(val_ids)}, Test IDs: {len(test_ids)}")
    print(f"IDs NOT masked: {len(unmask_ids)}")

    # 分兩大塊：要 mask 的那群 / 不 mask 的那群
    df_unmasked = df[df[args.id_col].isin(unmask_ids)].copy()
    df_masked = df[df[args.id_col].isin(mask_ids)].copy()

    # 再把 df_masked 用 id 切成 train/val/test
    df_train_masked = df_masked[df_masked[args.id_col].isin(train_ids)].copy()
    df_val_masked = df_masked[df_masked[args.id_col].isin(val_ids)].copy()
    df_test_masked = df_masked[df_masked[args.id_col].isin(test_ids)].copy()

    # 做出 "data 版"（mask 掉 egg_num）以及 "label 版"（保留原始 egg_num）
    def make_data_and_label(unmasked_df, masked_df, target_col):
        """回傳 (data_df, label_df)：
        - data_df: unmasked_df (原值) + masked_df (target_col = NaN)
        - label_df: unmasked_df (原值) + masked_df (原始 target_col)
        """
        # label 版：masked_df 保留原始 target_col
        label_df = pd.concat([unmasked_df, masked_df], axis=0)

        # data 版：要把 masked_df 的 target_col 清空
        masked_no_target = masked_df.copy()
        masked_no_target[target_col] = np.nan
        data_df = pd.concat([unmasked_df, masked_no_target], axis=0)

        # 如果你有時間序或其他排序欄位，也可以在這裡 sort_values 一下
        data_df = data_df.reset_index(drop=True)
        label_df = label_df.reset_index(drop=True)

        return data_df, label_df

    # 產生 train/val/test 的 data & label
    train_data, train_label = make_data_and_label(df_unmasked, df_train_masked, args.target_col)
    val_data, val_label = make_data_and_label(df_unmasked, df_val_masked, args.target_col)
    test_data, test_label = make_data_and_label(df_unmasked, df_test_masked, args.target_col)

    # 建輸出資料夾
    os.makedirs(args.output_dir, exist_ok=True)

    # 存成 CSV（你要改成 Excel 也可以）
    train_data_path = os.path.join(args.output_dir, "train_data.csv")
    train_label_path = os.path.join(args.output_dir, "train_label.csv")
    val_data_path = os.path.join(args.output_dir, "val_data.csv")
    val_label_path = os.path.join(args.output_dir, "val_label.csv")
    test_data_path = os.path.join(args.output_dir, "test_data.csv")
    test_label_path = os.path.join(args.output_dir, "test_label.csv")

    train_data.to_csv(train_data_path, index=False)
    train_label.to_csv(train_label_path, index=False)
    val_data.to_csv(val_data_path, index=False)
    val_label.to_csv(val_label_path, index=False)
    test_data.to_csv(test_data_path, index=False)
    test_label.to_csv(test_label_path, index=False)

    print("Saved files:")
    print(f"  Train data : {train_data_path}")
    print(f"  Train label: {train_label_path}")
    print(f"  Val   data : {val_data_path}")
    print(f"  Val   label: {val_label_path}")
    print(f"  Test  data : {test_data_path}")
    print(f"  Test  label: {test_label_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split egg_num dataset with masked IDs")
    parser.add_argument("--ratio", type=float, default=0.5, help="Ratio of IDs to mask (default: 0.5)")
    parser.add_argument("--ratio_train", type=float, default=0.8, help="Ratio of training IDs within masked IDs (default: 0.8)")
    parser.add_argument("--ratio_valid", type=float, default=0.1, help="Ratio of validation IDs within masked IDs (default: 0.1)")
    parser.add_argument("--input", type=str, required=True, help="Path to input Excel file")
    parser.add_argument("--output_dir", type=str, default="splits", help="Directory to save output files")
    parser.add_argument("--id_col", type=str, default="grid_id", help="Column name for ID")
    parser.add_argument("--target_col", type=str, default="egg_num", help="Column name for target (egg_num)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--random_seed", action="store_true", help="Use a randomly generated seed (overrides --seed)")
    args = parser.parse_args()

    if args.random_seed:
        args.seed = random.SystemRandom().randint(0, 2**32 - 1)
        print(f"Generated random seed: {args.seed}")

    main(args)
