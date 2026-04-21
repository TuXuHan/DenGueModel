import pandas as pd
import numpy as np

# Load your processed data
df = pd.read_csv("dataset/all_processed_data_9box_nexty.csv")

# Load your split files
def load_ids(path):
    with open(path, 'r') as f:
        return [int(line.strip()) for line in f]

train_ids = []
for i in range(8):
    train_ids.extend(load_ids(f"dataset/unlabeled_split_{i}.txt"))
    
val_ids = load_ids("dataset/unlabeled_split_8.txt")
test_ids = load_ids("dataset/unlabeled_split_9.txt")

# Filter Data
train_data = df[df['id'].isin(train_ids)]['egg_num']
val_data = df[df['id'].isin(val_ids)]['egg_num']
test_data = df[df['id'].isin(test_ids)]['egg_num']

print("=== DIAGNOSIS ===")
print(f"Total rows: {len(df)}")
print(f"Train rows: {len(train_data)}, {len(train_data)/len(df)*100:.2f}%")
print(f"Val   rows: {len(val_data)}, {len(val_data)/len(df)*100:.2f}%")
print(f"Test  rows: {len(test_data)}, {len(test_data)/len(df)*100:.2f}%")
print(f"Train Max Egg: {train_data.max()}")
print(f"Val   Max Egg: {val_data.max()}")
print(f"Test  Max Egg: {test_data.max()}")
print(f"Train Mean:    {train_data.mean():.2f}")
print(f"Val   Mean:    {val_data.mean():.2f}")
print(f"Test  Mean:    {test_data.mean():.2f}")
print(f"Train Variance:{train_data.var():.2f}")
print(f"Val   Variance:{val_data.var():.2f}")
print(f"Test  Variance:{test_data.var():.2f}")