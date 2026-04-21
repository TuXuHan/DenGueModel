import pandas as pd
df = pd.read_csv("dataset/all_processed_data_9box_nexty.csv")
min_egg = df['egg_num'].min()
print(f"Minimum Egg Count: {min_egg}")

if min_egg < 0:
    print("⚠ Found negative values! This causes the log1p error.")
    print(df[df['egg_num'] < 0].head())
else:
    print("✅ No negative values found in egg_num column.")