import pandas as pd
import numpy as np
from data.collection import load_dataset
from src.feature_engineering import build_feature_matrix

raw = load_dataset()
df = build_feature_matrix(raw)

# Check fertility stats
for iso in ['ESP','USA','JPN','IND','NGA','ETH']:
    sub = df[df['country_id'] == iso].sort_values('year')
    fert = sub['fertility'].dropna()
    if len(fert):
        print(f"{iso}: fertility last5 = {list(fert.tail(5).values)}, mean={fert.mean():.2f}")

# Find any NaN fertility issues
print(f"\nFertility NaN count: {df['fertility'].isna().sum()}")
print(f"Fertility dtype: {df['fertility'].dtype}")
print(f"Fertility range: {df['fertility'].min():.3f} to {df['fertility'].max():.3f}")

# Check for infinities
for col in ['life_exp','expected_school','mean_school','gni_ppp','fertility']:
    inf_count = np.isinf(df[col]).sum()
    nan_count = df[col].isna().sum()
    if inf_count or nan_count:
        print(f"{col}: inf={inf_count}, nan={nan_count}")
