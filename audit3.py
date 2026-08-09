import pandas as pd
import numpy as np

wb = pd.read_parquet('data/worldbank_cache.parquet')

# Check actual year values
for iso in ['ESP','NRU','GUY']:
    sub = wb[wb['iso3'] == iso].sort_values('year')
    print(f"\n{iso} years: {sorted(sub['year'].unique())[:5]} ... {sorted(sub['year'].unique())[-5:]}")
    print(f"  life_exp: {sub[['year','life_exp']].dropna().tail(3).to_string(index=False)}")
    print(f"  gni_ppp: {sub[['year','gni_ppp']].dropna().tail(3).to_string(index=False)}")
    print(f"  mean_school: {sub[['year','mean_school']].dropna().tail(3).to_string(index=False)}")
    print(f"  expected_school: {sub[['year','expected_school']].dropna().tail(3).to_string(index=False)}")

# After feature engineering
from data.collection import load_dataset
from src.feature_engineering import build_feature_matrix

raw = load_dataset()
df = build_feature_matrix(raw)

# Check 2025 values for problem countries
for iso in ['ESP','ARE','OMN','PAK','QAT','VEN','NRU','GUY']:
    row = df[(df['country_id'] == iso) & (df['year'] == 2025)]
    if len(row):
        r = row.iloc[0]
        print(f"\n{iso} 2025 after FE: life_exp={r['life_exp']:.1f}, expected_school={r['expected_school']:.1f}, mean_school={r['mean_school']:.1f}, gni_ppp={r['gni_ppp']:.0f}")
    else:
        print(f"\n{iso}: no 2025 row")
