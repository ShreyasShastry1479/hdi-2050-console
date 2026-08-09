import pandas as pd
import numpy as np

df = pd.read_parquet('data/output/projections_baseline.parquet')

# Check 2050 data
df2050 = df[df['year'] == 2050]
print("=== Countries with HDI <= 0.1 in 2050 ===")
bad = df2050[df2050['predicted_hdi'] <= 0.1][['country_id','country_name','predicted_hdi','life_exp','expected_school','mean_school','gni_ppp']]
print(bad.to_string())

print("\n=== Countries with HDI > 0.95 in 2050 (likely too high) ===")
high = df2050[df2050['predicted_hdi'] > 0.95][['country_id','country_name','predicted_hdi','life_exp','gni_ppp']]
print(high.to_string())

print("\n=== Current 2025 values for known-good countries ===")
df2025 = df[df['year'] == 2025]
for iso in ['ESP','USA','JPN','DEU','GBR','FRA','IND','CHN','BRA']:
    row = df2025[df2025['country_id'] == iso]
    if len(row):
        r = row.iloc[0]
        print(f"  {iso} ({r['country_name']}): HDI={r['predicted_hdi']:.4f}, LifeExp={r['life_exp']:.1f}, GNI={r['gni_ppp']:.0f}")

print("\n=== WB data coverage for zero-HDI countries ===")
wb = pd.read_parquet('data/worldbank_cache.parquet')
for iso in ['ESP','ARE','OMN','PAK','QAT','VEN']:
    sub = wb[wb['iso3'] == iso]
    if len(sub):
        cols_with_data = [c for c in ['life_exp','gni_ppp','mean_school','expected_school'] if sub[c].notna().any()]
        print(f"  {iso}: {len(sub)} rows, cols with data: {cols_with_data}")
    else:
        print(f"  {iso}: NO DATA in WB cache")
