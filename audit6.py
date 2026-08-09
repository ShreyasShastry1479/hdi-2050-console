import pandas as pd
wb = pd.read_parquet('data/worldbank_cache.parquet')

# Check 2025 values specifically
wb25 = wb[wb['year'] == 2025]
print(f"2025 rows: {len(wb25)}")
for col in ['life_exp','gni_ppp','mean_school','expected_school','fertility']:
    vals = wb25[col].dropna()
    print(f"  {col}: {len(vals)} non-NaN, unique values: {len(vals.unique())}, median={vals.median():.2f}")
    # Show some examples
    for iso in ['ESP','USA','NGA','IND']:
        row = wb25[wb25['iso3'] == iso]
        if len(row):
            print(f"    {iso}: {row[col].values[0]}")

# Check 2024 values
wb24 = wb[wb['year'] == 2024]
print(f"\n2024 rows: {len(wb24)}")
for col in ['life_exp','gni_ppp','fertility']:
    vals = wb24[col].dropna()
    print(f"  {col}: {len(vals)} non-NaN, unique: {len(vals.unique())}")
    for iso in ['ESP','USA','NGA']:
        row = wb24[wb24['iso3'] == iso]
        if len(row):
            print(f"    {iso}: {row[col].values[0]:.2f}")
