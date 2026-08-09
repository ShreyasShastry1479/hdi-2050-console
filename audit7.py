import pandas as pd
wb = pd.read_parquet('data/worldbank_cache.parquet')
print(f"Max year: {wb['year'].max()}")
print(f"Min year: {wb['year'].min()}")

# Check 2024 values for problem countries
wb24 = wb[wb['year'] == 2024]
for iso in ['ESP','ARE','OMN','PAK','QAT','VEN']:
    row = wb24[wb24['iso3'] == iso]
    if len(row):
        r = row.iloc[0]
        print(f"{iso}: life_exp={r.get('life_exp','N/A')}, gni={r.get('gni_ppp','N/A')}, mean_school={r.get('mean_school','N/A')}, expected_school={r.get('expected_school','N/A')}")
    else:
        print(f"{iso}: no 2024 data")
