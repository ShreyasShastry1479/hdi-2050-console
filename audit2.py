import pandas as pd
import numpy as np

wb = pd.read_parquet('data/worldbank_cache.parquet')
for iso in ['ESP','ARE','OMN','PAK','QAT','VEN','NRU','GUY']:
    sub = wb[wb['iso3'] == iso].sort_values('year')
    if len(sub):
        le = sub['life_exp'].dropna()
        gni = sub['gni_ppp'].dropna()
        ms = sub['mean_school'].dropna()
        es = sub['expected_school'].dropna()
        print(f"{iso}: life_exp last={le.iloc[-1]:.1f} ({le.index[0]}-{le.index[-1]}), "
              f"gni last={gni.iloc[-1]:.0f}, "
              f"mean_school last={ms.iloc[-1] if len(ms) else 'N/A'}, "
              f"expected_school last={es.iloc[-1] if len(es) else 'N/A'}")
    else:
        print(f"{iso}: NO DATA")

# Check which countries have the most missing data
print("\n=== Countries with < 10 years of life_exp data ===")
counts = wb.groupby('iso3')['life_exp'].apply(lambda x: x.notna().sum())
sparse = counts[counts < 10].sort_values()
for iso, n in sparse.items():
    name = wb[wb['iso3']==iso]['country_name'].iloc[0] if 'country_name' in wb.columns else iso
    print(f"  {iso} ({name}): {n} years")
