import pandas as pd
wb = pd.read_parquet('data/worldbank_cache.parquet')
for col in ['life_exp','gni_ppp','mean_school','expected_school','internet','fertility','urbanization','trade_openness','renewable_share','physicians','health_exp','population']:
    if col in wb.columns:
        max_yr = wb.dropna(subset=[col])['year'].max()
        min_yr = wb.dropna(subset=[col])['year'].min()
        n = wb[col].notna().sum()
        print(f"  {col:20s}: {min_yr:.0f}-{max_yr:.0f} ({n} obs)")
