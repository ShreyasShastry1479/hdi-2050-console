import pandas as pd
wb = pd.read_parquet('data/worldbank_cache.parquet')
cub = wb[wb['iso3'] == 'CUB']
print(f"Cuba rows: {len(cub)}")
print(f"Cuba GNI: {cub['gni_ppp'].dropna().values}")
print(f"Cuba life_exp: {cub['life_exp'].dropna().head(3).values} ... {cub['life_exp'].dropna().tail(3).values}")
