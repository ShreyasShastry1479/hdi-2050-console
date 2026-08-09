import pandas as pd
df = pd.read_parquet('data/worldbank_cache.parquet')
print(f'Rows: {len(df)}, Countries: {df["country_id"].nunique()}')
print(f'Years: {df["year"].min()}-{df["year"].max()}')
if "country_name" in df.columns:
    names = df["country_name"].unique()[:15]
    print(f'Sample: {list(names)}')
print(f'Life exp range: {df["life_exp"].min():.1f} - {df["life_exp"].max():.1f}')
print(f'GNI range: {df["gni_ppp"].min():.0f} - {df["gni_ppp"].max():.0f}')
