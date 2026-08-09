import pandas as pd
from data.collection import load_dataset
from src.feature_engineering import build_feature_matrix

raw = load_dataset()
df = build_feature_matrix(raw)

# Check Albania, Ukraine, Liechtenstein, Luxembourg
for iso in ['ALB','UKR','LIE','LUX','SGP','IRL','NOR','USA','GBR']:
    row = df[(df['country_id'] == iso) & (df['year'] == 2024)]
    if len(row):
        r = row.iloc[0]
        print(f"{iso}: life_exp={r['life_exp']:.1f}, exp_school={r['expected_school']:.2f}, mean_school={r['mean_school']:.2f}, gni={r['gni_ppp']:.0f}")
    else:
        print(f"{iso}: no 2024 data")

# Check what WB has for LIE, LUX, SGP, IRL
wb = pd.read_parquet('data/worldbank_cache.parquet')
for iso in ['LIE','LUX','SGP','IRL']:
    sub = wb[wb['iso3'] == iso]
    if len(sub):
        latest = sub[sub['year'] == sub['year'].max()]
        print(f"\n{iso} WB latest ({latest['year'].values[0]}): gni={latest['gni_ppp'].values[0] if 'gni_ppp' in latest else 'N/A'}, life_exp={latest['life_exp'].values[0] if 'life_exp' in latest else 'N/A'}")
    else:
        print(f"\n{iso}: NOT IN WB DATA")
