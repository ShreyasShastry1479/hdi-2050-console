import pandas as pd
from data.collection import load_dataset
from src.feature_engineering import build_feature_matrix

raw = load_dataset()
df = build_feature_matrix(raw)

# Check expected_school for major countries in 2024
for iso in ['USA','GBR','DEU','JPN','FRA','CHN','IND','BRA','ESP','NOR','SWE','ITA','NGA','PAK']:
    row = df[(df['country_id'] == iso) & (df['year'] == 2024)]
    if len(row):
        r = row.iloc[0]
        print(f"{iso}: exp_school={r['expected_school']:.2f}, mean_school={r['mean_school']:.2f}, life_exp={r['life_exp']:.1f}, gni={r['gni_ppp']:.0f}")

# How many countries have the default exp_school=10.2?
df24 = df[df['year'] == 2024]
n_default = len(df24[abs(df24['expected_school'] - 10.2) < 0.01])
print(f"\nCountries with exp_school=10.2 in 2024: {n_default}/{len(df24)}")
