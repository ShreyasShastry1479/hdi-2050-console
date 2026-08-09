import pandas as pd
proj = pd.read_parquet('data/output/projections_baseline.parquet')
print(list(proj.columns))
for iso in ['CHN', 'IND', 'NGA', 'BRA']:
    sub = proj[(proj['archetype'].isin(['lower_middle','upper_middle','high_development'])) | True]
    rows = proj[proj['country_id'] == iso].sort_values('year')
    if len(rows) == 0:
        continue
    print(f"\n=== {iso} ===")
    print(rows[['year', 'life_exp', 'gni_ppp', 'predicted_hdi']].iloc[[0,1,2,-3,-2,-1]].to_string(index=False))
