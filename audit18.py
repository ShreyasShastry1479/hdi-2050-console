import pandas as pd
proj = pd.read_parquet('data/output/projections_baseline.parquet')
for iso in ['USA', 'GBR', 'JPN', 'DEU', 'FRA', 'NOR', 'CHN', 'IND', 'ALB']:
    rows = proj[proj['country_id'] == iso].sort_values('year')
    if len(rows) == 0:
        continue
    first = rows.iloc[0]
    last = rows.iloc[-1]
    print(f"{iso}: ExpSch {first['expected_school']:.1f} -> {last['expected_school']:.1f}, MeanSch {first['mean_school']:.1f} -> {last['mean_school']:.1f}")
