import pandas as pd
ds = pd.read_parquet('data/full_dataset.parquet')
for iso in ['CHN', 'IND', 'BRA', 'IDN', 'MEX', 'TUR', 'THA', 'VNM', 'PHL']:
    c = ds[ds['country_id'] == iso].sort_values('year')
    last = c.iloc[-1]
    print(f"{iso}: GNI={last['gni_ppp']:.0f} LifeExp={last['life_exp']:.1f} ExpSch={last['expected_school']:.1f} MeanSch={last['mean_school']:.1f} SecEnroll={last.get('secondary_enroll', 0):.0f} TerEnroll={last.get('tertiary_enroll', 0):.0f}")
