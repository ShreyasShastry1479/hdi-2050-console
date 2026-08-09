import pandas as pd
ds = pd.read_parquet('data/full_dataset.parquet')
usa = ds[ds['country_id'] == 'USA'].sort_values('year')
print("USA expected_school trajectory:")
print(usa[['year', 'expected_school', 'mean_school', 'secondary_enroll', 'tertiary_enroll']].to_string(index=False))
print()
gbr = ds[ds['country_id'] == 'GBR'].sort_values('year')
print("GBR expected_school trajectory:")
print(gbr[['year', 'expected_school', 'mean_school', 'secondary_enroll', 'tertiary_enroll']].to_string(index=False))
