import pandas as pd
from data.collection import _fetch_all_from_wb, _clean_wb_data
from data.reference import REFERENCE_HDI_2024

wb = pd.read_parquet('data/worldbank_cache.parquet')
df = _clean_wb_data(wb)

# Check Liechtenstein 2024
lie = df[(df['country_id'] == 'LIE') & (df['year'] == 2024)]
if len(lie):
    print(f"LIE 2024: gni={lie.iloc[0]['gni_ppp']:.0f}, exp_school={lie.iloc[0]['expected_school']:.2f}, mean_school={lie.iloc[0]['mean_school']:.2f}")
else:
    print("LIE: no 2024 row in cleaned data")

# Check if LIE is in reference
print(f"LIE in reference: {'LIE' in REFERENCE_HDI_2024}")
print(f"REF LIE: {REFERENCE_HDI_2024.get('LIE', {})}")

# Check LIE in raw WB
lie_raw = wb[wb['iso3'] == 'LIE']
print(f"LIE raw WB rows: {len(lie_raw)}")
if len(lie_raw):
    print(f"  years: {sorted(lie_raw['year'].unique())}")
    print(f"  gni values: {lie_raw['gni_ppp'].dropna().values}")
