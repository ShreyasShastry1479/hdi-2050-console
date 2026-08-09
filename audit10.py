import pandas as pd
wb = pd.read_parquet('data/worldbank_cache.parquet')

# Check enrollment coverage
for ind in ['secondary_enroll', 'tertiary_enroll']:
    if ind in wb.columns:
        coverage = wb.groupby('iso3')[ind].apply(lambda x: x.notna().sum())
        n_countries = len(coverage[coverage > 0])
        n_zero = len(coverage[coverage == 0])
        print(f"{ind}: {n_countries} countries with data, {n_zero} without")
        # Show which major countries are missing
        missing = coverage[coverage == 0].index
        major = ['USA','GBR','DEU','JPN','FRA','CHN','IND','BRA','ESP','NOR','SWE','ITA']
        missing_major = [c for c in major if c in missing]
        print(f"  Major countries without: {missing_major}")

# Check which countries have GNI floor (400)
wb_latest = wb.sort_values('year').groupby('iso3').last()
low_gni = wb_latest[wb_latest['gni_ppp'] < 1000][['gni_ppp','life_exp']]
print(f"\nCountries with GNI < 1000 in latest year: {len(low_gni)}")
for iso, row in low_gni.head(10).iterrows():
    print(f"  {iso}: GNI={row['gni_ppp']:.0f}")
