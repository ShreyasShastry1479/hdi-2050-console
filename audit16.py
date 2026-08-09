import pandas as pd
proj = pd.read_parquet('data/output/projections_baseline.parquet')
bl = proj[proj['year'] == 2050]

# Check Albania's forecast trajectory
alb = proj[(proj['country_id'] == 'ALB') & (proj['scenario'] == 'baseline')][['year', 'life_exp', 'gni_ppp', 'expected_school', 'mean_school', 'predicted_hdi']]
print("=== Albania trajectory ===")
print(alb.to_string(index=False))

# Check China, India, Nigeria trajectories
for iso in ['CHN', 'IND', 'NGA', 'BRA', 'ZAF', 'IDN']:
    row = bl[(bl['country_id'] == iso) & (bl['scenario'] == 'baseline')]
    if len(row):
        r = row.iloc[0]
        print(f"\n{r['country_name']} ({iso}): HDI={r['predicted_hdi']:.4f} LifeExp={r['life_exp']:.1f} GNI={r['gni_ppp']:.0f} ExpSch={r['expected_school']:.1f} MeanSch={r['mean_school']:.1f}")

# Check what Albania's data looks like
print("\n=== Albania raw data ===")
print(alb[['year','life_exp','gni_ppp','predicted_hdi']].to_string(index=False))

# Check some high-ranked unexpected countries
print("\n=== Top 15 ===")
top = bl.sort_values('predicted_hdi', ascending=False).head(15)
for _, r in top.iterrows():
    print(f"  {r['country_id']:5s} {r['country_name']:20s} HDI={r['predicted_hdi']:.4f} GNI={r['gni_ppp']:.0f} LifeExp={r['life_exp']:.1f}")
