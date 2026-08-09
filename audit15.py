import pandas as pd

proj = pd.read_parquet('data/output/projections_baseline.parquet')
print("Columns:", list(proj.columns))
print(proj.head(2))
bl = proj[proj['year'] == 2050].copy()
hdi_col = [c for c in bl.columns if 'hdi' in c.lower()]
print("HDI columns:", hdi_col)
bl = bl[['country_id', 'country_name', hdi_col[0]]].sort_values(hdi_col[0], ascending=False)

# Find where country_name == country_id (meaning name wasn't resolved)
bad = bl[bl['country_name'] == bl['country_id']]
print("\n=== Country names showing ISO3 codes ===")
for _, r in bad.iterrows():
    print(f"  {r['country_id']}: {r['country_name']}")

print(f"\nTotal: {len(bad)} out of {len(bl)} countries")

# Also check which are just short (<=3 chars)
bad2 = bl[bl['country_name'].str.len() <= 3]
print(f"\nShort names (<=3 chars): {len(bad2)}")
for _, r in bad2.iterrows():
    print(f"  {r['country_id']}: {r['country_name']}")
