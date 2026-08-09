import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from data.collection import load_dataset
df = load_dataset(force_refresh=True)
print(f'Rows: {len(df)}, Countries: {df["country_id"].nunique()}')
for v in ['gini', 'infant_mortality', 'rule_of_law', 'political_stability', 'rd_expenditure', 'dependency_ratio', 'broadband', 'climate_risk']:
    if v in df.columns:
        n = df[v].notna().sum()
        print(f'  {v}: {n}/{len(df)} non-null, range [{df[v].min():.3f}, {df[v].max():.3f}]')
    else:
        print(f'  {v}: MISSING')
