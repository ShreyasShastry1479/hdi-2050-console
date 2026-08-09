import pandas as pd
df = pd.read_parquet('data/output/projections_baseline.parquet')
df2050 = df[df['year']==2050].copy()
df2050 = df2050.sort_values('predicted_hdi', ascending=False).reset_index(drop=True)
df2050.index += 1

print('=== TOP 40 (2050) ===')
cols = ['country_id','country_name','predicted_hdi','life_exp','expected_school','mean_school','gni_ppp']
print(df2050[cols].head(40).to_string())

print()
print('=== KEY COUNTRIES ===')
for cid in ['CHN','IND','ISR','LUX','UKR','RUS','ARG','FRO','LIE','MCO','IMN','GIB','CUB','VEN','MMR','ETH','COD','BGD','VNM','IDN','PHL','TUR','POL','EST','CZE','LVA','BGR','MYS','KAZ','SAU','BHR','PRT','ESP','ITA']:
    row = df2050[df2050['country_id']==cid]
    if not row.empty:
        r = row.iloc[0]
        rank = df2050[df2050['country_id']==cid].index[0]
        print(f'{cid}: #{rank} HDI={r["predicted_hdi"]:.3f} LE={r["life_exp"]:.1f} GNI=${r["gni_ppp"]:,.0f} ES={r["expected_school"]:.1f} MS={r["mean_school"]:.1f}')
    else:
        print(f'{cid}: MISSING')
