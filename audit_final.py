import pandas as pd
df = pd.read_parquet('data/output/projections_baseline.parquet')
latest = df[df['year']==2050][['country_id','country_name','life_exp','expected_school','mean_school','gni_ppp','predicted_hdi']].sort_values('predicted_hdi', ascending=False)

targets = ['CHN','IND','JPN','KOR','BRA','IDN','VNM','TUR','MEX','NGA','PAK','AFG','SYR','YEM','SOM','SSD','SDN','PRK','UKR','RUS','ARG','ZAF','EGY','IRN','CUB','VEN','MMR','ETH','COD']
print('=== KEY COUNTRIES ===')
for iso in targets:
    r = latest[latest['country_id']==iso]
    if len(r):
        r = r.iloc[0]
        print(f"  {iso:4s} {r['country_name']:20s} HDI={r['predicted_hdi']:.4f}  LE={r['life_exp']:.1f}  GNI={r['gni_ppp']:>10,.0f}  ES={r['expected_school']:.1f}  MS={r['mean_school']:.1f}")

print('\n=== TOP 10 ===')
for _, r in latest.head(10).iterrows():
    print(f"  {r['country_id']:4s} {r['country_name']:20s} HDI={r['predicted_hdi']:.4f}")

print('\n=== BOTTOM 15 ===')
for _, r in latest.tail(15).iterrows():
    print(f"  {r['country_id']:4s} {r['country_name']:20s} HDI={r['predicted_hdi']:.4f}  LE={r['life_exp']:.1f}  GNI={r['gni_ppp']:>8,.0f}")
