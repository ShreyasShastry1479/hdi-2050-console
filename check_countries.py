import pandas as pd

from data.undp_hdi import UNDP_HDI_COUNTRIES_193


df = pd.read_csv("data/output/hdi_2050_rankings.csv")
print(f"Total rows: {len(df)}")
print(f"Unique ISO3: {df['ISO3'].nunique()}")

duplicates = df[df.duplicated(subset=["ISO3"], keep=False)]
if not duplicates.empty:
    raise RuntimeError(f"Duplicate ISO3 rows found:\n{duplicates[['ISO3', 'Country']]}")

missing = set(UNDP_HDI_COUNTRIES_193) - set(df["ISO3"])
extra = set(df["ISO3"]) - set(UNDP_HDI_COUNTRIES_193)
if missing or extra:
    raise RuntimeError(f"Country universe mismatch. Missing={sorted(missing)} Extra={sorted(extra)}")

if len(df) != 193:
    raise RuntimeError(f"Expected 193 UNDP HDI countries, got {len(df)}")

print("Country universe matches the 193 UNDP HDI countries/territories.")
