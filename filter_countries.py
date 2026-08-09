import pandas as pd
from pathlib import Path

from data.undp_hdi import UNDP_HDI_COUNTRIES_193


csv_path = Path("data/output/hdi_2050_rankings.csv")
df = pd.read_csv(csv_path)

print(f"Before filtering: {len(df)} rows")

df_filtered = df[df["ISO3"].isin(UNDP_HDI_COUNTRIES_193)].copy()
df_filtered = df_filtered.drop_duplicates(subset=["ISO3"], keep="first")

missing = set(UNDP_HDI_COUNTRIES_193) - set(df_filtered["ISO3"])
if missing:
    raise RuntimeError(f"Missing UNDP HDI countries: {sorted(missing)}")

df_filtered = df_filtered.sort_values("HDI_2050", ascending=False).reset_index(drop=True)
df_filtered["Rank"] = range(1, len(df_filtered) + 1)

if len(df_filtered) != 193:
    raise RuntimeError(f"Expected 193 UNDP HDI countries, got {len(df_filtered)}")

df_filtered.to_csv(csv_path, index=False)

print(f"After filtering: {len(df_filtered)} rows")
print(f"Saved to {csv_path}")
