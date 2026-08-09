"""Ethnic composition projections to 2050 for the 193 UNDP countries.

Standalone algorithm built on data/ethnicity.py. Reads total population
projections (UN WPP via data/population_weights_2024_2050.csv) to derive
absolute group sizes, runs the differential-fertility / migration /
assimilation projection, and writes:

    data/output/ethnic_composition_2050.csv      (long form, all groups)
    data/output/ethnic_composition_2050_wide.csv (top groups per country)
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import config  # noqa: E402
from data.ethnicity import (  # noqa: E402
    ETHNIC_COMPOSITION_2024,
    build_ethnic_composition_table,
    project_ethnic_composition,
)
from data.undp_hdi import UNDP_HDI_COUNTRIES_193  # noqa: E402

POP_WEIGHTS = Path("data/population_weights_2024_2050.csv")
MIGRATION_SCENARIO = "baseline"
FERTILITY_CONVERGENCE = 0.5


def load_population_maps() -> tuple[dict, dict]:
    """Return ({iso3: population_2024}, {iso3: population_2050})."""
    if not POP_WEIGHTS.exists():
        return {}, {}
    df = pd.read_csv(POP_WEIGHTS)
    pop2024 = dict(zip(df["ISO3"], df["Population_2024"]))
    pop2050 = dict(zip(df["ISO3"], df["Population_2050"]))
    return pop2024, pop2050


def main():
    pop2024, pop2050 = load_population_maps()

    print("=" * 72)
    print("  ETHNIC COMPOSITION PROJECTIONS TO 2050 (193 UNDP countries)")
    print("  Model: differential fertility + migration + assimilation")
    print(f"  Migration scenario: {MIGRATION_SCENARIO} | "
          f"fertility convergence: {FERTILITY_CONVERGENCE}")
    print("=" * 72)

    # Scope strictly to the 193 UNDP HDI countries
    extra = set(ETHNIC_COMPOSITION_2024) - set(UNDP_HDI_COUNTRIES_193)
    if extra:
        print(f"  Note: {sorted(extra)} excluded (not in UNDP-193 universe)")

    table = build_ethnic_composition_table(
        population_2050_by_country=pop2050,
        population_2024_by_country=pop2024,
        migration_scenario=MIGRATION_SCENARIO,
        fertility_convergence=FERTILITY_CONVERGENCE,
    )
    table = table[table["ISO3"].isin(UNDP_HDI_COUNTRIES_193)].copy()

    missing = set(UNDP_HDI_COUNTRIES_193) - set(table["ISO3"])
    if missing:
        raise RuntimeError(f"Missing ethnic composition for: {sorted(missing)}")

    # Validate shares sum to 100 within each country/year
    # (tolerance allows for per-group rounding to 2 decimals)
    for col in ["Share_2024_pct", "Share_2050_pct"]:
        sums = table.groupby("ISO3")[col].sum()
        bad = sums[abs(sums - 100.0) > 0.1]
        if not bad.empty:
            raise RuntimeError(f"{col} not summing to 100 for: {bad.to_dict()}")

    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / "ethnic_composition_2050.csv"
    table.to_csv(long_path, index=False)

    # Wide summary: top 5 groups per country, sorted by 2050 share
    wide_rows = []
    for iso3, grp in table.groupby("ISO3"):
        top = grp.sort_values("Share_2050_pct", ascending=False).head(5)
        row = {"ISO3": iso3, "Country": top["Country"].iloc[0]}
        for i, (_, r) in enumerate(top.iterrows(), start=1):
            row[f"Top{i}"] = r["Group"]
            row[f"Top{i}_2050_pct"] = round(r["Share_2050_pct"], 1)
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows).sort_values("ISO3")
    wide_path = out_dir / "ethnic_composition_2050_wide.csv"
    wide.to_csv(wide_path, index=False)

    print(f"\n  Long-form table : {long_path}  ({len(table)} rows)")
    print(f"  Wide summary    : {wide_path}  ({len(wide)} countries)")

    # ---------------- Audit ----------------
    print("\n  === AUDIT ===")
    n_unchanged = (table.groupby("ISO3")["Change_pp"].sum().abs() < 1e-6).sum()
    print(f"  Countries projected        : {table['ISO3'].nunique()}")
    print(f"  Ethnic groups in table     : {len(table)}")
    print(f"  Median |change| per group  : {table['Change_pp'].abs().median():.2f} pp")
    print(f"  Mean |change| per group    : {table['Change_pp'].abs().mean():.2f} pp")
    print(f"  Most gainers / losers      : see summary below")

    anchor_change = table.groupby("ISO3").apply(
        lambda g: g.loc[g["Anchor"], "Change_pp"].sum(), include_groups=False
    )
    print(f"\n  Top 10 anchor-group share declines (majority shrinking):")
    print(anchor_change.nsmallest(10).to_string())

    print(f"\n  Top 10 anchor-group share gains (majority consolidating):")
    print(anchor_change.nlargest(10).to_string())

    print(f"\n  Largest minority gains (pp, 2024->2050):")
    biggest = table.sort_values("Change_pp", ascending=False).head(12)[
        ["ISO3", "Country", "Group", "Share_2024_pct", "Share_2050_pct", "Change_pp"]
    ]
    print(biggest.to_string(index=False))

    print(f"\n  Largest minority losses (pp, 2024->2050):")
    smallest = table.sort_values("Change_pp").head(12)[
        ["ISO3", "Country", "Group", "Share_2024_pct", "Share_2050_pct", "Change_pp"]
    ]
    print(smallest.to_string(index=False))

    print(f"\n  Population totals (2030-check): "
          f"sum(Pop_2050) = {table['Pop_2050'].sum():,.0f} "
          f"(world 2050 ~9.7bn, includes only listed groups)")
    print("  Done.")


if __name__ == "__main__":
    main()
