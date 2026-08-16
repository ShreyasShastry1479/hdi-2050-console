"""Generate scenario-style religious composition projections for the Mosaic."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import config  # noqa: E402
from data.religion_model import build_religion_table  # noqa: E402
from data.undp_hdi import UNDP_HDI_COUNTRIES_193  # noqa: E402

POP_WEIGHTS = Path("data/population_weights_2024_2050.csv")
HDI_RANKINGS = Path("data/output/hdi_2050_rankings.csv")
DEMOGRAPHIC_CONTEXT = Path("data/output/demographic_context_2050.csv")


def load_population_maps() -> tuple[dict, dict]:
    if not POP_WEIGHTS.exists():
        return {}, {}
    df = pd.read_csv(POP_WEIGHTS)
    return dict(zip(df["ISO3"], df["Population_2024"])), dict(zip(df["ISO3"], df["Population_2050"]))


def load_hdi_context() -> dict:
    if not HDI_RANKINGS.exists():
        return {}
    cols = [
        "HDI_Baseline", "HDI_2024", "HDI_2050",
        "ISO3", "EducationIndex_2025", "EducationIndex_2050",
        "IncomeIndex_2025", "IncomeIndex_2050", "Urbanization_2024", "DependencyPressure",
        "Migration_Intensity_2050",
    ]
    df = pd.read_csv(HDI_RANKINGS, usecols=lambda c: c in cols)
    context = df.set_index("ISO3").to_dict("index")
    if DEMOGRAPHIC_CONTEXT.exists():
        migration_cols = [
            "ISO3", "Birth_Replacement_Pressure_2050",
            "Europe_Migration_Response_2050", "Migration_Intensity_2050",
            "Policy_Openness", "SSA_LateMigration_DestinationResponse_2050",
            "BroadLabor_Migration_ProgramIntensity_2050",
            "SSA_SourcePoolTransition_2050", "SSA_SourcePoolCapacity_2050",
        ]
        migration = pd.read_csv(
            DEMOGRAPHIC_CONTEXT, usecols=lambda c: c in migration_cols)
        migration = migration.drop_duplicates("ISO3").set_index("ISO3")
        for iso3, values in migration.to_dict("index").items():
            context.setdefault(iso3, {}).update(values)
    return context


def main():
    pop2024, pop2050 = load_population_maps()
    missing_pop = sorted(set(UNDP_HDI_COUNTRIES_193) - set(pop2050))
    if missing_pop:
        raise RuntimeError(f"Missing population rows for: {missing_pop}")

    table = build_religion_table(pop2024, pop2050, load_hdi_context())
    sums = table.groupby("ISO3")["Share_2050_pct"].sum()
    bad = sums[(sums - 100.0).abs() > 0.2]
    if not bad.empty:
        raise RuntimeError(f"Religion shares do not sum to 100: {bad.to_dict()}")

    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / "religious_composition_2050_model.csv"
    table.to_csv(long_path, index=False)

    wide_rows = []
    for iso3, grp in table.groupby("ISO3"):
        top = grp.sort_values("Share_2050_pct", ascending=False).head(5)
        row = {"ISO3": iso3, "Country": top["Country"].iloc[0]}
        for i, (_, r) in enumerate(top.iterrows(), start=1):
            row[f"Top{i}"] = r["Group"]
            row[f"Top{i}_2050_pct"] = round(r["Share_2050_pct"], 1)
        wide_rows.append(row)
    wide_path = out_dir / "religious_composition_2050_wide.csv"
    pd.DataFrame(wide_rows).sort_values("ISO3").to_csv(wide_path, index=False)

    print("=" * 78)
    print("  RELIGIOUS COMPOSITION PROJECTIONS TO 2050 -- SCENARIO LAYER")
    print("=" * 78)
    print(f"  Long-form table : {long_path} ({len(table)} rows)")
    print(f"  Wide summary    : {wide_path} ({table['ISO3'].nunique()} countries)")
    print(f"  Median |change| : {table['Change_pp'].abs().median():.2f} pp")
    print("\n  Top projected 2050 religious populations:")
    print(table.sort_values("Pop_2050", ascending=False).head(10)[["Country", "Group", "Share_2050_pct", "Pop_2050"]].to_string(index=False))


if __name__ == "__main__":
    main()
